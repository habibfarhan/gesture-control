"""Use the hand as a mouse.

Pose to enter cursor mode: thumb and index open, middle/ring/pinky closed.
While in cursor mode the wrist (point 0) moves the mouse relatively (like a
trackpad). Closing the index finger holds the left button down and opening
it lets go, so a quick close is a click and moving while closed is a drag.
The thumb does the same for the right button. Opening middle/ring/pinky, or
losing the hand, leaves the mode and lets go of any held button.
"""

import math
from collections import deque

from evdev import UInput, ecodes as e
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark as L

from fingers import PoseTimer

SENSITIVITY = 3500        # screen pixels per full camera width of wrist movement
LOST_FRAMES = 5           # frames without the hand before leaving cursor mode
CLICK_REWIND_FRAMES = 5   # movement to undo on click (the finger dips as it closes)
CLICK_DEADZONE = 0.012    # wrist must move this far (fraction of camera width)
                          # from where it was at a press/release before the
                          # cursor follows again; absorbs clicking jitter
CLICK_SETTLE_TIME = 0.3   # seconds after a release before the lock lifts anyway


class VirtualMouse:
    """A kernel-level virtual mouse, so it works on Wayland as well as X11."""

    def __init__(self):
        self._ui = UInput(
            {e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT], e.EV_REL: [e.REL_X, e.REL_Y]},
            name="gesture-control-mouse",
        )

    def move(self, dx, dy):
        if dx:
            self._ui.write(e.EV_REL, e.REL_X, dx)
        if dy:
            self._ui.write(e.EV_REL, e.REL_Y, dy)
        self._ui.syn()

    def press(self, button):
        self._ui.write(e.EV_KEY, button, 1)
        self._ui.syn()

    def release(self, button):
        self._ui.write(e.EV_KEY, button, 0)
        self._ui.syn()

    def close(self):
        self._ui.close()


class OneEuroFilter:
    """Smooths heavily when the finger is still, lightly when it moves fast."""

    def __init__(self, min_cutoff=1.5, beta=5.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = self._dx = self._t = None

    def __call__(self, x, t):
        if self._t is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = max(t - self._t, 1e-6)
        self._t = t
        dx = (x - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        return self._x

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


def in_cursor_pose(states):
    return (not states["thumb"] and not states["index"]
            and states["middle"] and states["ring"] and states["pinky"])


class CursorController:
    def __init__(self, mouse, frame_aspect):
        """`frame_aspect` is camera height / width, so vertical and horizontal
        hand movement map to the same pixels per centimetre."""
        self.mouse = mouse
        self.frame_aspect = frame_aspect
        self.active = False
        self.hand = None
        self._filter_x = OneEuroFilter()
        self._filter_y = OneEuroFilter()
        self._last = None
        self._remainder = [0.0, 0.0]
        self._recent = deque(maxlen=CLICK_REWIND_FRAMES)
        self._missing = 0
        self._held = set()
        self._pose_timer = PoseTimer(in_cursor_pose)
        self._locked = False
        self._lock_pos = None
        self._lock_until = None

    def update(self, seen, t):
        """`seen` is the output of `StableFingers.update` for this frame."""
        held_hand = self._pose_timer.update(seen, t)
        if not self.active:
            if held_hand:
                self._enter(held_hand)
            return

        if self.hand not in seen:
            self._missing += 1
            if self._missing >= LOST_FRAMES:
                self._exit()
            # Keep any held button during a brief dropout so a drag survives.
            self._stop_tracking()
            return
        self._missing = 0

        landmarks, prev, states = seen[self.hand]
        if not (states["middle"] and states["ring"] and states["pinky"]):
            self._exit()
            return

        self._set_button(e.BTN_LEFT, states["index"], prev["index"], t)
        self._set_button(e.BTN_RIGHT, states["thumb"], prev["thumb"], t)

        # Track the wrist, not the index tip: the tip dips when clicking.
        self._move(landmarks[L.WRIST], t)

    def _set_button(self, button, closed, was_closed, t):
        if closed and was_closed is False and button not in self._held:
            self._rewind()
            self.mouse.press(button)
            self._held.add(button)
            # Held: stay locked until the hand deliberately moves (a drag).
            self._lock(until=None)
        elif not closed and button in self._held:
            self.mouse.release(button)
            self._held.discard(button)
            # Opening the finger jitters the wrist too; lock briefly.
            self._lock(until=t + CLICK_SETTLE_TIME)

    def _lock(self, until):
        """Pin the cursor where it is. `_move` lifts the lock once the wrist
        leaves the deadzone around this spot, or after `until` (if given)."""
        self._locked = True
        self._lock_pos = None  # filled in with this frame's wrist position
        self._lock_until = until

    def _unlock(self):
        self._locked = False
        self._lock_pos = None
        self._lock_until = None

    def release_all(self):
        for button in self._held:
            self.mouse.release(button)
        self._held.clear()

    def _move(self, point, t):
        x = self._filter_x(point.x, t)
        y = self._filter_y(point.y, t)
        if self._last is None:
            self._last = (x, y)
            return
        if self._locked:
            if self._lock_pos is None:
                self._lock_pos = (x, y)
            dist = math.hypot(x - self._lock_pos[0],
                              (y - self._lock_pos[1]) * self.frame_aspect)
            expired = self._lock_until is not None and t >= self._lock_until
            if dist < CLICK_DEADZONE and not expired:
                # Follow the wrist silently so leaving the lock doesn't jump.
                self._last = (x, y)
                self._remainder = [0.0, 0.0]
                return
            self._unlock()
        dx = (x - self._last[0]) * SENSITIVITY + self._remainder[0]
        dy = (y - self._last[1]) * SENSITIVITY * self.frame_aspect + self._remainder[1]
        self._last = (x, y)
        ix, iy = int(dx), int(dy)
        self._remainder = [dx - ix, dy - iy]
        if ix or iy:
            self.mouse.move(ix, iy)
            self._recent.append((ix, iy))

    def _rewind(self):
        """Undo the last few frames of movement, which were the finger starting
        to close rather than the user aiming."""
        dx = -sum(m[0] for m in self._recent)
        dy = -sum(m[1] for m in self._recent)
        if dx or dy:
            self.mouse.move(dx, dy)
        self._recent.clear()

    def _stop_tracking(self):
        # Next movement starts fresh from wherever the hand is then, so the
        # cursor doesn't jump.
        self._last = None
        self._remainder = [0.0, 0.0]
        self._filter_x.reset()
        self._filter_y.reset()
        self._recent.clear()

    def _enter(self, hand):
        self.active = True
        self.hand = hand
        self._missing = 0
        self._stop_tracking()

    def _exit(self):
        self.release_all()
        self._unlock()
        self.active = False
        self.hand = None
        self._stop_tracking()
