"""Open/closed checks for each finger, independent of how the hand is turned.

`landmarks` are MediaPipe's 21 hand *world* landmarks (3D, in metres). Each
check measures points along the hand's own axes instead of the screen's:

- "up" runs from the wrist (0) to the base of the middle finger (9)
- "across" runs from the base of the pinky (17) to the base of the index (5)

So "the tip is below the joint" means below it along the hand, whichever way
the hand is rotated. Because "across" always points toward the thumb side,
the thumb check works for either hand, palm or back facing the camera.
"""

import math

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark as L


def _vec(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def _hand_axes(landmarks):
    up = _vec(landmarks[L.WRIST], landmarks[L.MIDDLE_FINGER_MCP])
    across = _vec(landmarks[L.PINKY_MCP], landmarks[L.INDEX_FINGER_MCP])
    # Make "across" exactly perpendicular to "up".
    k = _dot(across, up) / _dot(up, up)
    across = tuple(a - k * u for a, u in zip(across, up))
    return up, across


def _tip_below_dip(landmarks, tip, dip):
    up, _ = _hand_axes(landmarks)
    return _dot(_vec(landmarks[dip], landmarks[tip]), up) < 0


def is_index_closed(landmarks):
    return _tip_below_dip(landmarks, L.INDEX_FINGER_TIP, L.INDEX_FINGER_DIP)


def is_middle_closed(landmarks):
    return _tip_below_dip(landmarks, L.MIDDLE_FINGER_TIP, L.MIDDLE_FINGER_DIP)


def is_ring_closed(landmarks):
    return _tip_below_dip(landmarks, L.RING_FINGER_TIP, L.RING_FINGER_DIP)


def is_pinky_closed(landmarks):
    return _tip_below_dip(landmarks, L.PINKY_TIP, L.PINKY_DIP)


def is_thumb_closed(landmarks):
    # Open, the thumb tip sticks out past the joint below it (toward the thumb
    # side); closed, it folds back across the palm toward the pinky.
    _, across = _hand_axes(landmarks)
    return _dot(_vec(landmarks[L.THUMB_IP], landmarks[L.THUMB_TIP]), across) < 0


def finger_states(landmarks):
    """Return {finger name: closed?} for all five fingers of one hand."""
    return {
        "thumb": is_thumb_closed(landmarks),
        "index": is_index_closed(landmarks),
        "middle": is_middle_closed(landmarks),
        "ring": is_ring_closed(landmarks),
        "pinky": is_pinky_closed(landmarks),
    }


CONFIRM_FRAMES = 2  # frames a finger state must hold before it counts
MODE_ENTER_S = 0.25  # a mode's pose must be held this long before it turns on,
                     # so passing through it (e.g. while making a fist) doesn't
MODE_ENTER_MAX_TRAVEL = 0.08  # ...and held still: moving the wrist further than
                              # this (fraction of the frame) restarts the timer,
                              # so poses passed through mid-swipe don't count
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class Debounced:
    """A boolean that only changes after the new value holds for a few frames."""

    def __init__(self, frames):
        self.frames = frames
        self.reset()

    def reset(self):
        self.value = None
        self._candidate = None
        self._count = 0

    def update(self, raw):
        if self.value is None or raw == self.value:
            self.value = raw
            self._count = 0
        elif raw == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = raw, 1
        if self._count >= self.frames:
            self.value = raw
            self._count = 0
        return self.value


class StableFingers:
    """Debounces finger states per hand so brief tracking flickers are ignored."""

    def __init__(self):
        self._fingers = {
            hand: {f: Debounced(CONFIRM_FRAMES) for f in FINGERS}
            for hand in ("Left", "Right")
        }

    def update(self, hands):
        """`hands` is a list of (landmarks, hand label, raw finger states).

        Returns {hand label: (landmarks, previous states, current states)}.
        """
        seen = {}
        for landmarks, hand, raw in hands:
            if hand in seen:  # MediaPipe occasionally labels both hands the same
                continue
            fingers = self._fingers[hand]
            prev = {f: fingers[f].value for f in FINGERS}
            states = {f: fingers[f].update(raw[f]) for f in FINGERS}
            seen[hand] = (landmarks, prev, states)
        # Forget hands that left the frame so they can't fire a stale click.
        for hand, fingers in self._fingers.items():
            if hand not in seen:
                for d in fingers.values():
                    d.reset()
        return seen


class PoseTimer:
    """Tracks how long each hand has continuously, and steadily, held a pose."""

    def __init__(self, in_pose):
        self._in_pose = in_pose
        self._since = {}  # hand -> (start time, wrist x, wrist y)

    def update(self, seen, t):
        """Call every frame. Returns a hand that has held the pose still for
        `MODE_ENTER_S`, or None."""
        for hand in list(self._since):
            if hand not in seen:
                del self._since[hand]
        for hand, (landmarks, _, states) in seen.items():
            if not self._in_pose(states):
                self._since.pop(hand, None)
                continue
            wrist = landmarks[0]
            start = self._since.get(hand)
            if start is None or math.hypot(wrist.x - start[1], wrist.y - start[2]) > MODE_ENTER_MAX_TRAVEL:
                self._since[hand] = (t, wrist.x, wrist.y)
        for hand, (since, _, _) in self._since.items():
            if t - since >= MODE_ENTER_S:
                return hand
        return None
