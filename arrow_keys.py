"""Hold an arrow key by pointing the thumb sideways.

Pose: thumb open, index/middle/ring/pinky closed (a hitchhiker thumb).
Thumb pointing right on screen holds the right arrow; pointing left holds the
left arrow. Pointing up or down holds nothing. Breaking the pose, or losing
the hand, releases the key.
"""

from evdev import ecodes as e
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark as L

from cursor_control import LOST_FRAMES
from fingers import PoseTimer

# The thumb counts as pointing sideways once its horizontal extent is this many
# times its vertical extent (1.0 = 45 degrees). The lower value keeps a held key
# held, so a thumb hovering near the boundary doesn't flicker the key.
START_RATIO = 1.3
KEEP_RATIO = 0.8


def in_arrow_pose(states):
    return (not states["thumb"] and states["index"] and states["middle"]
            and states["ring"] and states["pinky"])


class ArrowKeys:
    def __init__(self, keyboard, frame_aspect):
        self._keyboard = keyboard
        self.frame_aspect = frame_aspect
        self.hand = None
        self.held = None  # KEY_LEFT / KEY_RIGHT currently held, or None
        self._missing = 0
        self._pose_timer = PoseTimer(in_arrow_pose)

    @property
    def active(self):
        return self.hand is not None

    def update(self, seen, t, blocked=False):
        """`seen` is the output of `StableFingers.update` for this frame.
        `blocked` is true while another mode (e.g. a cursor drag, whose closed
        index finger makes this same pose) owns the hand."""
        held_hand = self._pose_timer.update(seen, t)
        if blocked:
            self._exit()
            return
        if not self.active:
            if held_hand:
                self.hand = held_hand
                self._missing = 0
            else:
                return

        if self.hand not in seen:
            self._missing += 1
            if self._missing >= LOST_FRAMES:
                self._exit()
            return
        self._missing = 0

        landmarks, _, states = seen[self.hand]
        if not in_arrow_pose(states):
            self._exit()
            return
        self._hold(self._direction(landmarks))

    def _direction(self, landmarks):
        base, tip = landmarks[L.THUMB_MCP], landmarks[L.THUMB_TIP]
        dx = tip.x - base.x
        dy = (tip.y - base.y) * self.frame_aspect
        ratio = KEEP_RATIO if self.held else START_RATIO
        if abs(dx) < abs(dy) * ratio:
            return None
        key = e.KEY_RIGHT if dx > 0 else e.KEY_LEFT
        # Hysteresis only applies to keeping the same key held.
        if self.held and key != self.held and abs(dx) < abs(dy) * START_RATIO:
            return None
        return key

    def _hold(self, key):
        if key == self.held:
            return
        if self.held:
            self._keyboard.release(self.held)
        if key:
            self._keyboard.press(key)
        self.held = key

    def _exit(self):
        self._hold(None)
        self.hand = None

    def release_all(self):
        self._exit()
