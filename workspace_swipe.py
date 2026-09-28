"""Switch Hyprland workspaces with a sideways swipe, like the 3-finger touchpad
gesture.

The swipe (tuned on a recording of the real gesture): an open hand moves
quickly sideways while the palm flips over. Moving left on screen, the palm
turns from facing the camera to facing away; moving right, the reverse. As on
the touchpad, moving the hand left goes to the workspace on the right.

Between swipes the hand is brought back with the fingers curled, which is what
stops the return stroke from counting as a swipe the other way.
"""

import json
import subprocess
import threading

import numpy as np
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark as L

MIN_DISTANCE = 0.15   # wrist travel, fraction of camera width
MAX_DURATION = 0.4    # seconds; slower sideways movement is not a swipe
FLIP_MIN = 0.3        # palm must clearly face one way before and the other after
MAX_CLOSED = 1        # fingers allowed to read closed (the thumb often tucks in)
OPEN_FRACTION = 0.6   # share of stroke frames where the hand must be open
SAME_COOLDOWN = 0.5   # seconds before another swipe the same way
OPPOSITE_COOLDOWN = 0.8  # longer, so the return stroke can't swipe back


def palm_facing(world, hand):
    """+1 when the palm faces the camera, -1 when the back of the hand does."""
    w = np.array([[p.x, p.y, p.z] for p in world])
    up = w[L.MIDDLE_FINGER_MCP] - w[L.WRIST]
    across = w[L.INDEX_FINGER_MCP] - w[L.PINKY_MCP]
    normal = np.cross(up, across)
    facing = normal[2] / np.linalg.norm(normal)
    # Left and right hands are mirror images, so their normals point opposite ways.
    return -facing if hand == "Right" else facing


class _Sample:
    __slots__ = ("t", "x", "palm", "open")

    def __init__(self, t, x, palm, is_open):
        self.t, self.x, self.palm, self.open = t, x, palm, is_open


class WorkspaceSwipe:
    def __init__(self, switch=None):
        self._switch = switch or switch_workspace
        self._history = {}  # hand -> recent samples
        self._last_fire = {}  # direction -> time
        self.fired = None  # (direction, time) for the on-screen flash

    def update(self, hands, t):
        """`hands` is a list of (screen landmarks, world landmarks, hand label,
        raw finger states). Raw states: the stroke is too quick to debounce."""
        # Only age samples out; don't drop a hand that vanishes for a frame or
        # two, since motion blur mid-swipe often hides it.
        for history in self._history.values():
            history[:] = [s for s in history if t - s.t <= MAX_DURATION]
        for landmarks, world, hand, raw in hands:
            history = self._history.setdefault(hand, [])
            sample = _Sample(t, landmarks[L.WRIST].x, palm_facing(world, hand),
                             sum(raw.values()) <= MAX_CLOSED)
            history.append(sample)
            direction = self._detect(history, sample)
            if direction and self._fire(direction, t):
                history.clear()

    def _detect(self, history, end):
        # The stroke has to end open with the palm clearly flipped.
        if not end.open or abs(end.palm) < FLIP_MIN:
            return None
        for i, start in enumerate(history[:-1]):
            dx = end.x - start.x
            if abs(dx) < MIN_DISTANCE or abs(start.palm) < FLIP_MIN:
                continue
            moving_left = dx < 0
            flipped = (start.palm > 0 > end.palm) if moving_left else (start.palm < 0 < end.palm)
            if not flipped:
                continue
            stroke = history[i:]
            if sum(s.open for s in stroke) / len(stroke) < OPEN_FRACTION:
                continue
            return "right" if moving_left else "left"
        return None

    def _fire(self, direction, t):
        opposite = "left" if direction == "right" else "right"
        if (t - self._last_fire.get(direction, -1e9) < SAME_COOLDOWN
                or t - self._last_fire.get(opposite, -1e9) < OPPOSITE_COOLDOWN):
            return False
        self._last_fire[direction] = t
        self.fired = (direction, t)
        threading.Thread(target=self._switch, args=(direction,), daemon=True).start()
        return True


def _hyprctl_json(what):
    return json.loads(subprocess.run(["hyprctl", what, "-j"], capture_output=True,
                                     text=True).stdout)


def switch_workspace(direction):
    """Go to the neighbouring workspace on this monitor, like the touchpad
    gesture: stop at the first one going left, create a new one going right."""
    active = _hyprctl_json("activeworkspace")
    ids = sorted(w["id"] for w in _hyprctl_json("workspaces")
                 if w["monitor"] == active["monitor"] and w["id"] > 0)
    current = active["id"]
    if direction == "right":
        later = [i for i in ids if i > current]
        target = str(later[0]) if later else "r+1"
    else:
        earlier = [i for i in ids if i < current]
        if not earlier:
            return
        target = str(earlier[-1])
    subprocess.run(["hyprctl", "dispatch", f'hl.dsp.focus({{workspace="{target}"}})'],
                   capture_output=True)
