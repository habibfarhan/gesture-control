"""Change system volume by raising or lowering the hand.

Pose to enter volume mode: thumb and pinky open, index/middle/ring closed.
While in volume mode, moving the wrist (point 0) up raises the volume and
moving it down lowers it, relative to where the hand was on entering. Breaking
the pose, or losing the hand, leaves the mode.
"""

import os
import re
import subprocess
from pathlib import Path

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark as L

from cursor_control import LOST_FRAMES, OneEuroFilter
from fingers import PoseTimer

VOLUME_PER_HEIGHT = 1.5   # volume change (1.0 = 100%) per full camera height moved
STEP = 0.02               # volume changes in 2% steps
MAX_VOLUME = 1.0

SINK = "@DEFAULT_AUDIO_SINK@"

# Same popup the HyDE volume keys show (~/.local/lib/hyde/volumecontrol.sh):
# same app name and replace id, so gesture and key popups replace each other.
NOTIFY_APP = "HyDE Notify"
NOTIFY_ID = "8"
ICON_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"),
                "icons/Wallbash-Icon/media")


def in_volume_pose(states):
    return (not states["thumb"] and not states["pinky"]
            and states["index"] and states["middle"] and states["ring"])


def get_volume():
    out = subprocess.run(["wpctl", "get-volume", SINK],
                         capture_output=True, text=True).stdout
    match = re.search(r"Volume:\s*([\d.]+)", out)
    return float(match.group(1)) if match else None


def change_volume(delta):
    sign = "+" if delta > 0 else "-"
    # Fire and forget so the camera loop never waits on wpctl.
    subprocess.Popen(["wpctl", "set-volume", "-l", str(MAX_VOLUME), SINK,
                      f"{abs(delta) * 100:.0f}%{sign}"])


def sink_name():
    out = subprocess.run(["wpctl", "inspect", SINK],
                         capture_output=True, text=True).stdout
    match = re.search(r'node\.description = "(.*)"', out)
    return match.group(1) if match else ""


def notify_volume(level, sink=""):
    vol = round(level * 100)
    knob = min((vol + 2) // 5 * 5, 100)
    bar = "." * max(vol // 15 - 1, 0)
    try:
        subprocess.Popen(["notify-send", "-a", NOTIFY_APP, "-r", NOTIFY_ID,
                          "-t", "2000", "-i", str(ICON_DIR / f"knob-{knob}.svg"),
                          f"{vol}{bar}", sink])
    except FileNotFoundError:
        pass  # no notify-send; the camera window still shows the level


class VolumeController:
    def __init__(self, set_volume=change_volume, read_volume=get_volume,
                 notify=notify_volume, read_sink=sink_name):
        self._set_volume = set_volume
        self._read_volume = read_volume
        self._notify = notify
        self._read_sink = read_sink
        self._sink = ""
        self.active = False
        self.hand = None
        self.level = None  # last known volume, for the on-screen readout
        self._filter = OneEuroFilter()
        self._last = None
        self._acc = 0.0
        self._missing = 0
        self._pose_timer = PoseTimer(in_volume_pose)

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
            self._last = None
            self._filter.reset()
            return
        self._missing = 0

        landmarks, _, states = seen[self.hand]
        if not in_volume_pose(states):
            self._exit()
            return

        y = self._filter(landmarks[L.WRIST].y, t)
        if self._last is None:
            self._last = y
            return
        # Screen y grows downward, so moving up (smaller y) is positive.
        self._acc += (self._last - y) * VOLUME_PER_HEIGHT
        self._last = y
        steps = int(self._acc / STEP)
        if steps:
            self._acc -= steps * STEP
            delta = steps * STEP
            self._set_volume(delta)
            if self.level is not None:
                self.level = min(max(self.level + delta, 0.0), MAX_VOLUME)
                self._notify(self.level, self._sink)

    def _enter(self, hand):
        self.active = True
        self.hand = hand
        self.level = self._read_volume()
        self._sink = self._read_sink()
        self._missing = 0
        self._last = None
        self._acc = 0.0
        self._filter.reset()

    def _exit(self):
        self.active = False
        self.hand = None
