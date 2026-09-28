"""Who each hand belongs to.

Bodies come from MediaPipe's pose model, which finds several people at once.
Each body is followed from frame to frame by where its shoulders are, so it
keeps the same id while it stays in view. A hand belongs to the body whose
wrist it sits on.

The pose model is slower than the hand model, so it runs in the background
(live-stream mode) and the camera loop never waits for it: bodies update
whenever a result is ready, usually a frame or so behind the hands.
"""

import itertools
import math
import threading
from pathlib import Path

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

POSE_MODEL = Path(__file__).parent / "models" / "pose_landmarker_lite.task"

MAX_PEOPLE = 4
MAX_JUMP = 0.15       # a body's shoulders may move this far (fraction of the
                      # frame) between results and still keep its id
LOST_S = 0.5          # a body unseen for this long is forgotten
HAND_MAX_DIST = 0.1   # a hand's wrist must be this close to a body's wrist

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16


def center(landmarks):
    a, b = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)


class Person:
    def __init__(self, pid, landmarks, t):
        self.id = pid
        self.update(landmarks, t)

    def update(self, landmarks, t):
        self.landmarks = landmarks
        self.center = center(landmarks)
        self.seen = t


class People:
    def __init__(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=MAX_PEOPLE,
            result_callback=self._on_result,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._latest = None
        self._ids = itertools.count(1)
        self.tracked = {}     # id -> Person, including ones briefly out of sight
        self.visible = set()  # ids found in the latest pose result

    def _on_result(self, result, image, timestamp_ms):
        with self._lock:
            self._latest = (result.pose_landmarks, timestamp_ms / 1000)

    def update(self, mp_image, timestamp_ms, t):
        """Hand this frame to the pose model, and take in whatever result has
        arrived since the last call (if any)."""
        self._landmarker.detect_async(mp_image, timestamp_ms)
        with self._lock:
            latest, self._latest = self._latest, None
        if latest is not None:
            self._track(*latest)
        for pid in [pid for pid, p in self.tracked.items() if t - p.seen > LOST_S]:
            del self.tracked[pid]
            self.visible.discard(pid)

    def _track(self, found, t):
        centers = [center(lm) for lm in found]
        # Give each body the id of the nearest tracked one, closest pairs first.
        pairs = sorted(
            (math.dist(c, p.center), i, pid)
            for i, c in enumerate(centers)
            for pid, p in self.tracked.items()
        )
        matched = {}
        for dist, i, pid in pairs:
            if dist > MAX_JUMP:
                break
            if i not in matched and pid not in matched.values():
                matched[i] = pid

        self.visible = set()
        for i, landmarks in enumerate(found):
            pid = matched.get(i)
            if pid is None:
                pid = next(self._ids)
                self.tracked[pid] = Person(pid, landmarks, t)
            else:
                self.tracked[pid].update(landmarks, t)
            self.visible.add(pid)

    def owner_of(self, hand_landmarks):
        """The id of the visible body this hand is attached to, or None."""
        wrist = hand_landmarks[0]
        best, best_dist = None, HAND_MAX_DIST
        for pid in self.visible:
            body = self.tracked[pid].landmarks
            for i in (L_WRIST, R_WRIST):
                dist = math.hypot(body[i].x - wrist.x, body[i].y - wrist.y)
                if dist < best_dist:
                    best, best_dist = pid, dist
        return best

    def close(self):
        self._landmarker.close()
