"""Record hand landmarks while demonstrating a gesture, for tuning detectors.

Keys (with the window focused):
  l  - label the following frames "swipe_left"
  r  - label the following frames "swipe_right"
  n  - label the following frames "other" (normal hand movement, not a swipe)
  q  - save and quit

Frames are written to recordings/<timestamp>.json. Nothing controls the
computer in this script.
"""

import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from fingers import finger_states
from hand_view import CAMERA_INDEX, MODEL_PATH, draw_hand, user_hand

LABELS = {ord("l"): "swipe_left", ord("r"): "swipe_right", ord("n"): "other"}
WINDOW_NAME = "Gesture Recorder"


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX} (is hand_view.py running?)")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )
    out_dir = Path(__file__).parent / "recordings"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"

    frames = []
    label = "other"
    start = time.monotonic()
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            t = time.monotonic() - start
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))

            hands = []
            for landmarks, world, handedness in zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness
            ):
                hand = user_hand(handedness[0].category_name)
                draw_hand(frame, landmarks, hand)
                hands.append({
                    "hand": hand,
                    "screen": [[p.x, p.y, p.z] for p in landmarks],
                    "world": [[p.x, p.y, p.z] for p in world],
                    "closed": finger_states(world),
                })
            frames.append({"t": round(t, 4), "label": label, "hands": hands})

            color = {"swipe_left": (255, 0, 255), "swipe_right": (0, 255, 255)}.get(label, (200, 200, 200))
            cv2.putText(frame, f"REC  label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "l=left r=right n=other q=save", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in LABELS:
                label = LABELS[key]
            elif key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
    out_path.write_text(json.dumps({"frames": frames}))
    print(f"Saved {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
