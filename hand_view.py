"""Show the webcam feed with MediaPipe hand landmarks drawn on each detected hand.

Press q or Esc to quit.
"""

import time
from pathlib import Path

import cv2
from evdev import ecodes as e
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
)

from arrow_keys import ArrowKeys
from control_gate import ControlGate, Hand
from cursor_control import CursorController, VirtualMouse
from dictation import Dictation, VirtualKeyboard
from fist_tap import FistTap
from fingers import StableFingers, finger_states
from people import People
from volume_control import VolumeController
from workspace_swipe import WorkspaceSwipe, palm_facing

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
CAMERA_INDEX = 0
WINDOW_NAME = "Gesture Control - Hand View"
ARROW_KEYS_ENABLED = False  # thumb-sideways pose holds left/right arrow

FINGERTIPS = {4, 8, 12, 16, 20}
LINE_COLOR = (255, 255, 255)
POINT_COLOR = (0, 200, 0)
TIP_COLOR = (0, 0, 255)


def user_hand(label):
    # MediaPipe reports handedness from the camera's point of view, which is
    # the opposite of the user's own hand even on the mirrored frame.
    return {"Left": "Right", "Right": "Left"}.get(label, label)


def draw_hand(frame, landmarks, label):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], LINE_COLOR, 2)

    for i, (x, y) in enumerate(points):
        color = TIP_COLOR if i in FINGERTIPS else POINT_COLOR
        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)
        # Landmark number, outlined so it reads on any background.
        label_pos = (x + 8, y - 8)
        cv2.putText(frame, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3)
        cv2.putText(frame, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)

    wx, wy = points[0]
    cv2.putText(frame, label, (wx - 20, wy + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)


def draw_finger_states(frame, states, hand):
    # Left hand's readout on the left edge, right hand's on the right edge.
    x = 10 if hand == "Left" else frame.shape[1] - 170
    cv2.putText(frame, f"{hand} hand", (x, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    for i, (finger, closed) in enumerate(states.items()):
        color = TIP_COLOR if closed else POINT_COLOR
        text = f"{finger}: {'closed' if closed else 'open'}"
        cv2.putText(frame, text, (x, 95 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)


def draw_people(frame, people, gate):
    """Label each tracked body at its shoulders; the one with control in green."""
    h, w = frame.shape[:2]
    for pid in people.visible:
        cx, cy = people.tracked[pid].center
        owner = gate.owner and gate.owner[0] == pid
        text = "YOU (control)" if owner else f"person {pid}"
        color = (0, 255, 0) if owner else (200, 200, 200)
        cv2.putText(frame, text, (int(cx * w) - 50, int(cy * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=4,  # several people's hands can be in view
    )

    frame_aspect = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    mouse = VirtualMouse()
    cursor = CursorController(mouse, frame_aspect)
    volume = VolumeController()
    stable = StableFingers()
    keyboard = VirtualKeyboard()
    fist_tap = FistTap(keyboard)
    arrows = ArrowKeys(keyboard, frame_aspect)
    swipe = WorkspaceSwipe()
    dictation = Dictation(keyboard)
    people = People()
    gate = ControlGate(people)

    # Resizable, with the picture scaled to fit (default windows are fixed-size).
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    start = time.monotonic()
    prev = start
    fps = 0.0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break

            # Mirror the preview so moving your hand right moves it right on screen.
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_time = time.monotonic() - start
            result = landmarker.detect_for_video(mp_image, int(frame_time * 1000))

            people.update(mp_image, int(frame_time * 1000), frame_time)

            hands = []
            for landmarks, world, handedness in zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness
            ):
                label = user_hand(handedness[0].category_name)
                hands.append(Hand(landmarks, world, label, finger_states(world),
                                  palm_facing(world, label)))
            # Only the hand that asked for control gets to use any gesture.
            owner = gate.update(hands, frame_time)
            owned = [owner] if owner else []

            for hand in hands:
                draw_hand(frame, hand.landmarks, hand.label)
                draw_finger_states(frame, hand.states, hand.label)
            draw_people(frame, people, gate)

            seen = stable.update([(h.landmarks, h.label, h.states) for h in owned])
            cursor.update(seen, frame_time)
            volume.update(seen, frame_time)
            fist_tap.update(seen, frame_time)
            if ARROW_KEYS_ENABLED:
                arrows.update(seen, frame_time, blocked=cursor.active or volume.active)
            swipe.update([(h.landmarks, h.world, h.label, h.states) for h in owned],
                         frame_time)

            if gate.owner_lost:
                ctl_text, ctl_color = (f"CONTROL: {gate.owner_label} hand - lost you, "
                                       "come back into view"), (0, 200, 255)
            elif gate.owner:
                ctl_text, ctl_color = f"CONTROL: {gate.owner_label} hand", (0, 255, 0)
            else:
                ctl_text, ctl_color = "NO CONTROL: hold open back of hand to camera", (0, 0, 255)
            cv2.putText(frame, ctl_text, (10, frame.shape[0] - 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ctl_color, 2)
            h, w = frame.shape[:2]
            for hand, progress in gate.progress.items():
                x = int(hand.landmarks[0].x * w) - 80
                y = int(hand.landmarks[0].y * h) + 45
                if gate.owner:
                    verb = "releasing"
                else:
                    verb = "claiming"
                cv2.putText(frame, verb, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 0), 2)
                cv2.rectangle(frame, (x, y + 10), (x + 160, y + 25), (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y + 10), (x + int(160 * progress), y + 25),
                              (255, 255, 0), -1)

            if not dictation.ready:
                mic_text, mic_color = "MIC: loading speech model...", (0, 200, 255)
            elif dictation.dictating:
                mic_text, mic_color = "DICTATING - say 'stop dictate'", (0, 0, 255)
            else:
                mic_text, mic_color = "MIC: say 'dictate'", (200, 200, 200)
            cv2.putText(frame, mic_text, (10, frame.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mic_color, 2)

            if cursor.active:
                cv2.putText(frame, f"CURSOR MODE ({cursor.hand} hand)",
                            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
            elif volume.active:
                level = f" {volume.level * 100:.0f}%" if volume.level is not None else ""
                cv2.putText(frame, f"VOLUME MODE ({volume.hand} hand){level}",
                            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 128, 0), 2)

            if swipe.fired and frame_time - swipe.fired[1] < 0.6:
                label = "WORKSPACE ->" if swipe.fired[0] == "right" else "<- WORKSPACE"
                cv2.putText(frame, label, (frame.shape[1] // 2 - 110, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

            if arrows.active:
                arrow = {e.KEY_LEFT: "<- LEFT", e.KEY_RIGHT: "RIGHT ->"}.get(arrows.held, "point thumb left/right")
                cv2.putText(frame, f"ARROW MODE ({arrows.hand} hand): {arrow}",
                            (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 255), 2)

            if fist_tap.fired_at is not None and frame_time - fist_tap.fired_at < 0.5:
                cv2.putText(frame, "SPACE", (frame.shape[1] // 2 - 60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(frame, f"FPS: {fps:.0f}  Hands: {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    arrows.release_all()
    people.close()
    dictation.close()
    keyboard.close()
    cursor.release_all()
    mouse.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
