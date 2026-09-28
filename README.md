# Gesture Control

Control a Linux desktop with your hands and voice through a webcam and microphone:
move the mouse, click and drag, change the volume, press Space, switch Hyprland
workspaces, and dictate text, without touching the keyboard or mouse.

Everything runs locally. Hand and body tracking use Google's MediaPipe, and
speech is transcribed offline with faster-whisper (on an NVIDIA GPU if you have
one, otherwise on the CPU). Mouse and keyboard input goes through a kernel-level
virtual device (`/dev/uinput`), so it works on Wayland as well as X11.

> Built with [Claude Code](https://claude.com/claude-code).

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup after cloning](#setup-after-cloning)
- [Running it](#running-it)
- [Using it](#using-it)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Project layout](#project-layout)

## Features

- **Hand-over control.** Nothing responds until someone asks for control with a
  deliberate gesture, and then only that person's hand is obeyed. Other people
  in view, including their same (left or right) hand, are ignored.
- **Cursor mode.** Move the mouse like a trackpad with your wrist; close your
  index finger to left-click or drag, close your thumb to right-click.
- **Volume mode.** Raise or lower your hand to change the system volume.
- **Fist tap.** A quick open-fist-open presses Space (play/pause in most players).
- **Workspace swipe.** Swipe sideways to switch Hyprland workspaces, like the
  three-finger touchpad gesture.
- **Arrow keys** (off by default). Point your thumb left or right to hold an arrow key.
- **Voice dictation.** Say "dictate" and it types what you say into the
  focused window; say "stop dictate" to stop.

## Requirements

| What | Why | Notes |
|---|---|---|
| Linux | Virtual mouse/keyboard via `/dev/uinput` | Tested on CachyOS (Arch) with Hyprland 0.56 |
| [uv](https://docs.astral.sh/uv/) | Installs Python 3.12 and all dependencies | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| A webcam | Hand and body tracking | 640x480 at 30 FPS is plenty |
| A microphone | Dictation | |
| PortAudio | Microphone access (`sounddevice`) | Arch: `portaudio`, Debian/Ubuntu: `libportaudio2` |
| PipeWire (`wpctl`) | Volume control | Standard on most current distros |
| Hyprland (`hyprctl`) | Workspace swipe | Everything else works without it |
| NVIDIA GPU + driver (optional) | Fast dictation | Falls back to the CPU automatically |

About 3 GB of disk space is needed: the Python packages include NVIDIA's CUDA
libraries (about 1.3 GB, installed even if you have no NVIDIA GPU), and the
speech model (about 500 MB) is downloaded on first run.

## Setup after cloning

### 1. Clone the repository

```bash
git clone https://github.com/habibfarhan/gesture-control.git
cd gesture-control
```

### 2. Install the system packages

Arch / CachyOS / Manjaro:

```bash
sudo pacman -S --needed portaudio pipewire wireplumber
```

Debian / Ubuntu:

```bash
sudo apt install libportaudio2 pipewire wireplumber
```

### 3. Allow your user to create virtual input devices

The app creates a virtual mouse and keyboard through `/dev/uinput`. Check who
can write to it:

```bash
ls -l /dev/uinput
```

If it belongs to the `input` group (for example `crw-rw---- root input`), add
yourself to that group, then **log out and back in** (or reboot):

```bash
sudo usermod -aG input "$USER"
```

If it belongs to `root` only, add a udev rule that hands it to the `input` group,
and make sure the `uinput` module loads at boot:

```bash
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' \
  | sudo tee /etc/udev/rules.d/99-uinput.rules
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
sudo modprobe uinput
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG input "$USER"
```

Then log out and back in. Confirm with `id -nG`; the list should include `input`.

> Being in the `input` group also lets your user read raw input from all input
> devices. That is normal for tools like this, but worth knowing.

### 4. Install the Python dependencies

```bash
uv sync
```

This creates `.venv/` with Python 3.12 and everything listed in `pyproject.toml`.
The first sync downloads roughly 2 GB, mostly the NVIDIA libraries.

### 5. (Optional) Check the GPU

If you have an NVIDIA GPU, make sure the driver works:

```bash
nvidia-smi
```

Nothing else is needed: the CUDA 12 cuBLAS and cuDNN 9 libraries come from
`uv sync`, and `dictation.py` loads them itself, so there is no
`LD_LIBRARY_PATH` to set.

### 6. Model files

The MediaPipe models are included in `models/`:

- `hand_landmarker.task` - hand tracking
- `pose_landmarker_lite.task` - body tracking (who a hand belongs to)

If they are ever missing, download them again:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o models/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

The Whisper speech model (`small.en`) downloads automatically from Hugging Face
the first time you run the app, and is cached in `~/.cache/huggingface/`.

## Running it

```bash
uv run hand_view.py
```

A window opens showing the camera with the tracked hands and bodies drawn on
it. The terminal shows dictation messages; look for:

```
[dictation] small.en on GPU
[dictation] listening - say 'dictate' to start
```

(or `on CPU` if no usable GPU was found; the reason is printed just above).

Quit with **q** or **Esc** in the camera window, or close the window.

The window can be resized and the picture scales with it. On Hyprland, hold
your main modifier and drag with the right mouse button.

## Using it

### 1. Take control

Nothing responds until you ask for control. Hold one hand up with:

- the **back of your hand** facing the camera,
- the hand **upright** (fingers pointing up, within 30 degrees),
- **all five fingers open**,
- and keep it **still for 1 second**.

A "claiming" bar fills under your hand, and the status line changes to
`CONTROL: Right hand` (or Left). From then on only **that hand of yours**
controls anything. Your body is labelled `YOU (control)` in green.

To give up control, make the same gesture again with the same hand ("releasing").
After that nothing responds until someone asks again. Holding the gesture past
1 second doesn't flip control straight back; lower your hand first.

While you have control:

- nobody else's hands do anything, including their same hand;
- your other hand doesn't do anything either;
- if you step out of view, control waits 3 seconds for you to come back to about
  the same spot, and then ends so someone else can take it.

Your shoulders need to be in view: a hand is matched to a person by the body
it's attached to.

### 2. Gestures

Each mode's pose has to be held briefly (a quarter of a second) and fairly still
before it turns on, so passing through a pose by accident doesn't trigger it.

| Gesture | Pose | What it does |
|---|---|---|
| **Cursor mode** | Thumb and index finger open, the other three closed (an "L") | Moving your wrist moves the mouse, like a trackpad |
| Left click / drag | In cursor mode, close your index finger | Quick close = click; move while closed = drag |
| Right click | In cursor mode, close your thumb | |
| Leave cursor mode | Open any of your middle, ring or pinky fingers | Also releases any held button |
| **Volume mode** | Thumb and pinky open, the other three closed ("call me") | Raise your hand to turn the volume up, lower it to turn it down |
| **Space** | Open hand, then a quick fist, then open again | Presses Space (play/pause) |
| **Next workspace** | Open hand, swipe quickly to the left while the hand turns over | Goes to the workspace on the right (Hyprland) |
| **Previous workspace** | Open hand, swipe quickly to the right while the hand turns over | Goes to the workspace on the left (Hyprland) |
| **Arrow keys** (off by default) | Thumbs-up turned sideways | Holds Left or Right arrow while you point |

Bring your hand back between workspace swipes with your fingers curled, so the
return stroke doesn't count as a swipe the other way.

### 3. Dictation

Dictation is controlled by voice and works whether or not a hand has control.

- Say **"dictate"** to start. Everything you say afterwards is typed into the
  focused window.
- Say **"stop dictate"** (or "stop dictating") to stop.

Speech is transcribed in chunks: a chunk ends when you pause for about 0.3
seconds (or after 20 seconds of continuous speech). Each chunk is logged in the
terminal:

```
[dictation] 3.4 s of audio, ended by pause, transcribed in 0.11 s
[dictation] heard: 'Hello there.'
```

## Configuration

Settings are constants at the top of each file.

| Setting | File | Default | Meaning |
|---|---|---|---|
| `CAMERA_INDEX` | `hand_view.py` | `0` | Which camera to use (`/dev/video0`) |
| `ARROW_KEYS_ENABLED` | `hand_view.py` | `False` | Turn on the thumb-sideways arrow keys |
| `HOLD_S` | `control_gate.py` | `1.0` | Seconds the control gesture must be held |
| `MAX_TILT_DEG` | `control_gate.py` | `30` | How far from upright the hand may lean for the control gesture |
| `LOST_GRACE_S` | `control_gate.py` | `3.0` | How long control waits for its owner to come back into view |
| `SENSITIVITY` | `cursor_control.py` | `3500` | Cursor speed: screen pixels per full camera width of hand movement |
| `VOLUME_PER_HEIGHT` | `volume_control.py` | `1.5` | Volume change per full camera height of hand movement |
| `MODEL_NAME` | `dictation.py` | `"small.en"` | Whisper model: `tiny.en` / `base.en` are faster, `medium.en` / `large-v3` more accurate |
| `BEAM_SIZE` | `dictation.py` | `5` | Lower (e.g. `1`) is faster, slightly less accurate |
| `USE_GPU` | `dictation.py` | `True` | Set `False` to force the CPU |
| `END_SILENCE_FRAMES` | `dictation.py` | `10` | Pause that ends a chunk, in 30 ms frames (10 = 300 ms). Raise it if sentences get split mid-way |
| `VAD_AGGRESSIVENESS` | `dictation.py` | `2` | 0-3; higher ignores more background noise |

## Troubleshooting

**`PermissionError` / `Permission denied: '/dev/uinput'`**
Your user can't create virtual input devices. Follow
[step 3](#3-allow-your-user-to-create-virtual-input-devices), and remember to log
out and back in afterwards.

**`Could not open camera 0`**
Another program is using the camera (a video call, or a second copy of this app),
or your camera has a different index. List cameras with `ls /dev/video*` and try
another `CAMERA_INDEX`.

**Nothing responds to my gestures**
Check the status line at the bottom of the window. If it says `NO CONTROL`, take
control first (back of hand, upright, fingers open, 1 second). Make sure your
shoulders are in view, since hands are matched to bodies.

**The "claiming" bar never appears**
All five fingers must read `open` in the finger list on screen, the back of the
hand must face the camera (not the palm), and the hand must point up.

**`[dictation] GPU unavailable (...), using CPU`**
The reason is in the message. Check `nvidia-smi` works; if the libraries are
missing, run `uv sync` again. The CPU works too, just more slowly; a smaller
`MODEL_NAME` helps there.

**Dictation types late**
Look at the `ended by` part of the log. `time limit` means background noise
kept it from hearing a pause: raise `VAD_AGGRESSIVENESS` to 3 or use a closer
microphone. Long chunks ending by `pause` are just long sentences.

**`sounddevice` fails with `PortAudio library not found`**
Install PortAudio ([step 2](#2-install-the-system-packages)).

**Workspace swipe does nothing**
It needs Hyprland (it calls `hyprctl`). The other features don't.

## How it works

- **Tracking.** Each camera frame is mirrored and passed to MediaPipe's hand
  landmarker (21 points per hand, up to 4 hands). The pose landmarker (up to 4
  bodies) runs in the background in live-stream mode, so the camera loop never
  waits for it.
- **Finger states.** Each finger is open or closed, measured along the hand's
  own axes from its 3D world landmarks, so it works however the hand is turned.
  States are debounced over a couple of frames to ignore flicker.
- **Who owns a hand.** A hand belongs to the body whose wrist it sits on.
  Bodies keep an id by following their shoulders from frame to frame. Because
  bodies update a frame or so behind hands, the controlling hand also stays
  attached by following its own position between body updates.
- **Control gate.** Only the controlling hand's data is passed on to the
  gesture modes; everyone else's hands are dropped before any gesture sees them.
- **Output.** The virtual mouse and keyboard are `evdev` `UInput` devices.
  Volume uses `wpctl`, and workspaces use `hyprctl`.
- **Dictation.** The microphone is split into speech chunks with WebRTC voice
  activity detection, and each chunk is transcribed by faster-whisper. The
  words between "dictate" and "stop dictate" are typed through the virtual
  keyboard.

## Project layout

| File | Purpose |
|---|---|
| `hand_view.py` | Entry point: camera loop, drawing, wiring everything together |
| `control_gate.py` | Who has control; the claim/release gesture |
| `people.py` | Body tracking and matching hands to bodies |
| `fingers.py` | Open/closed finger detection, debouncing, pose timers |
| `cursor_control.py` | Cursor mode and the virtual mouse |
| `volume_control.py` | Volume mode |
| `fist_tap.py` | Quick fist = Space |
| `workspace_swipe.py` | Hyprland workspace swipe |
| `arrow_keys.py` | Thumb-sideways arrow keys |
| `dictation.py` | Voice dictation and the virtual keyboard |
| `record_gesture.py` | Records labelled hand landmarks to `recordings/`, for tuning detectors (controls nothing) |
| `models/` | MediaPipe model files |

---

Made with [Claude Code](https://claude.com/claude-code).
