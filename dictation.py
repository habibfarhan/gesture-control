"""Voice dictation: always listening, types what you say while dictating.

Say "dictate" to start typing your speech into the focused window, and
"stop dictate" (or "stop dictating") to stop. Speech is cut into utterances
with a voice-activity detector and transcribed offline with faster-whisper.
"""

import ctypes
import glob
import importlib.util
import os
import queue
import re
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
from evdev import UInput, ecodes as e
from faster_whisper import WhisperModel

MODEL_NAME = "small.en"   # tiny.en / base.en are faster, medium.en more accurate
BEAM_SIZE = 5
USE_GPU = True            # falls back to the CPU if CUDA can't be used

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
VAD_AGGRESSIVENESS = 2    # 0-3, higher ignores more background noise
PRE_ROLL_FRAMES = 10      # audio kept from just before speech starts (300 ms)
START_RATIO = 0.6         # fraction of pre-roll frames voiced to start an utterance
END_SILENCE_FRAMES = 10   # silence that ends an utterance (300 ms)
MIN_SPEECH_FRAMES = 8     # ignore blips shorter than this (240 ms)
MAX_UTTERANCE_S = 20

START_WORDS = {"dictate"}
STOP_WORDS = {"dictate", "dictating", "dictation"}  # said after "stop"

KEY_DELAY_S = 0.003


def _load_cuda_libs():
    """Load the pip-installed cuBLAS / cuDNN libraries so CTranslate2 can find
    them without LD_LIBRARY_PATH (which has to be set before Python starts).
    Once loaded, a later lookup by name finds them already in memory."""
    paths = []
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(package)
        if spec is None:
            return False
        for root in spec.submodule_search_locations:
            paths += sorted(glob.glob(os.path.join(root, "lib", "lib*.so*")))
    # Some depend on others; keep retrying until everything that can load has.
    while paths:
        failed = []
        for path in paths:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
        if len(failed) == len(paths):
            return False
        paths = failed
    return True


def load_model():
    """Whisper on the GPU if possible, otherwise on the CPU."""
    if USE_GPU and _load_cuda_libs():
        try:
            model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
            # The first GPU run is slow (setup); get it out of the way now.
            list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                                  language="en", beam_size=BEAM_SIZE)[0])
            print(f"[dictation] {MODEL_NAME} on GPU")
            return model
        except Exception as err:
            print(f"[dictation] GPU unavailable ({err}), using CPU")
    print(f"[dictation] {MODEL_NAME} on CPU")
    return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


class VirtualKeyboard:
    """Types ASCII text through a kernel-level virtual keyboard (US layout)."""

    _PLAIN = {" ": e.KEY_SPACE, "\n": e.KEY_ENTER, "-": e.KEY_MINUS,
              "=": e.KEY_EQUAL, "[": e.KEY_LEFTBRACE, "]": e.KEY_RIGHTBRACE,
              "\\": e.KEY_BACKSLASH, ";": e.KEY_SEMICOLON, "'": e.KEY_APOSTROPHE,
              "`": e.KEY_GRAVE, ",": e.KEY_COMMA, ".": e.KEY_DOT, "/": e.KEY_SLASH}
    _SHIFTED = {"_": "-", "+": "=", "{": "[", "}": "]", "|": "\\", ":": ";",
                '"': "'", "~": "`", "<": ",", ">": ".", "?": "/",
                "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6",
                "&": "7", "*": "8", "(": "9", ")": "0"}
    # Whisper sometimes emits typographic characters the keymap can't type.
    _REPLACE = {"’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
                "–": "-", "—": "-"}

    def __init__(self):
        self._keys = {}
        for c in "abcdefghijklmnopqrstuvwxyz":
            code = getattr(e, f"KEY_{c.upper()}")
            self._keys[c] = (code, False)
            self._keys[c.upper()] = (code, True)
        for c in "0123456789":
            self._keys[c] = (getattr(e, f"KEY_{c}"), False)
        for c, code in self._PLAIN.items():
            self._keys[c] = (code, False)
        for c, base in self._SHIFTED.items():
            self._keys[c] = (self._keys[base][0], True)
        codes = {code for code, _ in self._keys.values()}
        codes |= {e.KEY_LEFTSHIFT, e.KEY_LEFT, e.KEY_RIGHT, e.KEY_UP, e.KEY_DOWN}
        self._ui = UInput({e.EV_KEY: sorted(codes)}, name="gesture-control-keyboard")
        # Dictation types from its own thread; gestures tap keys from the main one.
        self._lock = threading.Lock()

    def type(self, text):
        for old, new in self._REPLACE.items():
            text = text.replace(old, new)
        for c in text:
            if c not in self._keys:
                continue
            code, shift = self._keys[c]
            with self._lock:
                if shift:
                    self._key(e.KEY_LEFTSHIFT, 1)
                self._key(code, 1)
                self._key(code, 0)
                if shift:
                    self._key(e.KEY_LEFTSHIFT, 0)
            time.sleep(KEY_DELAY_S)

    def tap(self, code):
        with self._lock:
            self._key(code, 1)
            self._key(code, 0)

    def press(self, code):
        with self._lock:
            self._key(code, 1)

    def release(self, code):
        with self._lock:
            self._key(code, 0)

    def _key(self, code, value):
        self._ui.write(e.EV_KEY, code, value)
        self._ui.syn()

    def close(self):
        self._ui.close()


def _norm(token):
    return re.sub(r"[^a-z]", "", token.lower())


def apply_commands(text, dictating):
    """Split a transcript on the start/stop commands.

    Returns (text to type, dictating afterwards). Words before "dictate" and
    after "stop dictate" are dropped; words in between are kept verbatim.
    """
    tokens = text.split()
    words = [_norm(t) for t in tokens]
    out = []
    i = 0
    while i < len(tokens):
        if dictating and words[i] == "stop" and i + 1 < len(tokens) and words[i + 1] in STOP_WORDS:
            dictating = False
            i += 2
            continue
        if not dictating and words[i] in START_WORDS:
            dictating = True
        elif dictating:
            out.append(tokens[i])
        i += 1
    # "Hello, stop dictating" shouldn't leave a dangling comma.
    return " ".join(out).rstrip(",;:- "), dictating


class Dictation:
    def __init__(self, keyboard):
        self.dictating = False
        self.ready = False
        self.last_heard = ""
        self._keyboard = keyboard
        self._audio = queue.Queue()
        self._running = True
        self._typed_this_session = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        self._thread.join(timeout=2)

    def _on_audio(self, indata, frames, time_info, status):
        self._audio.put(bytes(indata))

    def _run(self):
        model = load_model()
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                               channels=1, dtype="int16", callback=self._on_audio):
            self.ready = True
            print("[dictation] listening - say 'dictate' to start")
            for utterance, ended in self._utterances(vad):
                self._handle(model, utterance, ended)

    def _utterances(self, vad):
        """Yield (int16 audio, what ended it) for each stretch of speech."""
        pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        speech = []
        voiced_count = 0
        silence = 0
        in_speech = False
        max_frames = MAX_UTTERANCE_S * 1000 // FRAME_MS
        while self._running:
            try:
                frame = self._audio.get(timeout=0.2)
            except queue.Empty:
                continue
            if len(frame) != FRAME_SAMPLES * 2:
                continue
            voiced = vad.is_speech(frame, SAMPLE_RATE)
            if not in_speech:
                pre_roll.append((frame, voiced))
                if sum(v for _, v in pre_roll) >= START_RATIO * PRE_ROLL_FRAMES:
                    in_speech = True
                    speech = [f for f, _ in pre_roll]
                    voiced_count = sum(v for _, v in pre_roll)
                    silence = 0
                    pre_roll.clear()
                continue
            speech.append(frame)
            voiced_count += voiced
            silence = 0 if voiced else silence + 1
            if silence >= END_SILENCE_FRAMES or len(speech) >= max_frames:
                in_speech = False
                if voiced_count >= MIN_SPEECH_FRAMES:
                    ended = "pause" if silence >= END_SILENCE_FRAMES else "time limit"
                    yield np.frombuffer(b"".join(speech), dtype=np.int16), ended

    def _handle(self, model, audio, ended):
        samples = audio.astype(np.float32) / 32768.0
        start = time.monotonic()
        segments, _ = model.transcribe(samples, language="en", beam_size=BEAM_SIZE,
                                       vad_filter=True, condition_on_previous_text=False)
        text = "".join(s.text for s in segments).strip()
        took = time.monotonic() - start
        print(f"[dictation] {len(samples) / SAMPLE_RATE:.1f} s of audio, ended by "
              f"{ended}, transcribed in {took:.2f} s")
        if not text:
            return
        self.last_heard = text
        print(f"[dictation] heard: {text!r}")

        was_dictating = self.dictating
        to_type, self.dictating = apply_commands(text, self.dictating)
        if self.dictating and not was_dictating:
            print("[dictation] ON")
            self._typed_this_session = False
        if to_type:
            # Separate consecutive utterances with a space.
            prefix = " " if self._typed_this_session else ""
            self._keyboard.type(prefix + to_type)
            self._typed_this_session = True
        if was_dictating and not self.dictating:
            print("[dictation] OFF")
