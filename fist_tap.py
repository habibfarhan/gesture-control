"""Quick fist (open hand -> fist -> open hand) presses the spacebar.

Handy for play/pause in video players. Every step has to be quick, so
slowly making a fist, or holding one, does nothing.
"""

import math

from evdev import ecodes as e

QUICK_S = 0.5      # max time for open->fist and for fist->open
MAX_FIST_S = 0.6   # a fist held longer than this is not a tap
COOLDOWN_S = 0.6   # ignore further taps for this long after one fires
MAX_TRAVEL = 0.08  # the wrist must stay within this distance (fraction of the
                   # frame) of where the tap started; the curled-finger return
                   # between workspace swipes moves much further
KEY = e.KEY_SPACE


def all_open(states):
    return not any(states.values())


def all_closed(states):
    return all(states.values())


class _HandTap:
    def __init__(self):
        self.last_open = None     # last time the hand was fully open
        self.fist_start = None    # when the current fist began
        self.last_closed = None   # last time the hand was a full fist
        self.open_pos = None      # wrist position when last fully open
        self.travel = 0.0         # furthest the wrist has moved since then


class FistTap:
    def __init__(self, keyboard):
        self._keyboard = keyboard
        self._hands = {}
        self._cooldown_until = 0.0
        self.fired_at = None  # for the on-screen flash

    def update(self, seen, t):
        """`seen` is the output of `StableFingers.update` for this frame."""
        for hand in list(self._hands):
            if hand not in seen:
                del self._hands[hand]
        for hand, (landmarks, _, states) in seen.items():
            h = self._hands.setdefault(hand, _HandTap())
            wrist = landmarks[0]
            if h.open_pos is not None:
                h.travel = max(h.travel, math.hypot(wrist.x - h.open_pos[0],
                                                    wrist.y - h.open_pos[1]))
            if all_closed(states):
                if h.fist_start is None:
                    # Only a fist made quickly from an open hand counts.
                    if h.last_open is not None and t - h.last_open <= QUICK_S:
                        h.fist_start = t
                if h.fist_start is not None:
                    h.last_closed = t
            elif all_open(states):
                if (h.fist_start is not None
                        and t - h.last_closed <= QUICK_S
                        and h.last_closed - h.fist_start <= MAX_FIST_S
                        and h.travel <= MAX_TRAVEL
                        and t >= self._cooldown_until):
                    self._keyboard.tap(KEY)
                    self.fired_at = t
                    self._cooldown_until = t + COOLDOWN_S
                h.fist_start = None
                h.last_open = t
                h.open_pos = (wrist.x, wrist.y)
                h.travel = 0.0
            elif h.fist_start is not None and t - h.last_closed > QUICK_S:
                # Opened too slowly (or only partly); give up on this tap.
                h.fist_start = None
