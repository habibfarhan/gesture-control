"""Decides which hand, if any, is allowed to control the computer.

Nothing controls anything until a hand asks for control: hold it out with the
back of the hand facing the camera, upright (fingers pointing up), all five
fingers open, and still, for `HOLD_S`. Control then belongs to that hand
(left or right) of that person, and no other hand, including anyone else's
same hand, can take it. The same gesture from the controlling hand gives
control up, and then nothing controls anything until a hand asks again.

A hand counts as someone's when it sits on the wrist of their body (see
`people.py`), and people are told apart only by following their bodies. If
the owner's body drops out of view, a body that shows up where it was within
`LOST_GRACE_S` is taken to be them again; after that, control ends.
"""

import math

from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark as L

from fingers import CONFIRM_FRAMES, Debounced

HOLD_S = 1.0          # how long the gesture must be held
MAX_TRAVEL = 0.08     # moving the wrist further than this (fraction of the
                      # frame) during the hold restarts it
MIN_BACK = 0.5        # palm_facing() below minus this counts as the back of
                      # the hand facing the camera
MAX_TILT_DEG = 30     # the hand must point within this of straight up on screen
LOST_GRACE_S = 3.0    # how long control waits for the owner's body to return
RETURN_DIST = 0.2     # ...and how near where it was it must reappear
HAND_STICK = 0.08     # the owner's hand stays theirs, while the (slower) body
                      # tracking hasn't caught up, if it moves less than this
                      # between frames
STICK_S = 0.3         # ...and was seen at most this long ago


def is_upright(world):
    """Wrist-to-middle-knuckle points up on screen, ignoring any lean toward
    or away from the camera. World y grows downward, like the image."""
    dx = world[L.MIDDLE_FINGER_MCP].x - world[L.WRIST].x
    dy = world[L.MIDDLE_FINGER_MCP].y - world[L.WRIST].y
    return -dy > 0 and math.degrees(math.atan2(abs(dx), -dy)) <= MAX_TILT_DEG


def in_claim_pose(hand):
    return (not any(hand.states.values()) and hand.palm < -MIN_BACK
            and is_upright(hand.world))


class Hand:
    """One detected hand for this frame. `states` are raw finger states."""

    def __init__(self, landmarks, world, label, states, palm):
        self.landmarks = landmarks
        self.world = world
        self.label = label
        self.states = states
        self.palm = palm
        self.person = None  # body id, filled in by the gate


class ControlGate:
    def __init__(self, people):
        self._people = people
        self.owner = None       # (body id, or None while lost; hand label)
        self.progress = {}      # Hand -> 0..1 of the hold, for the on-screen bar
        self._since = {}        # (body id, label) -> (start time, wrist x, wrist y)
        self._spent = set()     # keys that must break the pose before it counts again
        self._pose = {}         # key -> Debounced claim pose
        self._last_body = None  # (center, time) the owner's body was last seen
        self._last_wrist = None  # (x, y, time) owner's hand was last seen

    @property
    def owner_label(self):
        return self.owner[1] if self.owner else None

    @property
    def owner_lost(self):
        return self.owner is not None and self.owner[0] is None

    def update(self, hands, t):
        """`hands` is this frame's list of `Hand`. Returns the hand with
        control, or None."""
        self._follow_owner(t)
        self.progress = {}

        for hand in hands:
            hand.person = self._people.owner_of(hand.landmarks)
        self._stick_owner_hand(hands, t)
        keyed = {}
        for hand in hands:
            if hand.person is not None:
                keyed.setdefault((hand.person, hand.label), hand)

        for key in list(self._pose):
            if key not in keyed:
                del self._pose[key]
                self._since.pop(key, None)
        self._spent &= set(keyed)

        for key, hand in keyed.items():
            posed = self._pose.setdefault(key, Debounced(CONFIRM_FRAMES))
            if not posed.update(in_claim_pose(hand)):
                self._since.pop(key, None)
                self._spent.discard(key)
                continue
            # Only the owner can use the gesture while someone has control.
            if key in self._spent or (self.owner and key != self.owner):
                continue
            wrist = hand.landmarks[0]
            start = self._since.get(key)
            if start is None or math.hypot(wrist.x - start[1], wrist.y - start[2]) > MAX_TRAVEL:
                start = self._since[key] = (t, wrist.x, wrist.y)
            held = t - start[0]
            if held < HOLD_S:
                self.progress[hand] = held / HOLD_S
                continue
            self.owner = None if self.owner == key else key
            del self._since[key]
            # Keep holding and nothing more happens; break the pose first.
            self._spent.add(key)
            self._last_wrist = None
            self._last_body = (self._people.tracked[key[0]].center, t) if self.owner else None

        owned = keyed.get(self.owner) if self.owner else None
        if owned:
            self._last_wrist = (owned.landmarks[0].x, owned.landmarks[0].y, t)
        return owned

    def _stick_owner_hand(self, hands, t):
        """Body positions lag the hands a little. If the pose result hasn't
        placed the owner's hand on any body, keep it theirs as long as it is
        where their hand just was."""
        if (not self.owner or self.owner[0] is None or self._last_wrist is None
                or t - self._last_wrist[2] > STICK_S):
            return
        pid, label = self.owner
        if any(h.person == pid and h.label == label for h in hands):
            return
        for hand in hands:
            wrist = hand.landmarks[0]
            if (hand.person is None and hand.label == label
                    and math.hypot(wrist.x - self._last_wrist[0],
                                   wrist.y - self._last_wrist[1]) < HAND_STICK):
                hand.person = pid
                return

    def _follow_owner(self, t):
        if not self.owner:
            return
        pid, label = self.owner
        tracked = self._people.tracked
        if pid in tracked:
            self._last_body = (tracked[pid].center, t)
            return
        # Their body is gone. Wait a little for it to come back where it was.
        if self._last_body is None or t - self._last_body[1] > LOST_GRACE_S:
            self.owner = None
            return
        where = self._last_body[0]
        for other in self._people.visible:
            if math.dist(tracked[other].center, where) < RETURN_DIST:
                self.owner = (other, label)
                self._last_body = (tracked[other].center, t)
                return
        self.owner = (None, label)
