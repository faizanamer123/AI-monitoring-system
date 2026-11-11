"""
What is a person DOING, as opposed to their posture -- from pose + nearby objects.

    on_call       a phone at the head, with a hand raised to it
    using_phone   a phone at a hand, away from the head (texting, scrolling)
    working       seated with a laptop, keyboard, monitor or mouse within reach

Anything else is "unknown" -- no activity we recognise, which the event log does not
store as a segment. Activity runs IN PARALLEL with posture: a person can be sitting and
on_call at once.

Distances are in torso lengths (see posture.py), so the rules hold near and far from
the camera. The phone outranks the computer: someone texting at their desk is on their
phone, not working, which is the distinction a productivity view needs.

What these rules do NOT claim: a phone hidden by the hand that holds it to the ear is
not seen, so that call is missed rather than guessed from a raised arm alone -- a hand
at the head is also someone scratching it, resting their chin, or adjusting glasses.
"""

from collections import deque

import numpy as np

from .objects import COMPUTER, centre
from .perception import L_EAR, L_WRIST, NOSE, R_EAR, R_WRIST
from .posture import MIN_CONF, SITTING, UNKNOWN, torso_length

ON_CALL, USING_PHONE, WORKING = "on_call", "using_phone", "working"
ACTIVITIES = (ON_CALL, USING_PHONE, WORKING)

# A phone held in front of the chest is within 0.7 torso lengths of the head, so that
# limit called most phone users "on a call": 316 false alarms against 40 real calls on MEVA.
CALL_DIST = 0.4     # phone centre within this many torso lengths of the head
CALL_HAND = 0.6     # and a wrist raised to within this many of the head
HOLD_DIST = 0.6     # ... of a wrist
REACH = 0.5         # computer centre within the person box widened by this fraction
# How detected hands (office/hands.py) combine with pose wrists as evidence that a phone is
# held, or raised to the head: "off" (wrists only), "or", "and", or "hand" (hands only).
# With the hand detector running, "and" is used: on MEVA it cut phone false alarms from
# 241 to 157 (precision 0.51 -> 0.59, recall 0.53 -> 0.47). Without it there are no hands
# to require, so wrists alone decide.
HAND_MODE = "off"
WITH_HANDS = "and"


def _hand_on(phone_box, hand_box, widen=0.2):
    """Does the phone touch this hand, allowing a little slack around the hand box?"""
    x1, y1, x2, y2 = hand_box
    mx, my = (x2 - x1) * widen, (y2 - y1) * widen
    return not (phone_box[2] < x1 - mx or phone_box[0] > x2 + mx
                or phone_box[3] < y1 - my or phone_box[1] > y2 + my)


def _evidence(by_wrist, by_hand, mode):
    return {"off": by_wrist, "or": by_wrist or by_hand, "and": by_wrist and by_hand,
            "hand": by_hand}[mode]


def _head(kp):
    seen = kp[NOSE:R_EAR + 1]
    seen = seen[seen[:, 2] >= MIN_CONF]
    return seen[:, :2].mean(axis=0) if len(seen) else None


def _scale(person):
    """Torso length, or a third of the box height when the torso is not visible."""
    torso = torso_length(person.keypoints)
    return torso if torso else (person.box[3] - person.box[1]) / 3


def _near_person(box, person, widen):
    x1, y1, x2, y2 = person.box
    mx, my = (x2 - x1) * widen, (y2 - y1) * widen
    cx, cy = centre(box)
    return x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my


def classify_activity(person, posture, things, hand_mode=None):
    """One frame: on_call / using_phone / working / unknown."""
    hand_mode = hand_mode or HAND_MODE
    kp = person.keypoints
    scale = _scale(person)
    head = _head(kp)
    wrists = [kp[i, :2] for i in (L_WRIST, R_WRIST) if kp[i, 2] >= MIN_CONF]

    hands = [t for t in things if t.name == "hand" and _near_person(t.box, person, 0.25)]
    holding = False
    for phone in (t for t in things if t.name == "cell phone" and _near_person(t.box, person, 0.25)):
        where = centre(phone.box)
        in_hand = any(_hand_on(phone.box, h.box) for h in hands)
        if head is not None and np.linalg.norm(where - head) <= CALL_DIST * scale and _evidence(
                any(np.linalg.norm(w - head) <= CALL_HAND * scale for w in wrists),
                in_hand and any(np.linalg.norm(centre(h.box) - head) <= CALL_HAND * scale
                                for h in hands), hand_mode):
            return ON_CALL
        if _evidence(any(np.linalg.norm(where - w) <= HOLD_DIST * scale for w in wrists), in_hand,
                     hand_mode):
            holding = True
    if holding:
        return USING_PHONE

    if posture == SITTING and any(t.name in COMPUTER and _near_person(t.box, person, REACH)
                                  for t in things):
        return WORKING
    return UNKNOWN


class ActivityTracker:
    """Per-person vote over a short window, like PostureTracker, so one frame where
    the phone is covered does not end a call."""

    def __init__(self, window_s=1.5, hand_mode=None):
        self.window_s = window_s
        self.hand_mode = hand_mode
        self.history = {}

    def update(self, person, posture, things, t):
        history = self.history.setdefault(person.track_id, deque())
        history.append((t, classify_activity(person, posture, things, self.hand_mode)))
        while history and t - history[0][0] > self.window_s:
            history.popleft()
        votes = [label for _, label in history]
        # ties to the most recent label -- deterministic, see PostureTracker
        return max(reversed(votes), key=votes.count)

    def forget(self, track_id):
        self.history.pop(track_id, None)
