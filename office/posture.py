"""
Sitting, standing or walking -- from body keypoints, with no training.

    classify_frame(person)      one frame -> sitting / standing / unknown
    PostureTracker.update(...)  adds walking (needs motion over time) and smooths

THE RULE. Everything is measured in TORSO LENGTHS (mid-shoulder to mid-hip), never in
pixels, so a person near the camera and one across the room are judged alike.

  * Knees visible: how far the knee sits BELOW the hip. Standing, the thigh hangs
    straight down, roughly 0.8-1.0 torso lengths. Sitting, it points forward, so on
    screen the knee is nearly level with the hip. This also holds for a camera looking
    down from the ceiling: a forward-pointing thigh foreshortens, it does not hang.

  * Knees hidden (behind a desk -- the usual office case): a TALL box, over ~2.8 torso
    lengths, can only be someone standing. A SHORT box proves nothing: a seated person
    and a standing person behind a desk both show only head to waist. Measured on
    MM-Office, reading a short box as "sitting" labelled people standing behind desks
    as seated. So a short box with hidden knees is unknown, and the event log carries
    the last known posture through it.

  * Neither clear: "unknown". Saying unknown is better than guessing -- downstream,
    a wrong "standing" becomes a wrong "stood up" event.

STILLNESS. The commonest office view is a ceiling camera looking down at someone at a
desk: torso visible, knees under the desk, every frame unknown by the rules above. What
does tell us something is that they have not moved. People rarely stand motionless for
long, so a person with no posture evidence who has stayed put for STILL_SECONDS is
called sitting. It can be fooled -- someone standing still behind a counter -- which is
why it applies only when the keypoints said nothing.

Walking is not a pose but a speed: the hips moving more than WALK_SPEED torso lengths
per second, measured as displacement across the window rather than summed frame steps,
so keypoint jitter on a seated person does not add up to a walk.
"""

from collections import deque

import numpy as np

from .perception import L_HIP, L_KNEE, L_SHOULDER, R_HIP, R_KNEE, R_SHOULDER

SITTING, STANDING, WALKING, UNKNOWN = "sitting", "standing", "walking", "unknown"
POSTURES = (SITTING, STANDING, WALKING)

MIN_CONF = 0.4            # keypoint confidence below this is treated as not seen
KNEE_DROP_STANDING = 0.6  # knee this many torso lengths below the hip -> standing
KNEE_DROP_SITTING = 0.4   # ... less than this -> sitting
BOX_STANDING = 2.8        # box height in torso lengths, when the knees are hidden
WALK_SPEED = 1.0          # torso lengths per second
STILL_SPEED = 0.3         # below this the person is not going anywhere
STILL_SECONDS = 5.0       # this long still, with no posture evidence -> sitting


def midpoint(kp, a, b, min_conf=MIN_CONF):
    """Mean of whichever of two keypoints is visible; None if neither is."""
    seen = [kp[i, :2] for i in (a, b) if kp[i, 2] >= min_conf]
    return np.mean(seen, axis=0) if seen else None


def torso_length(kp):
    shoulders, hips = midpoint(kp, L_SHOULDER, R_SHOULDER), midpoint(kp, L_HIP, R_HIP)
    if shoulders is None or hips is None:
        return None
    length = float(np.linalg.norm(shoulders - hips))
    return length if length > 1.0 else None


def classify_frame(person):
    """One frame, no history: sitting, standing or unknown."""
    kp = person.keypoints
    torso = torso_length(kp)
    if torso is None:
        return UNKNOWN

    hips, knees = midpoint(kp, L_HIP, R_HIP), midpoint(kp, L_KNEE, R_KNEE)
    if knees is not None:
        drop = (knees[1] - hips[1]) / torso
        if drop >= KNEE_DROP_STANDING:
            return STANDING
        if drop <= KNEE_DROP_SITTING:
            return SITTING
        return UNKNOWN

    x1, y1, x2, y2 = person.box
    return STANDING if (y2 - y1) / torso >= BOX_STANDING else UNKNOWN


class HeadDrop:
    """Sit-down and stand-up from the shoulders alone, for when the legs are hidden.

    Sitting drops the shoulders by about a thigh's length within a second or two, from
    any viewing angle, even from behind a chair. The feet must stay put: walking towards
    a high camera also moves a person down the image.
    """

    def __init__(self, drop=0.6, before_s=1.5, recent_s=0.4, window_s=2.5, foot_limit=0.5,
                 cooldown_s=3.0):
        self.drop = drop
        self.before_s = before_s
        self.recent_s = recent_s
        self.window_s = window_s
        self.foot_limit = foot_limit
        self.cooldown_s = cooldown_s
        self.history = {}
        self.last_fired = {}

    def update(self, person, t):
        """Returns ("sit down" or "stand up", when it happened), or None."""
        kp = person.keypoints
        shoulders, torso = midpoint(kp, L_SHOULDER, R_SHOULDER), torso_length(kp)
        if shoulders is None or torso is None:
            return None
        x1, _, x2, y2 = person.box
        history = self.history.setdefault(person.track_id, deque())
        history.append((t, shoulders[1], (x1 + x2) / 2, y2, torso))
        while t - history[0][0] > self.window_s:
            history.popleft()
        if t - history[0][0] < self.before_s or t - self.last_fired.get(person.track_id, -1e9) < self.cooldown_s:
            return None

        recent = [h for h in history if t - h[0] <= self.recent_s]
        earlier = [h for h in history if t - h[0] >= self.before_s]
        if not recent or not earlier:
            return None
        before, now = np.mean(earlier, axis=0), np.mean(recent, axis=0)
        scale = float(np.median([h[4] for h in history]))
        if np.hypot(now[2] - before[2], now[3] - before[3]) > self.foot_limit * scale:
            return None
        change = (now[1] - before[1]) / scale
        if abs(change) < self.drop:
            return None
        self.last_fired[person.track_id] = t
        return ("sit down" if change > 0 else "stand up"), float((now[0] + before[0]) / 2)

    def forget(self, track_id):
        self.history.pop(track_id, None)
        self.last_fired.pop(track_id, None)


class PostureTracker:
    """Per-person history: adds walking, and votes over a window so one bad frame
    doesn't flip the label."""

    def __init__(self, window_s=1.0, walk_speed=WALK_SPEED, still_speed=STILL_SPEED,
                 still_seconds=STILL_SECONDS):
        self.window_s = window_s
        self.walk_speed = walk_speed
        self.still_speed = still_speed
        self.still_seconds = still_seconds
        self.history = {}       # track_id -> deque of (t, static label, hip xy, torso)
        self.still_since = {}   # track_id -> when they stopped moving
        self.inferred = {}      # track_id -> onset, while SITTING comes from stillness

    def update(self, person, t, infer_sitting=True):
        """infer_sitting=False stops stillness from calling this person seated, for
        cameras with marked seats when the person is not at one."""
        kp = person.keypoints
        anchor = midpoint(kp, L_HIP, R_HIP)
        if anchor is None:
            x1, y1, x2, y2 = person.box
            anchor = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        history = self.history.setdefault(person.track_id, deque())
        history.append((t, classify_frame(person), anchor, torso_length(kp)))
        while history and t - history[0][0] > self.window_s:
            history.popleft()

        speed = self._speed(history)
        if speed < self.still_speed:
            self.still_since.setdefault(person.track_id, t)
        else:
            self.still_since.pop(person.track_id, None)

        self.inferred.pop(person.track_id, None)
        if speed > self.walk_speed:
            return WALKING
        votes = [label for _, label, _, _ in history if label != UNKNOWN]
        if votes:
            # Ties go to the most recent label. Not max(set(votes), ...): set order of
            # strings changes with Python's per-process hash seed, so a tied vote came
            # out differently from one run to the next on identical input.
            return max(reversed(votes), key=votes.count)
        still_since = self.still_since.get(person.track_id, t)
        if infer_sitting and t - still_since >= self.still_seconds:
            self.inferred[person.track_id] = still_since
            return SITTING
        return UNKNOWN

    def onset(self, track_id):
        """When the current label really began, if that is earlier than now.

        Stillness can only call someone seated after STILL_SECONDS, but they sat down
        when they stopped moving. Without this, every inferred sit-down is logged five
        seconds late -- outside any reasonable scoring tolerance."""
        return self.inferred.get(track_id)

    def _speed(self, history):
        """Torso lengths per second, end to end across the window."""
        if len(history) < 2:
            return 0.0
        (t0, _, p0, _), (t1, _, p1, _) = history[0], history[-1]
        torsos = [torso for *_, torso in history if torso]
        if t1 - t0 < 0.3 or not torsos:
            return 0.0      # too short a span to tell motion from jitter
        return float(np.linalg.norm(p1 - p0)) / float(np.median(torsos)) / (t1 - t0)

    def forget(self, track_id):
        self.history.pop(track_id, None)
        self.still_since.pop(track_id, None)
        self.inferred.pop(track_id, None)
