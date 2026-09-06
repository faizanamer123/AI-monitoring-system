"""
pipeline/tracker.py -- STAGE 4: follow hands through time.

    WEBCAM -> OpenCV frame
        -> 1. DETECT    boxes            HandDetection
        -> 2. SEGMENT   hand mask        HandMask
        -> 3. POSE      21 keypoints     HandPose
        -> 4. TRACK     ids, motion      TrackedHand      <- this file
        -> gesture / movement / trajectory

Detection is memoryless. It answers "where are the hands NOW" and nothing else, so
frame 100 and frame 101 come back as two unrelated lists of boxes. Every question
worth asking about a hand is about time: is it moving left, has the pinch been held
long enough to count, is this the same hand that was here a moment ago. Stitching
the per-frame lists into "hand 3 travelled from here to here" is this module's whole
job, and it is what turns a detector into a pipeline.

WHAT MAKES THIS HARD

Nothing here is deep learning; it is bookkeeping, and bookkeeping is where this kind
of pipeline actually breaks:

* A detector drops a hand for one or two frames all the time (motion blur, the hand
  half leaving the frame, a confidence wobble at the threshold). If a miss deletes
  the track, every id downstream renumbers and every gesture state machine resets.
  Hence `max_missing`: a track coasts, it does not die.
* Two hands in the same frame WILL cross over. Plain box overlap is ambiguous exactly
  at the moment of crossing, which is exactly the moment you must not get wrong, so
  association uses three signals rather than one: overlap against both the last box
  and the motion-predicted one, a strong handedness bias, and greedy best-first
  assignment.
* Velocity in pixels per FRAME is a silent lie. The same wave is 12 px/frame at 60 fps
  and 48 px/frame at 15 fps, so any threshold tuned on one machine is wrong on the
  next. Everything here is pixels per SECOND, measured against a real clock.

COORDINATE RULE (see pipeline/types.py): every coordinate this module reads, stores
or returns -- trajectory points, velocities, boxes -- is FULL-FRAME pixels. The single
exception in the contract is HandMask.mask, which is ROI-local and carries its own
origin; the tracker never touches a mask's pixels, it only carries the object through,
so ROI space cannot leak into a trajectory here.

USAGE

    tracker = HandTracker(max_missing=8, iou_threshold=0.25)
    while True:
        frame = ...                                   # owned by the runner
        dets = detector(frame)                        # stage 1
        poses = pose_estimator(frame, dets)           # stage 3, parallel to dets
        tracks = tracker.update(dets, poses, timestamp=time.monotonic())
        for t in tracks:
            print(t.track_id, t.speed, trajectory_direction(t))

See also pipeline/types.py (the shared contract), pipeline/detector.py (stage 1),
pipeline/pose.py (stage 3).
"""

import math
import time
from collections import deque

from pipeline.types import WRIST, TrackedHand

# --- defaults, all overridable per instance -------------------------------------
MAX_MISSING = 8             # frames a track may coast unmatched before it is dropped
IOU_THRESHOLD = 0.25        # minimum box overlap before an id may be reused
VELOCITY_WINDOW = 5         # observations in the velocity fit (~0.17 s at 30 fps)
TRAJECTORY_LEN = 64         # points kept per track; matches TrackedHand's own default
MAX_PREDICT_SECONDS = 0.25  # cap on how far a coasting track is extrapolated
HANDEDNESS_BONUS = 0.25     # ranking bonus when both sides agree on Left/Right
HANDEDNESS_PENALTY = 0.25   # ranking MULTIPLIER when both sides disagree
HANDEDNESS_MISMATCH_IOU = 0.80  # overlap required to match despite disagreeing
STATIC_DISTANCE = 20.0      # px of net travel below which a trajectory reads "static"

# Anchor kinds, recorded per observation. See _anchor() for why the distinction
# matters to velocity.
_ANCHOR_WRIST = "wrist"
_ANCHOR_CENTER = "center"


def box_iou(box_a, box_b):
    """Intersection over union of two (x1, y1, x2, y2) full-frame boxes.

    Areas follow types.HandDetection.area -- (x2 - x1) * (y2 - y1), no +1 -- so that
    IoU and .area cannot disagree about how big a hand is. (The EgoHands ground-truth
    helper get_bounding_boxes uses the inclusive-pixel convention with a +1; the one
    pixel of difference is far below any sane association threshold, but the two
    conventions must not be mixed inside a single formula.)

    Boxes arriving with the corners the wrong way round are normalised rather than
    silently returning a negative area, because a detector that emits x2 < x1 should
    show up as a bad match, not as a poisoned score.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1

    # NaN must be rejected explicitly. Every comparison with NaN is False, so a NaN
    # box slips through `inter_w <= 0.0` and then through both matching gates below
    # (`overlap <= 0.0` and `overlap < gate` are also False), letting a garbage
    # detection silently capture an existing track.
    if not all(math.isfinite(v) for v in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _points_of(track_or_points):
    """Accept either a TrackedHand or a bare sequence of (x, y) points."""
    traj = getattr(track_or_points, "trajectory", track_or_points)
    return list(traj) if traj is not None else []


def trajectory_length(track):
    """Total path length in FULL-FRAME pixels, summed along the stored trajectory.

    This is path length, not displacement: a hand that waves left-right-left-right
    covers a long path and ends where it started. Compare with trajectory_direction(),
    which reports net displacement.

    Two honest limits, both consequences of the contract rather than of this function:
    the deque holds at most `trajectory_len` points (64 by default, ~2 s at 30 fps), so
    this is the length of the REMEMBERED path; and points are only appended on frames
    where the hand was actually detected, so the straight segment across a dropout is
    the chord, not the true path.
    """
    points = _points_of(track)
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    return total


def trajectory_direction(track, min_distance=STATIC_DISTANCE, window=None):
    """Dominant direction of travel: "up" | "down" | "left" | "right" | "static".

    Net displacement (first point to last), not per-step motion, so a jittery detector
    does not produce a direction that flickers every frame. Below `min_distance` px of
    net travel the answer is "static" -- without that floor a hand held still returns
    a random direction driven purely by detector noise, which is worse than useless to
    a gesture layer.

    IMAGE COORDINATES: y grows DOWNWARD, so a negative dy is "up". This is the classic
    sign error in this function and the reason it exists as one shared helper instead
    of being reimplemented at each call site.

    Args:
        min_distance: px of net travel below which the answer is "static".
        window: use only the last N trajectory points. Default None = the whole
            remembered trajectory. Pass a small window (say 8) to ask "which way is
            this hand moving right now" rather than "where has it been".

    An exact |dx| == |dy| diagonal is reported as horizontal; the tie has to break
    somewhere and a consistent rule beats a coin toss.
    """
    points = _points_of(track)
    if window is not None:
        if window < 2:
            return "static"
        points = points[-int(window):]
    if len(points) < 2:
        return "static"

    x0, y0 = points[0]
    x1, y1 = points[-1]
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    if math.hypot(dx, dy) < float(min_distance):
        return "static"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


class _TrackState:
    """Per-track bookkeeping the TrackedHand contract deliberately does not carry.

    TrackedHand is the pipeline's shared output type; it is not the tracker's scratch
    space. Observation history, handedness votes and anchor kind live here instead, so
    that stage 4's internals never leak into what stages 5+ read.
    """

    __slots__ = ("hand", "obs", "anchor_kind", "votes", "last_time")

    def __init__(self, hand, window):
        self.hand = hand
        self.obs = deque(maxlen=window)   # (t_seconds, x, y), full-frame px
        self.anchor_kind = None
        self.votes = {"Left": 0, "Right": 0}
        self.last_time = None

    @property
    def handedness(self):
        """Majority-vote handedness over the track's life, or "unknown".

        A single frame's handedness is not trustworthy -- MediaPipe flips Left/Right
        on a hand seen edge-on, and a flip that is allowed to re-label the track would
        then push the NEXT frame's correct label into a mismatch penalty and split the
        track in two. Voting over the whole track makes one bad frame a non-event.
        """
        left, right = self.votes["Left"], self.votes["Right"]
        if left == right:
            return "unknown"
        return "Left" if left > right else "Right"


class HandTracker:
    """Assigns stable ids to hands across frames (multi-object tracking, SORT-style).

    Per frame: score every (existing track, new detection) pair, greedily take the
    best pairs, coast the tracks nobody claimed, and open a new id for the detections
    nobody claimed.

    ASSOCIATION SCORE. Overlap alone is not enough at the one moment that matters --
    two hands crossing -- so the affinity of a pair is:

        1. The BETTER of two overlaps: the detection against the track's last box, and
           the detection against its MOTION-PREDICTED box (the last box shifted by the
           track's smoothed velocity times the real elapsed time). A hand moving
           80 px/frame has almost no overlap with where it was but near-perfect overlap
           with where it was going; a hand REVERSING -- a wave -- is the opposite, and
           gets extrapolated straight past the truth. Scoring both and keeping the max
           means prediction can only add matches, never take them away. Prediction is
           capped at MAX_PREDICT_SECONDS so a long-coasting track does not extrapolate
           to Mars.
        2. A floor: pairs below `iou_threshold` overlap are not matchable at all.
           Handedness NEVER creates a match that overlap does not support -- it only
           decides who gets one -- so a Left hand cannot capture a Left track on the
           far side of the frame.
        3. Handedness, applied only when BOTH the track's voted handedness and the
           detection's are known. Agreement adds `handedness_bonus` to the ranking
           score, so where two tracks compete for one detection the correctly-handed
           one wins even on slightly worse overlap. Disagreement raises that pair's
           gate to `handedness_mismatch_iou` (0.80) and drops its ranking score by
           `handedness_penalty`. A Left hand inheriting a Right hand's id is the exact
           failure this prevents: at the moment two hands cross, overlap says "either",
           and handedness says which.

           Why a raised gate rather than a hard veto: detectors flip Left/Right on a
           hand seen edge-on, and vetoing a flipped label would tear one hand's track
           in two every time it happened. A pair that disagrees on handedness but
           overlaps by 0.80+ is almost certainly one hand whose label wobbled, not two
           hands that swapped places -- hands cannot teleport onto each other. Set
           handedness_mismatch_iou above 1.0 for a true veto.

    GREEDY, NOT HUNGARIAN. Highest affinity first, each track and detection used once.
    A webcam frame holds two to four hands; on a matrix that small, greedy and optimal
    assignment agree except in cases the gating threshold has already thrown out, and
    greedy costs a few microseconds with no scipy dependency in the real-time path.
    (The one case where they differ -- two hands overlapping so heavily that swapping
    the assignment lowers total cost -- is exactly where motion prediction and
    handedness are doing the deciding anyway.) If a future stage ever tracks ten
    hands, swap _associate() for scipy.optimize.linear_sum_assignment on the same
    affinity matrix; nothing else changes.

    WHAT update() RETURNS. The tracks backed by a detection in THIS frame -- matched
    and newly opened alike -- in id order, i.e. the hands actually on screen now.
    Coasting tracks stay alive and keep their ids but are NOT returned, because drawing
    a coasted box paints a hand where the detector says there is none; read
    `tracker.tracks` when you want those too. The TrackedHand objects are mutated in
    place across frames, so a caller that stashes last frame's list is holding this
    frame's data.
    """

    def __init__(self, max_missing=MAX_MISSING, iou_threshold=IOU_THRESHOLD,
                 velocity_window=VELOCITY_WINDOW, trajectory_len=TRAJECTORY_LEN,
                 handedness_bonus=HANDEDNESS_BONUS,
                 handedness_penalty=HANDEDNESS_PENALTY,
                 handedness_mismatch_iou=HANDEDNESS_MISMATCH_IOU,
                 predict=True, max_predict_seconds=MAX_PREDICT_SECONDS):
        """
        Args:
            max_missing: consecutive unmatched frames a track survives. 8 frames is
                ~0.27 s at 30 fps -- long enough to ride out blur and threshold
                wobble, short enough that a hand which left the frame does not
                capture the id of the next hand to appear in that corner.
            iou_threshold: minimum box overlap for reusing an id. Too high renumbers
                on fast motion; too low teleports ids between hands. For equal boxes of
                width W moving d px between frames, IoU = (W - d) / (W + d), so 0.25
                tolerates a hand crossing 0.6 of its own width per frame from a cold
                start -- and considerably more once prediction has a velocity to work
                with.
            velocity_window: observations in the velocity fit. Larger is smoother and
                laggier; 5 is ~0.17 s at 30 fps.
            trajectory_len: points remembered per track (deque maxlen).
            handedness_bonus / handedness_penalty: ranking bias for pairs that agree
                / disagree on Left-vs-Right. See the class docstring.
            handedness_mismatch_iou: overlap a pair must reach to match DESPITE
                disagreeing on handedness. Above 1.0 this becomes a hard veto.
            predict: motion-compensate the track box before scoring overlap.
            max_predict_seconds: cap on the extrapolation interval.
        """
        if max_missing < 0:
            raise ValueError(f"max_missing must be >= 0, got {max_missing}")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if velocity_window < 2:
            raise ValueError(f"velocity_window must be >= 2, got {velocity_window}")
        if trajectory_len < 1:
            raise ValueError(f"trajectory_len must be >= 1, got {trajectory_len}")

        self.max_missing = int(max_missing)
        self.iou_threshold = float(iou_threshold)
        self.velocity_window = int(velocity_window)
        self.trajectory_len = int(trajectory_len)
        self.handedness_bonus = float(handedness_bonus)
        self.handedness_penalty = float(handedness_penalty)
        self.handedness_mismatch_iou = float(handedness_mismatch_iou)
        self.predict = bool(predict)
        self.max_predict_seconds = float(max_predict_seconds)

        self._states = {}       # track_id -> _TrackState, insertion ordered
        self._next_id = 1       # ids start at 1: `if track_id:` on a real id is a
                                # bug magnet, and id 0 would read as "no track"
        self.frame_index = 0
        self.rejected_detections = 0

    # --- public API ---------------------------------------------------------------

    @property
    def tracks(self):
        """Every live track in id order, INCLUDING ones coasting through a dropout."""
        return [self._states[i].hand for i in sorted(self._states)]

    def handedness_of(self, track_id):
        """Majority-voted "Left" | "Right" | "unknown" for a live track.

        Prefer this over track.detection.handedness anywhere a decision sticks around
        (labels, gesture state machines): the per-frame value flickers, this does not.
        """
        state = self._states.get(track_id)
        return state.handedness if state else "unknown"

    def reset(self):
        """Forget every track and restart ids at 1 (new camera, new session)."""
        self._states.clear()
        self._next_id = 1
        self.frame_index = 0

    def update(self, detections, poses=None, masks=None, timestamp=None):
        """Advance the tracker by one frame.

        Args:
            detections: this frame's HandDetection list (may be empty or None).
            poses: HandPose per detection, POSITIONALLY PARALLEL to `detections`;
                None, or a list with None holes. A length mismatch raises ValueError
                rather than silently pairing hand 0's box with hand 1's landmarks --
                that misalignment is invisible in a demo and fatal in a gesture.
            masks: HandMask per detection, same parallel-list rule. Carried through
                untouched; the tracker never reads mask pixels, so its ROI-local
                coordinates cannot contaminate frame-space trajectories.
            timestamp: seconds as a float (time.monotonic() is ideal). Defaults to
                time.monotonic() when omitted. Velocity is px/SECOND, so a real clock
                is not optional -- but be consistent: do not mix your own timeline
                with the default one inside a single run.

        Returns:
            Every track backed by a detection this frame -- matched or newly opened --
            ordered by track_id. Tracks coasting through a dropout are alive but
            absent from this list; see the class docstring and `tracks`.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)
        if math.isnan(now) or math.isinf(now):
            raise ValueError(f"timestamp must be finite, got {timestamp!r}")

        dets = [] if detections is None else list(detections)
        poses = self._parallel(poses, len(dets), "poses")
        masks = self._parallel(masks, len(dets), "masks")

        # Drop degenerate and non-finite boxes before association. A zero-area box has
        # zero IoU with everything -- including a track opened from that same box -- so
        # it can never re-match. Left in, it opens a BRAND NEW track id on every frame
        # and _next_id grows without bound, while the thing it describes never gets a
        # stable id. A box with no area is not a hand; refusing it is the honest answer.
        keep = []
        for index, det in enumerate(dets):
            if det is None:
                continue
            box = getattr(det, "box", None)
            if box is None or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            keep.append(index)
        if len(keep) != len(dets):
            self.rejected_detections += len(dets) - len(keep)
            dets = [dets[i] for i in keep]
            poses = [poses[i] for i in keep]
            masks = [masks[i] for i in keep]

        self.frame_index += 1

        matches, unmatched_dets = self._associate(dets, now)

        for track_id, det_index in matches.items():
            self._observe(self._states[track_id], dets[det_index],
                          poses[det_index], masks[det_index], now)

        for track_id, state in self._states.items():
            if track_id not in matches:
                # Coasting: age counts frames SEEN, so it does not tick here. Only
                # `missing` moves, and it is what max_missing is measured against.
                state.hand.missing += 1

        seen_ids = set(matches)
        for det_index in unmatched_dets:
            fresh = self._spawn(dets[det_index], poses[det_index], masks[det_index], now)
            seen_ids.add(fresh.track_id)

        dead = [i for i, s in self._states.items() if s.hand.missing > self.max_missing]
        for track_id in dead:
            del self._states[track_id]

        # Matched tracks AND ids opened this frame: a hand that has just appeared is
        # on screen now, and a runner that only drew matched tracks would show every
        # new hand one frame late.
        return [self._states[i].hand for i in sorted(seen_ids)]

    # --- internals ----------------------------------------------------------------

    @staticmethod
    def _parallel(values, n, name):
        """Normalise an optional parallel list, or explain loudly why it is wrong."""
        if values is None:
            return [None] * n
        values = list(values)
        if len(values) != n:
            raise ValueError(
                f"{name} must be parallel to detections: got {len(values)} {name} "
                f"for {n} detections")
        return values

    def _associate(self, dets, now):
        """Greedy highest-affinity-first matching.

        Returns ({track_id: det_index}, [indices of detections nobody claimed]).
        """
        candidates = []
        for track_id, state in self._states.items():
            last = state.hand.detection.box
            predicted = self._predicted_box(state, now)
            track_hand = state.handedness
            for det_index, det in enumerate(dets):
                # The BETTER of "where it was" and "where it was heading". Prediction
                # alone is not safe: a hand that reverses -- which is to say, a hand
                # waving -- gets extrapolated the wrong way and its box lands where the
                # hand is not, so a pure predicted-box tracker renumbers on every
                # direction change. Taking the max means extrapolation can only ever
                # add a match, never remove one.
                overlap = box_iou(last, det.box)
                if predicted is not last:
                    overlap = max(overlap, box_iou(predicted, det.box))
                if overlap <= 0.0:
                    continue
                det_hand = getattr(det, "handedness", "unknown")
                known = track_hand in ("Left", "Right") and det_hand in ("Left", "Right")
                disagree = known and track_hand != det_hand

                # The gate is pure overlap; handedness can only tighten it, never
                # loosen it. The score below is for RANKING competing pairs.
                gate = self.iou_threshold
                if disagree:
                    gate = max(gate, self.handedness_mismatch_iou)
                if overlap < gate:
                    continue

                score = overlap
                if disagree:
                    score *= self.handedness_penalty
                elif known:
                    score += self.handedness_bonus
                candidates.append((score, track_id, det_index, state.hand.missing))

        # Descending affinity. Ties break towards the track seen most recently, then
        # the oldest id, then the first detection: a track that has been coasting for
        # six frames is sitting on a six-frame-old box, so on equal evidence the fresh
        # track is the better bet -- and a fully deterministic order is what makes a
        # tracker testable at all, since anything left to dict iteration order turns
        # into a heisenbug the first time a detector reorders its output.
        candidates.sort(key=lambda c: (-c[0], c[3], c[1], c[2]))

        matches = {}
        taken_dets = set()
        for _score, track_id, det_index, _missing in candidates:
            if track_id in matches or det_index in taken_dets:
                continue
            matches[track_id] = det_index
            taken_dets.add(det_index)

        unmatched = [i for i in range(len(dets)) if i not in taken_dets]
        return matches, unmatched

    def _predicted_box(self, state, now):
        """Where the track's box should be NOW, given its smoothed velocity.

        Constant-velocity extrapolation, in seconds, capped at max_predict_seconds.
        Without it, a hand moving faster than about half its own box width per frame
        has too little overlap with its previous position and gets a new id every
        frame; with it, the overlap is near-perfect. _associate() scores this box AND
        the un-extrapolated one and keeps the better, so a wrong prediction (a hand
        reversing) costs nothing. The cap matters because a track that has been
        coasting for eight frames holds a stale velocity, and an uncapped extrapolation
        would fling its box off-screen and then match nothing at all.

        Returns the box object itself when there is nothing to predict from, which is
        how _associate() knows it can skip the second overlap computation.
        """
        box = state.hand.detection.box
        if not self.predict or len(state.obs) < 2 or state.last_time is None:
            return box              # identity, so _associate() can skip the second IoU
        dt = now - state.last_time
        if dt <= 0.0:
            return box
        dt = min(dt, self.max_predict_seconds)
        vx, vy = state.hand.velocity
        x1, y1, x2, y2 = (float(v) for v in box)
        return (x1 + vx * dt, y1 + vy * dt, x2 + vx * dt, y2 + vy * dt)

    @staticmethod
    def _anchor(detection, pose):
        """The single point that represents this hand, in FULL-FRAME px.

        The WRIST landmark when a pose is available, the box centre otherwise -- and
        the kind is returned alongside because the two are NOT interchangeable. The
        wrist sits near the bottom edge of the hand box, tens of pixels from its
        centre, so a frame where the pose drops out shifts the anchor by that offset.
        Fed straight into a difference, that offset reads as a hand teleporting at
        several hundred px/s. _observe() uses the kind to keep velocity honest across
        the switch.

        Why the wrist at all: it is the stablest of the 21 landmarks. Fingertips move
        relative to the hand while the hand itself stays put (a wiggling finger is not
        a moving hand), and the box centre drifts whenever the fingers spread or the
        detector's box breathes. The wrist moves when the hand moves.
        """
        if pose is not None:
            points = getattr(pose, "points", None)
            if points is not None and len(points) > WRIST:
                x, y = points[WRIST][0], points[WRIST][1]
                return float(x), float(y), _ANCHOR_WRIST
        cx, cy = detection.center
        return float(cx), float(cy), _ANCHOR_CENTER

    def _observe(self, state, detection, pose, mask, now):
        """Fold this frame's observation into an existing track."""
        hand = state.hand
        hand.detection = detection
        hand.pose = pose
        hand.mask = mask
        hand.missing = 0
        hand.age += 1

        handedness = getattr(detection, "handedness", "unknown")
        if handedness in state.votes:
            state.votes[handedness] += 1

        x, y, kind = self._anchor(detection, pose)

        # Two history resets, both for the same reason: a velocity is only meaningful
        # between two observations of the SAME point measured on a forward clock.
        if kind != state.anchor_kind:
            state.obs.clear()               # wrist <-> centre switch, see _anchor()
            state.anchor_kind = kind
        if state.last_time is not None and now < state.last_time:
            state.obs.clear()               # clock went backwards (re-run, reset)

        state.obs.append((now, x, y))
        state.last_time = now
        hand.trajectory.append((x, y))

        # None means "not enough same-anchor history on a forward clock to measure":
        # carrying the previous velocity for a frame beats claiming the hand stopped
        # dead, because a hand whose pose flickered did not stop moving.
        velocity = self._fit_velocity(state.obs)
        if velocity is not None:
            hand.velocity = velocity

    @staticmethod
    def _fit_velocity(obs):
        """Least-squares slope of x(t) and y(t) over the window -> (vx, vy) px/SECOND.

        A raw last-minus-previous difference is unusable downstream: box jitter of a
        few px across a 33 ms frame is already ~100 px/s of pure noise, and gesture
        thresholds sit right in that band. Fitting a line over the window averages the
        jitter down while staying EXACT for constant velocity -- which is why a
        synthetic hand moving 300 px/s reads as exactly 300 px/s at 15 fps and at
        60 fps, the property that makes this frame-rate independent.

        Least squares over the window rather than an endpoint difference because the
        endpoint difference throws away the middle samples and lets the newest, noisiest
        sample set the answer.

        Returns None when the window holds fewer than two observations or when every
        observation shares a timestamp (dt = 0 is not a slow hand, it is no clock).
        """
        n = len(obs)
        if n < 2:
            return None
        mean_t = sum(o[0] for o in obs) / n
        denom = sum((o[0] - mean_t) ** 2 for o in obs)
        if denom <= 0.0:
            return None
        mean_x = sum(o[1] for o in obs) / n
        mean_y = sum(o[2] for o in obs) / n
        vx = sum((o[0] - mean_t) * (o[1] - mean_x) for o in obs) / denom
        vy = sum((o[0] - mean_t) * (o[2] - mean_y) for o in obs) / denom
        return (vx, vy)

    def _spawn(self, detection, pose, mask, now):
        """Open a new id for a detection nothing claimed.

        New tracks are live immediately rather than waiting N frames to be "confirmed".
        A confirmation delay buys fewer spurious ids at the cost of latency on every
        real hand, and for an interactive camera pipeline latency is the expensive
        side of that trade. Spurious ids die on their own via max_missing.
        """
        x, y, kind = self._anchor(detection, pose)
        hand = TrackedHand(
            track_id=self._next_id,
            detection=detection,
            pose=pose,
            mask=mask,
            trajectory=deque([(x, y)], maxlen=self.trajectory_len),
            velocity=(0.0, 0.0),   # one observation is a position, not a motion
            age=1,
            missing=0,
        )
        state = _TrackState(hand, self.velocity_window)
        state.anchor_kind = kind
        state.obs.append((now, x, y))
        state.last_time = now
        handedness = getattr(detection, "handedness", "unknown")
        if handedness in state.votes:
            state.votes[handedness] += 1
        self._states[self._next_id] = state
        self._next_id += 1
        return hand
