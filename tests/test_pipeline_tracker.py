"""Tests for pipeline/tracker.py -- STAGE 4, hands followed through time.

Tracking is pure logic: no camera, no model, no GPU. That is a gift, because it means
every failure mode can be reproduced exactly instead of being chased around a live
webcam. The synthetic hands here move on paths whose true velocity and true identity
are known to the frame, so "did the id survive the crossover" and "is 300 px/s
reported as 300 px/s" are checked against arithmetic rather than against a screenshot.

Three things get the hardest scrutiny, because they are what silently breaks:

  * IDENTITY through the moments where overlap is ambiguous -- two hands crossing, a
    hand vanishing for a few frames, detections arriving in a different order.
  * VELOCITY UNITS. Every velocity assertion is run at more than one frame rate. A
    per-frame velocity passes at 30 fps and is wrong by 2x at 60; only px/second
    survives both.
  * COORDINATE SPACE. Trajectories are full-frame pixels; HandMask is ROI-local with
    an origin. The tests deliberately use masks whose origin is far from (0, 0), so a
    tracker that leaked ROI coordinates into a trajectory would fail loudly instead of
    passing on a mask that happens to start at the top-left corner.

The last block runs on the real EgoHands annotations: real hands, real ground-truth
identity (own-left / own-right / other-left / other-right), real consecutive frames.
That is where the honest quality number comes from.

Run with:   .venv/bin/python -m pytest tests/test_pipeline_tracker.py -v
"""

import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO, needs_dataset
from pipeline.types import WRIST, HandDetection, HandMask, HandPose, TrackedHand
from pipeline.tracker import (
    HandTracker,
    box_iou,
    trajectory_direction,
    trajectory_length,
)

FPS = 30.0
FRAME_SHAPE = (720, 1280, 3)     # every EgoHands frame, and the pipeline's frame size


# --------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------

def box(cx, cy, w=100, h=100):
    """A detection box centred on (cx, cy), in full-frame pixels."""
    return (int(round(cx - w / 2)), int(round(cy - h / 2)),
            int(round(cx + w / 2)), int(round(cy + h / 2)))


def det(cx, cy, w=100, h=100, score=0.9, handedness="unknown"):
    return HandDetection(box=box(cx, cy, w, h), score=score, handedness=handedness)


def pose_at(x, y):
    """A HandPose whose WRIST sits at (x, y) full-frame; other joints are unused here."""
    points = np.zeros((21, 2), dtype=np.float32)
    points[WRIST] = (x, y)
    return HandPose(points=points)


def only(tracks):
    assert len(tracks) == 1, f"expected one track, got {[t.track_id for t in tracks]}"
    return tracks[0]


def ids_of(tracks):
    return [t.track_id for t in tracks]


def track_holding(tracks, detection):
    """The track that was given this exact detection object this frame, or None."""
    for t in tracks:
        if t.detection is detection:
            return t
    return None


# --------------------------------------------------------------------------------
# box_iou -- the primitive everything else is built on
# --------------------------------------------------------------------------------

def test_box_iou_identical_boxes_is_one():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_box_iou_disjoint_boxes_is_zero():
    assert box_iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_box_iou_touching_edges_is_zero():
    # Sharing an edge is not overlapping; a > 0 here would mean a tracker could match
    # two hands that merely stand next to each other.
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_box_iou_half_overlap():
    # [0,10] vs [5,15] in x, full overlap in y: inter 50, union 150.
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_box_iou_normalises_inverted_corners():
    assert box_iou((10, 10, 0, 0), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_box_iou_degenerate_box_is_zero_not_nan():
    value = box_iou((5, 5, 5, 5), (0, 0, 10, 10))
    assert value == 0.0 and not math.isnan(value)


# --------------------------------------------------------------------------------
# basic bookkeeping
# --------------------------------------------------------------------------------

def test_empty_frame_returns_no_tracks():
    tracker = HandTracker()
    assert tracker.update([], timestamp=0.0) == []
    assert tracker.update(None, timestamp=1 / FPS) == []


def test_new_hand_is_reported_on_its_first_frame():
    # A confirmation delay would make every hand appear one frame late; the runner
    # draws what update() returns, so a new hand must be in that list immediately.
    tracker = HandTracker()
    tracks = tracker.update([det(400, 300)], timestamp=0.0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1          # ids start at 1, never 0
    assert tracks[0].age == 1
    assert tracks[0].missing == 0


def test_returned_object_matches_the_tracked_hand_contract():
    tracker = HandTracker()
    detection = det(400, 300, handedness="Left")
    pose = pose_at(400, 340)
    mask = HandMask(mask=np.full((10, 10), 255, np.uint8), origin=(350, 250))
    track = only(tracker.update([detection], [pose], [mask], timestamp=0.0))

    assert isinstance(track, TrackedHand)
    assert track.detection is detection
    assert track.pose is pose
    assert track.mask is mask
    assert isinstance(track.trajectory, deque)
    assert track.trajectory.maxlen == 64          # types.TrackedHand's own default
    assert isinstance(track.velocity, tuple) and len(track.velocity) == 2
    assert track.gesture == "none" and track.gesture_confidence == 0.0


def test_stationary_hand_keeps_one_id_and_ages():
    tracker = HandTracker()
    for i in range(30):
        track = only(tracker.update([det(400, 300)], timestamp=i / FPS))
        assert track.track_id == 1
        assert track.age == i + 1
        assert track.missing == 0


def test_two_separate_hands_get_two_ids():
    tracker = HandTracker()
    tracks = tracker.update([det(200, 300), det(900, 300)], timestamp=0.0)
    assert sorted(ids_of(tracks)) == [1, 2]


def test_tracks_are_returned_in_id_order():
    tracker = HandTracker()
    tracker.update([det(200, 300), det(900, 300)], timestamp=0.0)
    tracks = tracker.update([det(900, 300), det(200, 300)], timestamp=1 / FPS)
    assert ids_of(tracks) == sorted(ids_of(tracks))


def test_ids_are_never_reused():
    tracker = HandTracker(max_missing=0)
    seen = set()
    for i in range(5):
        track = only(tracker.update([det(100 + 400 * (i % 2), 300)], timestamp=i / FPS))
        assert track.track_id not in seen
        seen.add(track.track_id)


def test_reset_clears_tracks_and_restarts_ids():
    tracker = HandTracker()
    tracker.update([det(400, 300)], timestamp=0.0)
    tracker.reset()
    assert tracker.tracks == []
    assert only(tracker.update([det(400, 300)], timestamp=0.0)).track_id == 1


def test_track_objects_are_stable_across_frames():
    # The runner may hold a reference between frames; the same hand must be the same
    # object, mutated in place, not a fresh copy with the same number on it.
    tracker = HandTracker()
    first = only(tracker.update([det(400, 300)], timestamp=0.0))
    second = only(tracker.update([det(405, 300)], timestamp=1 / FPS))
    assert first is second


# --------------------------------------------------------------------------------
# association: crossings, ordering, handedness
# --------------------------------------------------------------------------------

def test_two_hands_keep_their_ids_through_a_crossover():
    """The headline requirement: hands that swap sides do not swap ids.

    One hand runs left-to-right, the other right-to-left, 25 px per frame, and they
    pass through each other at frame 4. Handedness is unknown, so overlap and motion
    are doing all the work.
    """
    tracker = HandTracker()
    left_id = right_id = None
    for i in range(9):
        moving_right = det(300 + 25 * i, 300)
        moving_left = det(500 - 25 * i, 300)
        tracks = tracker.update([moving_right, moving_left], timestamp=i / FPS)
        assert len(tracks) == 2, f"lost a hand at frame {i}"
        a = track_holding(tracks, moving_right).track_id
        b = track_holding(tracks, moving_left).track_id
        if left_id is None:
            left_id, right_id = a, b
        assert (a, b) == (left_id, right_id), f"ids swapped at frame {i}"
    assert left_id != right_id


@pytest.mark.parametrize("speed", [10, 25, 40, 55])
def test_crossover_holds_at_several_speeds(speed):
    tracker = HandTracker()
    expected = None
    for i in range(13):
        a = det(200 + speed * i, 300)
        b = det(200 + speed * 12 - speed * i, 340)
        tracks = tracker.update([a, b], timestamp=i / FPS)
        assert len(tracks) == 2
        pair = (track_holding(tracks, a).track_id, track_holding(tracks, b).track_id)
        expected = expected or pair
        assert pair == expected, f"identity lost at frame {i}, speed {speed}"


def test_crossover_with_handedness_keeps_ids():
    tracker = HandTracker()
    expected = None
    for i in range(13):
        left = det(200 + 45 * i, 300, handedness="Left")
        right = det(200 + 45 * 12 - 45 * i, 340, handedness="Right")
        tracks = tracker.update([left, right], timestamp=i / FPS)
        assert len(tracks) == 2
        pair = (track_holding(tracks, left).track_id, track_holding(tracks, right).track_id)
        expected = expected or pair
        assert pair == expected
        # and the id keeps the handedness it was born with
        assert tracker.handedness_of(pair[0]) == "Left"
        assert tracker.handedness_of(pair[1]) == "Right"


def test_left_hand_does_not_inherit_a_right_hands_id():
    """A Right detection sitting on a Left track's box must not take its id."""
    tracker = HandTracker()
    tracker.update([det(400, 300, handedness="Left")], timestamp=0.0)
    # 0.6 IoU with the previous box -- plenty to match on overlap alone.
    tracks = tracker.update([det(425, 300, handedness="Right")], timestamp=1 / FPS)
    assert only(tracks).track_id == 2, "the Right hand inherited the Left hand's id"


def test_handedness_only_gates_it_never_invents_a_match():
    """Agreeing on handedness must not pull an id across the frame.

    The bonus biases WHO gets a match; it must never create one that overlap does not
    support, or a hand leaving the frame would hand its id to any same-handed hand
    that appeared somewhere else.
    """
    tracker = HandTracker()
    tracker.update([det(200, 300, handedness="Left")], timestamp=0.0)
    tracks = tracker.update([det(1000, 300, handedness="Left")], timestamp=1 / FPS)
    assert only(tracks).track_id == 2


def test_single_frame_handedness_flip_does_not_split_a_still_hand():
    """A near-identical box with a flipped label is a detector wobble, not a new hand.

    MediaPipe flips Left/Right on a hand seen edge-on. Vetoing the flip outright would
    renumber the hand for one frame and renumber it back, which is worse for anything
    downstream than tolerating a label that the majority vote will outvote anyway.
    """
    tracker = HandTracker()
    tracker.update([det(400, 300, handedness="Left")], timestamp=0.0)
    tracker.update([det(400, 300, handedness="Left")], timestamp=1 / FPS)
    flipped = only(tracker.update([det(402, 300, handedness="Right")], timestamp=2 / FPS))
    assert flipped.track_id == 1
    assert tracker.handedness_of(1) == "Left", "one bad frame outvoted the whole track"


def test_hard_handedness_veto_is_configurable():
    tracker = HandTracker(handedness_mismatch_iou=1.01)
    tracker.update([det(400, 300, handedness="Left")], timestamp=0.0)
    tracks = tracker.update([det(400, 300, handedness="Right")], timestamp=1 / FPS)
    assert only(tracks).track_id == 2


def test_detection_order_does_not_affect_ids():
    """A detector that reorders its output must not renumber the hands.

    Detection order is an implementation detail of stage 1 (NMS ordering, batch
    ordering); anything that leans on it produces a bug that only shows up when the
    detector is swapped.
    """
    positions = [(200, 300), (600, 320), (1000, 280)]
    rng = random.Random(11)
    tracker = HandTracker()
    first = {}
    for i in range(12):
        dets = [det(x + 3 * i, y) for x, y in positions]
        shuffled = dets[:]
        rng.shuffle(shuffled)
        tracks = tracker.update(shuffled, timestamp=i / FPS)
        assert len(tracks) == 3
        for slot, d in enumerate(dets):
            tid = track_holding(tracks, d).track_id
            first.setdefault(slot, tid)
            assert tid == first[slot], f"hand {slot} renumbered at frame {i}"


def test_greedy_matching_prefers_the_better_overlap():
    """Two tracks competing for one detection: the closer one wins, the other coasts."""
    tracker = HandTracker()
    a = det(400, 300)
    b = det(520, 300)
    tracker.update([a, b], timestamp=0.0)
    settled = tracker.update([a, b], timestamp=1 / FPS)
    id_a = track_holding(settled, a).track_id
    tracks = tracker.update([det(410, 300)], timestamp=2 / FPS)
    assert only(tracks).track_id == id_a


def test_motion_prediction_survives_an_acceleration_that_plain_iou_cannot():
    """Why the tracker extrapolates before scoring overlap.

    A hand established at 60 px/frame that speeds up to 100 px/frame has 0.09 raw
    overlap with its previous box -- under any sane threshold. Shifted by its own
    measured velocity first, it overlaps by 0.5 and keeps its id.
    """
    def run(predict):
        tracker = HandTracker(predict=predict)
        ids, x = [], 100.0
        for i in range(12):
            ids.append(only(tracker.update([det(x, 300, w=120, h=120),],
                                           timestamp=i / FPS)).track_id)
            x += 60 if i < 4 else 100
        return ids

    assert set(run(predict=True)) == {1}, "prediction failed to hold an accelerating hand"
    assert len(set(run(predict=False))) > 1, "control case unexpectedly held; test is stale"


def test_waving_hand_keeps_its_id_through_reversals():
    """A wave is the case that punishes motion prediction, so it gets its own test.

    Constant-velocity extrapolation points a reversing hand exactly the wrong way: at
    the turn, the predicted box sits 50 px past the truth while the hand has gone back
    the other way. Scoring the last box as well as the predicted one is what keeps the
    id here -- this test failed on a predicted-box-only tracker, which renumbered the
    hand at every turn and made every wave a sequence of one-frame-long strangers.
    """
    xs = [400, 450, 500, 450, 400, 450, 500, 450, 400]
    for predict in (True, False):
        tracker = HandTracker(predict=predict)
        ids = []
        for i, cx in enumerate(xs):
            ids.append(only(tracker.update([det(cx, 300)], timestamp=i / FPS)).track_id)
        assert set(ids) == {1}, f"wave renumbered with predict={predict}: {ids}"
    assert trajectory_length(tracker.tracks[0]) == pytest.approx(400.0)


def test_zero_area_detections_are_rejected_not_tracked():
    """A degenerate box must be dropped, not given a track id.

    This test previously asserted the opposite -- that the degenerate box opens its own
    id -- which encoded a leak rather than a behaviour. A zero-area box has zero IoU
    with everything, INCLUDING a track opened from that same box, so it can never
    re-match: fed one per frame, the tracker minted a brand-new id every frame and
    _next_id grew without bound while the thing it described never got a stable id.
    A box with no area is not a hand, so the tracker refuses it and says so via
    .rejected_detections.
    """
    tracker = HandTracker(max_missing=1)

    for i in range(6):
        assert tracker.update(
            [HandDetection(box=(10, 10, 10, 10), score=0.5)], timestamp=i / FPS
        ) == [], "a zero-area box must not produce a track"
    assert tracker.rejected_detections == 6

    # and a real hand arriving afterwards still gets the FIRST id, because the
    # degenerate boxes never consumed any
    tracks = tracker.update([det(400, 300)], timestamp=6 / FPS)
    assert only(tracks).track_id == 1


def test_non_finite_boxes_are_rejected():
    """NaN slips through every comparison, so it must be rejected explicitly.

    Every comparison against NaN is False, so a NaN box passes the `inter_w <= 0`
    guard in box_iou and then both matching gates, letting garbage capture a track.
    """
    nan = float("nan")
    tracker = HandTracker()
    assert tracker.update([HandDetection(box=(nan, 10, 100, 100), score=0.9)],
                          timestamp=0.0) == []
    assert tracker.rejected_detections == 1
    assert box_iou((0, 0, 10, 10), (nan, 0, 10, 10)) == 0.0


def test_dead_tracks_are_freed():
    """Long-running memory check: a webcam session must not accumulate track state."""
    tracker = HandTracker(max_missing=2)
    for i in range(60):
        tracker.update([det(100 + (i % 6) * 300, 300)], timestamp=i / FPS)
    assert len(tracker.tracks) <= 3


def test_frame_index_counts_updates():
    tracker = HandTracker()
    for i in range(5):
        tracker.update([], timestamp=i / FPS)
    assert tracker.frame_index == 5


def test_cold_start_speed_limit_fails_safely():
    """Past ~0.6 box-widths per frame a NEW track cannot be associated by overlap.

    For equal boxes of width W moving by d, IoU = (W - d) / (W + d), so the default
    0.25 threshold breaks at d = 0.6 W: 72 px/frame for a 120 px hand, about 2160 px/s
    at 30 fps. This is a real limit and it is documented rather than hidden -- but the
    failure mode is a NEW id, never a wrong one, which is the safe direction. (Once a
    track has two observations, prediction pushes the limit far higher; see above.)
    """
    def ids_at(step):
        tracker = HandTracker()
        ids, x = [], 100.0
        for i in range(6):
            ids.append(only(tracker.update([det(x, 300, w=120, h=120)],
                                           timestamp=i / FPS)).track_id)
            x += step
        return ids

    assert set(ids_at(72)) == {1}          # 0.60 W: still associated
    assert len(set(ids_at(90))) == 6       # 0.75 W: a fresh id every frame


# --------------------------------------------------------------------------------
# dropouts: missing, age, max_missing
# --------------------------------------------------------------------------------

def test_hand_missing_for_three_frames_keeps_its_id():
    tracker = HandTracker(max_missing=8)
    tracker.update([det(400, 300)], timestamp=0.0)
    for i in range(1, 4):
        assert tracker.update([], timestamp=i / FPS) == []
    back = only(tracker.update([det(405, 300)], timestamp=4 / FPS))
    assert back.track_id == 1
    assert back.missing == 0, "missing must reset when the hand comes back"
    assert back.age == 2, "age counts frames SEEN, not frames elapsed"


def test_missing_counter_increments_while_coasting():
    tracker = HandTracker(max_missing=8)
    tracker.update([det(400, 300)], timestamp=0.0)
    for i in range(1, 5):
        tracker.update([], timestamp=i / FPS)
        coasting = only(tracker.tracks)          # alive, but not returned by update()
        assert coasting.missing == i
        assert coasting.age == 1


def test_coasting_track_is_alive_but_not_reported():
    tracker = HandTracker(max_missing=4)
    tracker.update([det(400, 300)], timestamp=0.0)
    reported = tracker.update([], timestamp=1 / FPS)
    assert reported == [], "a coasted box is a hand the detector says is not there"
    assert ids_of(tracker.tracks) == [1]


def test_hand_gone_longer_than_max_missing_gets_a_new_id():
    tracker = HandTracker(max_missing=3)
    tracker.update([det(400, 300)], timestamp=0.0)
    for i in range(1, 5):                    # four misses, one more than allowed
        tracker.update([], timestamp=i / FPS)
    assert tracker.tracks == []
    fresh = only(tracker.update([det(400, 300)], timestamp=5 / FPS))
    assert fresh.track_id == 2
    assert fresh.age == 1
    assert len(fresh.trajectory) == 1, "a new id must not inherit the old path"


def test_exactly_max_missing_frames_still_survives():
    tracker = HandTracker(max_missing=3)
    tracker.update([det(400, 300)], timestamp=0.0)
    for i in range(1, 4):                    # exactly three misses
        tracker.update([], timestamp=i / FPS)
    assert only(tracker.update([det(400, 300)], timestamp=4 / FPS)).track_id == 1


def test_max_missing_zero_drops_immediately():
    tracker = HandTracker(max_missing=0)
    tracker.update([det(400, 300)], timestamp=0.0)
    tracker.update([], timestamp=1 / FPS)
    assert tracker.tracks == []


def test_one_hand_dropping_out_does_not_renumber_the_other():
    """The whole point of coasting: a miss on hand A must not disturb hand B."""
    tracker = HandTracker()
    a, b = det(200, 300), det(900, 300)
    tracks = tracker.update([a, b], timestamp=0.0)
    id_a = track_holding(tracks, a).track_id
    id_b = track_holding(tracks, b).track_id
    for i in range(1, 4):
        tracks = tracker.update([det(900, 300)], timestamp=i / FPS)
        assert ids_of(tracks) == [id_b]
    tracks = tracker.update([det(200, 300), det(900, 300)], timestamp=4 / FPS)
    assert sorted(ids_of(tracks)) == sorted([id_a, id_b])


# --------------------------------------------------------------------------------
# velocity: pixels per SECOND
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("fps", [15.0, 30.0, 60.0])
def test_velocity_is_pixels_per_second_at_any_frame_rate(fps):
    """The same physical motion must report the same velocity at 15, 30 and 60 fps.

    300 px/s to the right. A per-frame velocity would read 20, 10 and 5 here and every
    downstream threshold would silently mean something different on every machine.
    """
    tracker = HandTracker()
    track = None
    for i in range(12):
        t = i / fps
        track = only(tracker.update([det(200 + 300 * t, 300)], timestamp=t))
    vx, vy = track.velocity
    assert vx == pytest.approx(300.0, abs=1.0)
    assert vy == pytest.approx(0.0, abs=1.0)
    assert track.speed == pytest.approx(300.0, abs=1.0)


@pytest.mark.parametrize("fps", [15.0, 60.0])
def test_velocity_sign_convention_is_image_coordinates(fps):
    """+x is right and +y is DOWN, because that is how the frame array is indexed.

    The tolerance is 8 px/s rather than 1 because detection boxes are integers: the
    box centre lands on a half-pixel grid, and at 60 fps a 0.5 px quantum spread over
    a 67 ms window is worth ~7 px/s. That floor is a property of the box contract, not
    of the fit.
    """
    tracker = HandTracker()
    track = None
    for i in range(10):
        t = i / fps
        track = only(tracker.update([det(300 + 200 * t, 200 + 150 * t)], timestamp=t))
    vx, vy = track.velocity
    assert vx == pytest.approx(200.0, abs=8.0)
    assert vy == pytest.approx(150.0, abs=8.0)


def test_velocity_is_zero_on_the_first_frame():
    tracker = HandTracker()
    track = only(tracker.update([det(400, 300)], timestamp=0.0))
    assert track.velocity == (0.0, 0.0)
    assert track.speed == 0.0


def test_velocity_uses_elapsed_time_not_frame_count():
    """Irregular frame intervals -- a dropped frame, a busy CPU -- must not scale it."""
    tracker = HandTracker()
    times = [0.0, 0.05, 0.09, 0.21, 0.25, 0.40]   # jittery, ~200 px/s hand
    track = None
    for t in times:
        track = only(tracker.update([det(300 + 200 * t, 300)], timestamp=t))
    assert track.velocity[0] == pytest.approx(200.0, abs=6.0)


def test_velocity_smoothing_beats_a_raw_frame_difference():
    """A still hand with a jittery box must not read as a moving hand.

    The detector wobbles +-3 px here. Frame-to-frame differencing turns that into
    ~175 px/s of pure noise, which sits right on top of any real gesture threshold;
    the windowed fit brings it down to a few tens.
    """
    rng = random.Random(7)
    tracker = HandTracker()
    raw_peak, smoothed_peak, previous = 0.0, 0.0, None
    for i in range(40):
        cx, cy = 400 + rng.uniform(-3, 3), 300 + rng.uniform(-3, 3)
        track = only(tracker.update([det(cx, cy)], timestamp=i / FPS))
        centre = track.detection.center
        if previous is not None:
            raw_peak = max(raw_peak, math.hypot(centre[0] - previous[0],
                                                centre[1] - previous[1]) * FPS)
        previous = centre
        if i >= 5:
            smoothed_peak = max(smoothed_peak, track.speed)
    assert raw_peak > 150.0, "test fixture no longer produces the noise it claims to"
    assert smoothed_peak < 60.0
    assert smoothed_peak < raw_peak / 2.0


def test_repeated_timestamp_does_not_produce_nan_or_infinity():
    """Two frames stamped at the same instant is a division by zero waiting to happen."""
    tracker = HandTracker()
    tracker.update([det(400, 300)], timestamp=1.0)
    track = only(tracker.update([det(430, 300)], timestamp=1.0))
    assert all(math.isfinite(v) for v in track.velocity)


def test_backwards_timestamp_does_not_invert_the_velocity():
    tracker = HandTracker()
    for i in range(5):
        tracker.update([det(300 + 10 * i, 300)], timestamp=i / FPS)
    track = only(tracker.update([det(360, 300)], timestamp=0.0))   # clock jumped back
    assert all(math.isfinite(v) for v in track.velocity)
    assert track.velocity[0] >= 0.0, "a clock reset must not report the hand reversing"


def test_nan_timestamp_is_rejected():
    tracker = HandTracker()
    with pytest.raises(ValueError):
        tracker.update([det(400, 300)], timestamp=float("nan"))


# --------------------------------------------------------------------------------
# trajectory
# --------------------------------------------------------------------------------

def test_trajectory_uses_the_wrist_when_a_pose_is_available():
    """The wrist, not the box centre -- it is the stablest landmark on the hand."""
    tracker = HandTracker()
    track = only(tracker.update([det(400, 300)], [pose_at(390, 345)], timestamp=0.0))
    assert tuple(track.trajectory[-1]) == (390.0, 345.0)


def test_trajectory_falls_back_to_the_box_centre_without_a_pose():
    tracker = HandTracker()
    detection = det(400, 300)
    track = only(tracker.update([detection], timestamp=0.0))
    assert tuple(track.trajectory[-1]) == tuple(detection.center)


def test_trajectory_handles_a_none_hole_in_the_pose_list():
    tracker = HandTracker()
    a, b = det(200, 300), det(900, 300)
    tracks = tracker.update([a, b], [pose_at(195, 340), None], timestamp=0.0)
    assert tuple(track_holding(tracks, a).trajectory[-1]) == (195.0, 340.0)
    assert tuple(track_holding(tracks, b).trajectory[-1]) == tuple(b.center)


def test_trajectory_grows_one_point_per_seen_frame():
    tracker = HandTracker()
    for i in range(10):
        track = only(tracker.update([det(400 + 5 * i, 300)], timestamp=i / FPS))
    assert len(track.trajectory) == 10


def test_trajectory_does_not_grow_while_the_hand_is_missing():
    """No observation, no point -- otherwise the path length counts invented motion."""
    tracker = HandTracker(max_missing=8)
    tracker.update([det(400, 300)], timestamp=0.0)
    for i in range(1, 5):
        tracker.update([], timestamp=i / FPS)
    track = only(tracker.tracks)
    assert len(track.trajectory) == 1


def test_trajectory_respects_its_maxlen():
    tracker = HandTracker(trajectory_len=5)
    for i in range(20):
        track = only(tracker.update([det(400 + 2 * i, 300)], timestamp=i / FPS))
    assert len(track.trajectory) == 5
    assert track.trajectory.maxlen == 5


def test_losing_the_pose_does_not_spike_the_velocity():
    """The anchor trap, and the reason _anchor() reports its kind.

    The wrist sits ~40 px below the box centre. If the pose drops out and the tracker
    differences a wrist against a centre, that 40 px reads as 1200 px/s of motion on a
    hand that never moved. The trajectory does step (it stores whichever anchor it
    had, as documented); the velocity must not.
    """
    tracker = HandTracker()
    speeds = []
    for i in range(8):
        pose = [pose_at(400, 340)] if i < 4 else None
        track = only(tracker.update([det(400, 300)], pose, timestamp=i / FPS))
        speeds.append(track.speed)
    assert max(speeds) < 50.0, f"anchor switch leaked into velocity: {speeds}"
    points = [tuple(p) for p in track.trajectory]
    assert points[0] == (400.0, 340.0) and points[-1] == (400.0, 300.0)


# --------------------------------------------------------------------------------
# trajectory_length / trajectory_direction
# --------------------------------------------------------------------------------

def test_trajectory_length_is_path_not_displacement():
    tracker = HandTracker()
    for i, cx in enumerate((400, 450, 500, 450, 400)):      # out 100 px and back
        tracker.update([det(cx, 300)], timestamp=i / FPS)
    track = tracker.tracks[0]
    assert trajectory_length(track) == pytest.approx(200.0)
    assert tuple(track.trajectory[0]) == tuple(track.trajectory[-1])


def test_trajectory_length_of_a_new_track_is_zero():
    tracker = HandTracker()
    track = only(tracker.update([det(400, 300)], timestamp=0.0))
    assert trajectory_length(track) == 0.0


def test_trajectory_length_counts_diagonals_euclidean():
    tracker = HandTracker()
    tracker.update([det(400, 300)], timestamp=0.0)
    tracker.update([det(430, 340)], timestamp=1 / FPS)      # 3-4-5 triangle
    assert trajectory_length(tracker.tracks[0]) == pytest.approx(50.0)


@pytest.mark.parametrize("dx, dy, expected", [
    (200, 0, "right"),
    (-200, 0, "left"),
    (0, -200, "up"),        # y grows downward, so negative dy is UP
    (0, 200, "down"),
    (5, 5, "static"),
])
def test_trajectory_direction(dx, dy, expected):
    tracker = HandTracker()
    for i in range(5):
        tracker.update([det(600 + dx * i / 4.0, 360 + dy * i / 4.0)], timestamp=i / FPS)
    assert trajectory_direction(tracker.tracks[0]) == expected


def test_trajectory_direction_of_a_stationary_hand_is_static():
    """Without the min_distance floor this is a coin toss driven by detector noise."""
    rng = random.Random(3)
    tracker = HandTracker()
    for i in range(30):
        tracker.update([det(400 + rng.uniform(-3, 3), 300 + rng.uniform(-3, 3))],
                       timestamp=i / FPS)
    assert trajectory_direction(tracker.tracks[0]) == "static"


def test_trajectory_direction_window_sees_only_recent_motion():
    """A hand that went right and then left: the whole path says one thing, the last
    few frames say another. Both answers are correct for their question."""
    tracker = HandTracker()
    xs = [400 + 30 * i for i in range(6)] + [550 - 30 * i for i in range(1, 6)]
    for i, cx in enumerate(xs):
        tracker.update([det(cx, 300)], timestamp=i / FPS)
    track = tracker.tracks[0]
    assert trajectory_direction(track, window=5) == "left"
    assert trajectory_direction(track) == "static"      # net travel is ~10 px


def test_trajectory_helpers_accept_a_bare_point_list():
    assert trajectory_length([(0, 0), (3, 4)]) == pytest.approx(5.0)
    assert trajectory_direction([(0, 0), (0, -100)]) == "up"
    assert trajectory_direction([]) == "static"
    assert trajectory_direction([(0, 0)]) == "static"


# --------------------------------------------------------------------------------
# coordinate spaces: full-frame vs ROI
# --------------------------------------------------------------------------------

def test_mask_is_carried_through_untouched_with_its_origin():
    """The tracker must never reinterpret a mask; ROI space stays ROI space."""
    tracker = HandTracker()
    roi = np.zeros((60, 80), np.uint8)
    roi[10:50, 20:60] = 255
    mask = HandMask(mask=roi, origin=(640, 400))
    track = only(tracker.update([det(680, 430)], None, [mask], timestamp=0.0))
    assert track.mask is mask
    assert track.mask.origin == (640, 400)
    assert np.array_equal(track.mask.mask, roi)


def test_trajectory_stays_in_frame_space_when_the_mask_is_roi_local():
    """The classic break: an ROI-local coordinate leaking into a frame-space field.

    The mask here starts at (640, 400) and its hand pixels are at ROI-local (20..60,
    10..50). If any of that leaked into the trajectory the point would land near the
    top-left of the frame instead of on the hand.
    """
    tracker = HandTracker()
    roi = np.zeros((60, 80), np.uint8)
    roi[10:50, 20:60] = 255
    mask = HandMask(mask=roi, origin=(640, 400))
    detection = det(680, 430)
    track = only(tracker.update([detection], [pose_at(675, 445)], [mask], timestamp=0.0))

    x, y = track.trajectory[-1]
    assert (x, y) == (675.0, 445.0)                       # the full-frame wrist
    x1, y1, x2, y2 = detection.box
    assert x1 <= x <= x2 and y1 <= y <= y2
    full = mask.to_frame(FRAME_SHAPE)
    ys, xs = np.nonzero(full)
    assert xs.min() == 660 and ys.min() == 410            # origin + ROI offset


def test_velocity_is_measured_in_frame_space_not_roi_space():
    """Two frames whose ROI origin moves but whose ROI-local hand does not.

    A tracker that measured motion inside the ROI would call this a stationary hand.
    """
    tracker = HandTracker()
    roi = np.zeros((40, 40), np.uint8)
    roi[10:30, 10:30] = 255
    for i in range(6):
        origin = (600 + 10 * i, 300)
        detection = det(620 + 10 * i, 320, w=40, h=40)
        track = only(tracker.update([detection], None,
                                    [HandMask(mask=roi, origin=origin)], timestamp=i / FPS))
    assert track.velocity[0] == pytest.approx(300.0, abs=1.0)


# --------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------

def test_pose_list_length_mismatch_raises():
    """Silently mispairing hand 0's box with hand 1's landmarks is invisible in a demo
    and fatal in a gesture, so it is an error, not a shrug."""
    tracker = HandTracker()
    with pytest.raises(ValueError, match="parallel"):
        tracker.update([det(200, 300), det(900, 300)], [pose_at(200, 340)], timestamp=0.0)


def test_mask_list_length_mismatch_raises():
    tracker = HandTracker()
    mask = HandMask(mask=np.zeros((4, 4), np.uint8), origin=(0, 0))
    with pytest.raises(ValueError, match="parallel"):
        tracker.update([det(200, 300)], None, [mask, mask], timestamp=0.0)


@pytest.mark.parametrize("kwargs", [
    {"max_missing": -1},
    {"iou_threshold": 0.0},
    {"iou_threshold": 1.5},
    {"velocity_window": 1},
    {"trajectory_len": 0},
])
def test_bad_constructor_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HandTracker(**kwargs)


def test_timestamp_defaults_to_a_real_clock():
    """Omitting the timestamp must still produce px/second, not px/frame."""
    tracker = HandTracker()
    for cx in (400, 430, 460, 490):
        tracker.update([det(cx, 300)])
        time.sleep(0.01)
    track = tracker.tracks[0]
    assert track.speed > 100.0          # ~30 px per ~10 ms is fast in px/second
    assert math.isfinite(track.speed)


def test_handedness_of_unknown_track_is_unknown():
    assert HandTracker().handedness_of(99) == "unknown"


# --------------------------------------------------------------------------------
# real data: EgoHands ground truth
# --------------------------------------------------------------------------------

SLOT_HANDEDNESS = ("Left", "Right", "Left", "Right")   # own L/R, other L/R


def _frame_numbers(video):
    labelled = video.loc["labelled_frames"][0]
    return [int(labelled[i][0][0][0]) for i in range(len(labelled))]


def _consecutive_runs(numbers, max_gap):
    """Index runs of labelled frames close enough in time to be a real video clip.

    EgoHands labels 100 frames scattered across a 90 s video, so most neighbouring
    annotations are seconds apart and no tracker should be judged on them. A few
    hundred pairs, though, are 1-2 frames apart -- genuine consecutive video, with
    ground-truth identity attached. Those are the ones worth running a tracker on.
    """
    runs, current = [], [0]
    for i in range(1, len(numbers)):
        if numbers[i] - numbers[i - 1] <= max_gap:
            current.append(i)
        else:
            if len(current) > 1:
                runs.append(current)
            current = [i]
    if len(current) > 1:
        runs.append(current)
    return runs


def _gt_detections(video, index, with_handedness):
    """Ground-truth boxes for one labelled frame, as HandDetections plus their slots."""
    from get_bounding_boxes import get_bounding_boxes
    boxes = get_bounding_boxes(video, index)
    dets, slots = [], []
    for slot in range(4):
        x, y, w, h = boxes[slot]
        if w <= 0 or h <= 0:
            continue
        dets.append(HandDetection(
            box=(int(x), int(y), int(x + w), int(y + h)), score=1.0,
            handedness=SLOT_HANDEDNESS[slot] if with_handedness else "unknown"))
        slots.append(slot)
    return dets, slots


def _score_dataset(videos, max_gap, with_handedness, seed=0, **tracker_kwargs):
    """Run the tracker over every consecutive-frame run in the dataset.

    Returns (identity kept, transitions, id switches). "Kept" counts frame-to-frame
    transitions where a ground-truth hand that was present in both frames kept its
    track id; a "switch" is the dangerous failure, one id landing on two different
    ground-truth hands. Detections are shuffled every frame so nothing can be scored
    on list order.
    """
    rng = random.Random(seed)
    kept = total = switches = 0
    for row in range(len(videos)):
        video = videos.iloc[row]
        numbers = _frame_numbers(video)
        for run in _consecutive_runs(numbers, max_gap):
            tracker = HandTracker(**tracker_kwargs)
            previous, id_to_slot = {}, {}
            for index in run:
                dets, slots = _gt_detections(video, index, with_handedness)
                order = list(range(len(dets)))
                rng.shuffle(order)
                dets = [dets[i] for i in order]
                slots = [slots[i] for i in order]
                tracks = tracker.update(dets, timestamp=numbers[index] / FPS)
                current = {}
                for d, slot in zip(dets, slots):
                    track = track_holding(tracks, d)
                    if track is not None:
                        current[slot] = track.track_id
                for slot, tid in current.items():
                    if slot in previous:
                        total += 1
                        kept += previous[slot] == tid
                    if id_to_slot.get(tid, slot) != slot:
                        switches += 1
                    id_to_slot[tid] = slot
                previous = current
    return kept, total, switches


@needs_dataset
def test_real_egohands_consecutive_frames_keep_identity(videos):
    """The honest quality number, on real hands with real ground-truth identity.

    628 hand transitions across 48 videos, taken only where two labelled frames are
    adjacent in the source video (33 ms apart). Measured: 99.5% of hands keep their
    id and ZERO ids move to a different hand -- the residual failures are hands whose
    ground-truth overlap between the two frames is below the association threshold,
    so they open a new id rather than steal one.
    """
    kept, total, switches = _score_dataset(videos, max_gap=1, with_handedness=True)
    assert total > 500, f"expected a few hundred real transitions, got {total}"
    assert kept / total >= 0.98, f"identity kept on only {kept}/{total}"
    assert switches == 0, f"{switches} ids jumped to a different hand"


@needs_dataset
def test_real_egohands_two_frame_gap_still_holds(videos):
    """Same measurement with up to 2 frames (67 ms) between annotations: 98.7% kept."""
    kept, total, switches = _score_dataset(videos, max_gap=2, with_handedness=True)
    assert total > 1000
    assert kept / total >= 0.97
    assert switches == 0


@needs_dataset
def test_real_egohands_handedness_prevents_id_switches(videos):
    """Handedness earns its place on real data, at the frame gaps where overlap fails.

    Stretched to a 5-frame gap (167 ms, where ground-truth overlap has fallen to 0.71
    on average) the tracker starts confusing hands. Knowing Left from Right cuts the
    id switches by two thirds on exactly the same input.
    """
    _, _, blind = _score_dataset(videos, max_gap=5, with_handedness=False)
    _, _, informed = _score_dataset(videos, max_gap=5, with_handedness=True)
    assert blind > 0, "the control case no longer switches ids; test is stale"
    assert informed < blind
    assert informed <= 2


@needs_dataset
def test_real_frames_are_the_coordinate_space_the_tracker_reports_in(videos):
    """Tie the numbers to actual pixels: boxes, trajectory and frame must agree.

    Every coordinate in this pipeline is full-frame, and the frame is a real
    1280x720 JPEG. This reads one, tracks the hands annotated in it, and checks the
    trajectory lands inside the image and inside the hand's own box.
    """
    import cv2
    from get_frame_path import get_frame_path

    video = videos.iloc[0]
    numbers = _frame_numbers(video)
    runs = _consecutive_runs(numbers, 2)
    assert runs, "no consecutive labelled frames in the first video"
    run = runs[0]

    frame = cv2.imread(get_frame_path(video, run[0]))
    assert frame is not None and frame.shape == FRAME_SHAPE

    tracker = HandTracker()
    height, width = frame.shape[:2]
    for index in run:
        dets, _ = _gt_detections(video, index, with_handedness=True)
        assert dets, "picked a frame with no annotated hands"
        for d in dets:
            x1, y1, x2, y2 = d.box
            assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        tracks = tracker.update(dets, timestamp=numbers[index] / FPS)
        for track in tracks:
            x, y = track.trajectory[-1]
            assert 0 <= x < width and 0 <= y < height
            bx1, by1, bx2, by2 = track.detection.box
            assert bx1 <= x <= bx2 and by1 <= y <= by2


@needs_dataset
def test_real_segmentation_masks_survive_tracking_in_roi_space(videos):
    """A real hand silhouette, cropped to its box, tracked, and put back.

    This is the ROI/frame boundary on real data: the mask handed in is ROI-local, the
    detection box is full-frame, and after a round trip through the tracker the mask
    must still paint the hand where the hand actually is.
    """
    from get_bounding_boxes import get_bounding_boxes
    from get_segmentation_mask import get_segmentation_mask

    video = videos.iloc[0]
    numbers = _frame_numbers(video)
    index = _consecutive_runs(numbers, 2)[0][0]
    boxes = get_bounding_boxes(video, index)
    slot = next(s for s in range(4) if boxes[s][2] > 0)
    hand_type = ("my_left", "my_right", "your_left", "your_right")[slot]

    full_mask = get_segmentation_mask(video, index, hand_type)[:, :, 0]
    x, y, w, h = (int(v) for v in boxes[slot])
    roi = full_mask[y:y + h, x:x + w].copy()
    assert roi.any(), "the cropped ROI contains none of the hand"

    detection = HandDetection(box=(x, y, x + w, y + h), score=1.0,
                              handedness=SLOT_HANDEDNESS[slot])
    tracker = HandTracker()
    track = only(tracker.update([detection], None,
                                [HandMask(mask=roi, origin=(x, y))], timestamp=0.0))

    assert track.mask.pixel_count == int((roi > 0).sum())
    restored = track.mask.to_frame(FRAME_SHAPE)
    ys, xs = np.nonzero(restored)
    assert x <= xs.min() and xs.max() <= x + w
    assert y <= ys.min() and ys.max() <= y + h
    # the same pixels the dataset painted, back in frame space
    assert np.array_equal(restored[y:y + h, x:x + w] > 0, roi > 0)
    # and the trajectory point is the full-frame box centre, not an ROI-local one
    assert tuple(track.trajectory[-1]) == tuple(detection.center)


# --------------------------------------------------------------------------------
# real time
# --------------------------------------------------------------------------------

def test_tracking_cost_per_hand_per_frame(capsys):
    """Budget is 5 ms per hand per frame; this stage should not be visible at all.

    Two hands with poses, 2000 frames, measured end to end through update().
    """
    frames = []
    for i in range(2000):
        dets, poses = [], []
        for hand in range(2):
            cx = 300 + 120 * math.sin(i / 12.0) + hand * 400
            dets.append(det(cx, 300 + 40 * hand, handedness=("Left", "Right")[hand]))
            poses.append(pose_at(cx, 340 + 40 * hand))
        frames.append((dets, poses, i / FPS))

    tracker = HandTracker()
    for dets, poses, t in frames[:100]:          # warm up the interpreter
        tracker.update(dets, poses, timestamp=t)

    tracker = HandTracker()
    start = time.perf_counter()
    for dets, poses, t in frames:
        tracker.update(dets, poses, timestamp=t)
    elapsed = time.perf_counter() - start

    per_hand = elapsed / len(frames) / 2 * 1000.0
    with capsys.disabled():
        print(f"\n    tracker: {per_hand * 1000:.1f} us per hand per frame "
              f"({elapsed / len(frames) * 1000:.4f} ms per 2-hand frame)")
    assert per_hand < 1.0, f"{per_hand:.3f} ms/hand/frame is too slow for 30 fps"


def test_cost_does_not_explode_with_hand_count():
    """Association is O(tracks x detections); four hands must not cost sixteen."""
    def cost(n_hands):
        frames = [[det(200 + 200 * h + (i % 20) * 3, 300 + 30 * h) for h in range(n_hands)]
                  for i in range(600)]
        tracker = HandTracker()
        start = time.perf_counter()
        for i, dets in enumerate(frames):
            tracker.update(dets, timestamp=i / FPS)
        return (time.perf_counter() - start) / len(frames)

    cost(2)                                   # warm
    assert cost(4) < cost(2) * 8
