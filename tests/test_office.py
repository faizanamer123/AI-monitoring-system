"""
Office monitor logic: posture rules and the event timeline, on synthetic skeletons.

No model and no video -- these check the reasoning, so they run in milliseconds.

Run:  python -m pytest tests/test_office.py -v
"""

import numpy as np
import pytest

from office.events import EventLog, summarise
from office.perception import (L_HIP, L_KNEE, L_SHOULDER, R_HIP, R_KNEE, R_SHOULDER,
                               Person, drop_nested)
from office.posture import (SITTING, STANDING, UNKNOWN, WALKING, PostureTracker,
                            classify_frame)


def skeleton(x=100.0, y=100.0, torso=50.0, knee_drop=None, box_height=None, scale=1.0):
    """A person whose shoulders are at (x, y). knee_drop is in torso lengths; None hides
    the knees. Everything multiplies by `scale`, to check the rules are scale-free."""
    torso *= scale
    kp = np.zeros((17, 3))
    kp[[L_SHOULDER, R_SHOULDER]] = [[x - 10 * scale, y, 0.9], [x + 10 * scale, y, 0.9]]
    kp[[L_HIP, R_HIP]] = [[x - 8 * scale, y + torso, 0.9], [x + 8 * scale, y + torso, 0.9]]
    if knee_drop is not None:
        knee_y = y + torso + knee_drop * torso
        kp[[L_KNEE, R_KNEE]] = [[x - 8 * scale, knee_y, 0.9], [x + 8 * scale, knee_y, 0.9]]
    height = (box_height or 3.2) * torso
    box = (int(x - 20 * scale), int(y - 0.6 * torso), int(x + 20 * scale), int(y - 0.6 * torso + height))
    return Person(track_id=1, box=box, score=0.9, keypoints=kp)


# --- single-frame posture -------------------------------------------------------------

def test_knee_well_below_hip_is_standing():
    assert classify_frame(skeleton(knee_drop=0.9)) == STANDING


def test_knee_level_with_hip_is_sitting():
    assert classify_frame(skeleton(knee_drop=0.1)) == SITTING


def test_ambiguous_knee_says_unknown_rather_than_guessing():
    assert classify_frame(skeleton(knee_drop=0.5)) == UNKNOWN


@pytest.mark.parametrize("scale", [0.3, 1.0, 4.0])
def test_rules_are_scale_free(scale):
    """Near the camera and across the room must be judged the same."""
    assert classify_frame(skeleton(knee_drop=0.9, scale=scale)) == STANDING
    assert classify_frame(skeleton(knee_drop=0.1, scale=scale)) == SITTING


def test_hidden_knees_tall_box_is_standing():
    assert classify_frame(skeleton(knee_drop=None, box_height=3.3)) == STANDING


def test_hidden_knees_short_box_is_unknown_not_sitting():
    """Head-to-waist is all you see of someone seated OR standing behind a desk."""
    assert classify_frame(skeleton(knee_drop=None, box_height=2.0)) == UNKNOWN


def test_no_torso_is_unknown():
    person = skeleton(knee_drop=0.9)
    person.keypoints[[L_HIP, R_HIP], 2] = 0.1     # hips not seen
    assert classify_frame(person) == UNKNOWN


def test_low_confidence_knees_are_ignored():
    """A knee guessed at under a desk must not decide the posture."""
    person = skeleton(knee_drop=0.9, box_height=2.0)
    person.keypoints[[L_KNEE, R_KNEE], 2] = 0.1
    assert classify_frame(person) == UNKNOWN      # not STANDING from the unseen knees


# --- posture over time ----------------------------------------------------------------

def test_moving_hips_are_walking():
    tracker = PostureTracker()
    label = None
    for i in range(20):                           # 2 torso lengths per second
        label = tracker.update(skeleton(x=100 + i * 5, knee_drop=0.9), t=i * 0.05)
    assert label == WALKING


def test_jitter_on_a_seated_person_is_not_walking():
    tracker = PostureTracker()
    rng = np.random.default_rng(0)
    label = None
    for i in range(40):
        label = tracker.update(skeleton(x=100 + rng.normal(0, 2), knee_drop=0.1), t=i * 0.05)
    assert label == SITTING


def test_still_person_with_hidden_knees_becomes_sitting():
    """The ceiling-camera desk view: torso only, not moving."""
    tracker = PostureTracker()
    labels = [tracker.update(skeleton(knee_drop=None, box_height=2.0), t=i * 0.1)
              for i in range(70)]
    assert labels[20] == UNKNOWN                  # 2 s: too soon to say
    assert labels[-1] == SITTING                  # 6.9 s still


def test_stillness_does_not_override_visible_standing():
    tracker = PostureTracker()
    label = None
    for i in range(70):
        label = tracker.update(skeleton(knee_drop=0.9), t=i * 0.1)
    assert label == STANDING


def test_inferred_sitting_is_backdated_to_when_they_stopped():
    """Walk for 2 s, then stop with hidden knees: the sit-down is at ~2 s, not ~7 s."""
    tracker, log = PostureTracker(), EventLog("cam")
    for i in range(100):
        t = i * 0.1
        x = 100 + min(i, 20) * 8                   # moving until 2 s, then still
        person = skeleton(x=x, knee_drop=None, box_height=2.0)
        label = tracker.update(person, t)
        log.update(t, {1: label}, {1: tracker.onset(1)})
    log.close()
    sit = next(e for e in log.events if e.label == SITTING)
    # up to ~1 s late (the speed window), well inside the +-2 s scoring tolerance
    assert 2.0 <= sit.start <= 3.0


def test_moving_resets_the_stillness_clock():
    tracker = PostureTracker()
    for i in range(40):                           # 4 s still
        tracker.update(skeleton(knee_drop=None, box_height=2.0), t=i * 0.1)
    for i in range(40, 50):                       # 1 s moving
        tracker.update(skeleton(x=100 + (i - 39) * 8, knee_drop=None, box_height=2.0), t=i * 0.1)
    label = tracker.update(skeleton(x=180, knee_drop=None, box_height=2.0), t=5.5)
    assert label == UNKNOWN                       # 9 s since the start, but not 5 s still


def test_a_tied_vote_goes_to_the_most_recent_label():
    """Deterministic regardless of PYTHONHASHSEED -- a tie used to follow set order."""
    tracker = PostureTracker()
    tracker.update(skeleton(knee_drop=0.1), t=0.0)
    assert tracker.update(skeleton(knee_drop=0.9), t=0.1) == STANDING
    tracker = PostureTracker()
    tracker.update(skeleton(knee_drop=0.9), t=0.0)
    assert tracker.update(skeleton(knee_drop=0.1), t=0.1) == SITTING


def test_one_bad_frame_does_not_flip_the_label():
    tracker = PostureTracker()
    for i in range(10):
        tracker.update(skeleton(knee_drop=0.1), t=i * 0.1)
    assert tracker.update(skeleton(knee_drop=0.9), t=1.0) == SITTING


# --- the event timeline ---------------------------------------------------------------

def run(log, timeline):
    """timeline: [(t, {id: label}), ...]"""
    for t, labels in timeline:
        log.update(t, labels)
    return log


def test_timeline_records_appear_segments_and_leave():
    log = EventLog("cam", lost_after=1.0)
    run(log, [(t / 10, {1: SITTING}) for t in range(0, 50)]          # sit 0-5 s
             + [(t / 10, {1: STANDING}) for t in range(50, 80)]      # stand 5-8 s
             + [(t / 10, {}) for t in range(80, 110)])               # gone
    kinds = [(e.kind, e.label) for e in log.events]
    assert kinds == [("appeared", ""), ("posture", SITTING), ("posture", STANDING), ("left", "")]
    sit, stand, left = log.events[1], log.events[2], log.events[3]
    assert (sit.start, sit.end) == (0.0, 5.0)
    assert stand.start == 5.0 and stand.end == pytest.approx(7.9)
    assert left.start == pytest.approx(7.9)       # timed at the last sighting


def test_unknown_and_flicker_are_not_logged_as_activities():
    log = EventLog("cam", min_segment=1.0)
    run(log, [(0.0, {1: UNKNOWN}), (3.0, {1: SITTING}), (3.2, {1: STANDING}), (6.0, {1: STANDING})])
    log.close()
    labels = [e.label for e in log.events if e.kind == "posture"]
    assert labels == [STANDING]                   # unknown dropped, 0.2 s of sitting dropped


def test_end_of_video_closes_segments_without_a_left_event():
    log = EventLog("cam")
    run(log, [(t, {1: SITTING}) for t in range(5)])
    closed = log.close()
    assert [e.kind for e in closed] == ["posture"]
    assert not log.tracks


def test_people_are_kept_apart():
    log = EventLog("cam")
    run(log, [(t, {1: SITTING, 2: WALKING}) for t in range(4)])
    log.close()
    totals = summarise(log.events)
    assert totals[("cam", 1)] == {SITTING: 3.0}
    assert totals[("cam", 2)] == {WALKING: 3.0}


def test_short_unknown_gap_does_not_split_a_segment():
    log = EventLog("cam", hold_unknown=3.0)
    run(log, [(0.0, {1: SITTING}), (5.0, {1: SITTING}), (6.0, {1: UNKNOWN}),
              (7.0, {1: UNKNOWN}), (8.0, {1: SITTING}), (12.0, {1: SITTING})])
    log.close()
    segments = [(e.label, e.start, e.end) for e in log.events if e.kind == "posture"]
    assert segments == [(SITTING, 0.0, 12.0)]


def test_long_unknown_ends_the_segment_where_it_was_last_seen():
    log = EventLog("cam", hold_unknown=3.0)
    run(log, [(0.0, {1: SITTING}), (5.0, {1: SITTING})]
             + [(t, {1: UNKNOWN}) for t in (6.0, 7.0, 8.0, 9.0, 10.0)])
    log.close()
    segments = [(e.label, e.start, e.end) for e in log.events if e.kind == "posture"]
    assert segments == [(SITTING, 0.0, 5.0)]      # not 0-9: nothing was seen after 5 s


def test_brief_dropout_does_not_split_a_person():
    """A missed detection or two must not become left + appeared."""
    log = EventLog("cam", lost_after=2.0)
    run(log, [(0.0, {1: SITTING}), (1.0, {}), (2.0, {1: SITTING}), (3.0, {1: SITTING})])
    assert [e.kind for e in log.events] == ["appeared"]


# --- duplicate boxes ------------------------------------------------------------------

def box_person(track_id, box, score):
    return Person(track_id=track_id, box=box, score=score, keypoints=np.zeros((17, 3)))


def test_upper_body_box_inside_full_body_box_is_dropped():
    full = box_person(1, (100, 100, 200, 400), 0.8)
    upper = box_person(4, (105, 100, 195, 250), 0.6)
    assert [p.track_id for p in drop_nested([upper, full])] == [1]


def test_two_people_side_by_side_are_both_kept():
    left = box_person(1, (100, 100, 200, 400), 0.8)
    right = box_person(2, (180, 100, 280, 400), 0.7)   # 20 px overlap
    assert sorted(p.track_id for p in drop_nested([left, right])) == [1, 2]
