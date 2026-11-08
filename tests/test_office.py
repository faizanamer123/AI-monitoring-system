"""
Office monitor logic: posture rules and the event timeline, on synthetic skeletons.

No model and no video -- these check the reasoning, so they run in milliseconds.

Run:  python -m pytest tests/test_office.py -v
"""

import csv
import sqlite3

import numpy as np
import pytest

from office.activity import ON_CALL, USING_PHONE, WORKING, classify_activity
from office.cache import TrackWriter, read_tracks
from office.events import EventLog, PostureChanges, summarise
from office.objects import Thing
from office.perception import (L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_KNEE,
                               R_SHOULDER, R_WRIST, Person, drop_nested)
from office.posture import (SITTING, STANDING, UNKNOWN, WALKING, PostureTracker,
                            classify_frame)
from office.stitch import IdStitcher, touches_edge
from evaluate_office import intervals, merge
from office_monitor import Rules


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


# --- activity: phone and computer -----------------------------------------------------

def with_head_and_wrists(person, head=None, wrists=()):
    kp = person.keypoints
    if head is not None:
        kp[NOSE] = [*head, 0.9]
    for i, xy in zip((L_WRIST, R_WRIST), wrists):
        kp[i] = [*xy, 0.9]
    return person


def phone_at(x, y, size=10):
    return Thing("cell phone", (x - size, y - size, x + size, y + size), 0.8)


def test_phone_at_the_head_is_a_call():
    person = with_head_and_wrists(skeleton(), head=(100, 70), wrists=[(110, 75)])
    assert classify_activity(person, SITTING, [phone_at(108, 72)]) == ON_CALL


def test_phone_at_the_hand_is_using_phone():
    person = with_head_and_wrists(skeleton(), head=(100, 70), wrists=[(100, 160)])
    assert classify_activity(person, SITTING, [phone_at(102, 165)]) == USING_PHONE


def test_hand_at_the_head_without_a_phone_is_not_a_call():
    """Scratching your head is not a phone call."""
    person = with_head_and_wrists(skeleton(), head=(100, 70), wrists=[(105, 72)])
    assert classify_activity(person, SITTING, []) == UNKNOWN


def test_someone_elses_phone_does_not_count():
    person = with_head_and_wrists(skeleton(), head=(100, 70), wrists=[(100, 160)])
    assert classify_activity(person, SITTING, [phone_at(600, 165)]) == UNKNOWN


def test_seated_at_a_laptop_is_working():
    laptop = Thing("laptop", (90, 150, 150, 190), 0.9)
    assert classify_activity(skeleton(knee_drop=0.1), SITTING, [laptop]) == WORKING


def test_standing_by_a_laptop_is_not_working():
    laptop = Thing("laptop", (90, 150, 150, 190), 0.9)
    assert classify_activity(skeleton(knee_drop=0.9), STANDING, [laptop]) == UNKNOWN


def test_phone_outranks_the_computer():
    """Texting at your desk is phone time, not work time."""
    person = with_head_and_wrists(skeleton(knee_drop=0.1), head=(100, 70), wrists=[(100, 160)])
    laptop = Thing("laptop", (90, 150, 150, 190), 0.9)
    assert classify_activity(person, SITTING, [laptop, phone_at(102, 165)]) == USING_PHONE


def test_activity_runs_in_parallel_with_posture_and_owns_no_presence():
    log = EventLog("cam", kind="activity", presence=False)
    run(log, [(float(t), {1: ON_CALL}) for t in range(5)] + [(float(t), {}) for t in range(5, 10)])
    log.close()
    assert [(e.kind, e.label) for e in log.events] == [("activity", ON_CALL)]


def test_cache_round_trip_keeps_people_and_objects(tmp_path):
    writer = TrackWriter(tmp_path / "v.npz", fps=30)
    person = skeleton(knee_drop=0.1)
    writer.add(0.0, [person], [phone_at(50, 60)])
    writer.add(0.5, [], [])
    writer.close()
    frames = list(read_tracks(tmp_path / "v.npz"))
    assert [t for t, _, _ in frames] == [0.0, 0.5]
    (_, people, things), (_, no_people, no_things) = frames
    # float32 on disk, as the model emits; the synthetic skeleton is float64
    assert people[0].box == person.box and np.allclose(people[0].keypoints, person.keypoints)
    assert [(t.name, t.box) for t in things] == [("cell phone", (40, 50, 60, 70))]
    assert no_people == [] and no_things == []


# --- storage -----------------------------------------------------------------------------

def test_a_second_run_does_not_erase_the_first(tmp_path):
    from office.events import Event, EventStore

    for session in ("run1", "run2"):
        store = EventStore(tmp_path, session)
        store.write([Event("cam", 1, "posture", SITTING, 0.0, 5.0)])
        store.close()
    db = sqlite3.connect(tmp_path / "events.db")
    assert [r[0] for r in db.execute("SELECT session FROM events")] == ["run1", "run2"]
    db.close()
    with open(tmp_path / "events.csv", newline="") as f:
        assert [r["session"] for r in csv.DictReader(f)] == ["run1", "run2"]


# --- tuning: confirmation, stitching, edges, steadiness, camera agreement --------------

def test_short_lived_id_writes_nothing():
    log = EventLog("cam", confirm_s=3.0, lost_after=1.0)
    run(log, [(t / 10, {1: SITTING}) for t in range(15)] + [(t / 10, {}) for t in range(15, 40)])
    log.close()
    assert log.events == []


def test_confirmed_id_is_backdated_to_its_first_sighting():
    log = EventLog("cam", confirm_s=3.0)
    run(log, [(t / 10, {1: SITTING}) for t in range(50)])
    assert (log.events[0].kind, log.events[0].start) == ("appeared", 0.0)


def test_appearing_at_the_frame_edge_is_recorded():
    log = EventLog("cam", lost_after=1.0)
    log.update(0.0, {1: SITTING}, edges={1: True})
    log.update(1.0, {1: SITTING}, edges={1: False})
    log.update(3.0, {})
    presence = [(e.kind, e.label) for e in log.events if e.kind != "posture"]
    assert presence == [("appeared", "edge"), ("left", "inside")]


def test_touches_edge():
    assert touches_edge((2, 50, 40, 120), (320, 240))
    assert not touches_edge((100, 50, 140, 120), (320, 240))


def test_new_id_in_the_same_place_rejoins_the_lost_one():
    stitcher = IdStitcher(gap_s=5.0)
    stitcher.update([box_person(1, (100, 100, 140, 200), 0.9)], 0.0)
    out = stitcher.update([box_person(7, (104, 102, 144, 202), 0.9)], 3.0)
    assert [p.track_id for p in out] == [1]


def test_new_id_far_away_or_much_later_is_someone_else():
    stitcher = IdStitcher(gap_s=5.0)
    stitcher.update([box_person(1, (100, 100, 140, 200), 0.9)], 0.0)
    assert stitcher.update([box_person(7, (280, 100, 320, 200), 0.9)], 3.0)[0].track_id == 7
    stitcher = IdStitcher(gap_s=5.0)
    stitcher.update([box_person(1, (100, 100, 140, 200), 0.9)], 0.0)
    assert stitcher.update([box_person(7, (100, 100, 140, 200), 0.9)], 9.0)[0].track_id == 7


def test_a_person_still_in_view_cannot_be_rejoined():
    stitcher = IdStitcher(gap_s=5.0)
    stitcher.update([box_person(1, (100, 100, 140, 200), 0.9)], 0.0)
    out = stitcher.update([box_person(1, (100, 100, 140, 200), 0.9),
                           box_person(7, (110, 100, 150, 200), 0.8)], 1.0)
    assert sorted(p.track_id for p in out) == [1, 7]


def changes_from(timeline):
    """timeline: [(from_t, to_t, label), ...] at 10 samples per second"""
    detector = PostureChanges()
    found = []
    for start, end, label in timeline:
        for tick in range(int(start * 10), int(end * 10)):
            change = detector.update(1, label, tick / 10)
            if change:
                found.append(change)
    return found


def test_a_real_stand_up_is_reported_when_it_happened():
    found = changes_from([(0, 20, SITTING), (20, 30, STANDING)])
    assert [name for name, _ in found] == ["stand up"]
    assert found[0][1] == pytest.approx(19.95)


def test_a_flicker_of_standing_is_not_a_stand_up():
    assert changes_from([(0, 20, SITTING), (20, 20.5, STANDING), (20.5, 40, SITTING)]) == []


def test_stand_up_then_walk_away_is_one_change():
    found = changes_from([(0, 20, SITTING), (20, 21.5, STANDING), (21.5, 25, WALKING)])
    assert [name for name, _ in found] == ["stand up"]


def test_a_change_across_a_long_blind_spot_is_not_timed():
    assert changes_from([(0, 10, SITTING), (10, 30, UNKNOWN), (30, 40, STANDING)]) == []


class NoStore:
    def write(self, events):
        pass


def test_sitting_inferred_from_stillness_fills_the_timeline_but_is_no_sit_down():
    rules = Rules("cam", NoStore())
    for i in range(120):
        t = i * 0.1
        moving = min(i, 20) * 8                  # walks for 2 s, then stops
        rules.step(t, [skeleton(x=100 + moving, knee_drop=None, box_height=2.0)])
    events = rules.finish()
    assert any(e.kind == "posture" and e.label == SITTING for e in events)
    assert not any(e.kind == "change" for e in events)


def test_an_observed_sit_down_reaches_the_live_timeline():
    rules = Rules("cam", NoStore())
    for i in range(100):
        t = i * 0.1
        rules.step(t, [skeleton(knee_drop=0.9 if t < 5 else 0.1)])
    changes = [e for e in rules.finish() if e.kind == "change"]
    assert [e.label for e in changes] == ["sit down"]
    assert changes[0].start == pytest.approx(5.0, abs=1.0)


def test_a_few_seconds_out_of_view_does_not_split_the_timeline():
    rules = Rules("cam", NoStore(), stitch_gap_s=5.0)
    for i in range(300):
        t = i * 0.1
        rules.step(t, [] if 10 <= t < 14 else [skeleton(knee_drop=0.1)])
    posture = [e for e in rules.finish() if e.kind == "posture"]
    assert len(posture) == 1
    assert (posture[0].start, posture[0].end) == pytest.approx((0.0, 29.9))


def test_an_unreadable_person_keeps_their_posture_for_ten_seconds():
    rules = Rules("cam", NoStore())
    for i in range(300):
        t = i * 0.1
        unreadable = skeleton(knee_drop=None, box_height=2.0)
        unreadable.keypoints[[L_HIP, R_HIP], 2] = 0.1     # no torso, so no evidence at all
        rules.step(t, [unreadable if 10 <= t < 18 else skeleton(knee_drop=0.1)])
    posture = [e for e in rules.finish() if e.kind == "posture"]
    assert len(posture) == 1


def test_enter_and_exit_at_the_edge_only():
    log = EventLog("cam", lost_after=1.0)
    log.update(0.0, {}, edges={})
    log.update(3.0, {1: STANDING}, edges={1: True})
    log.update(5.0, {1: STANDING}, edges={1: True})
    log.update(7.0, {})
    log.update(10.0, {2: STANDING}, edges={2: False})
    log.update(12.0, {2: STANDING}, edges={2: False})
    log.update(14.0, {})
    assert [e.label for e in log.events if e.kind == "change"] == ["enter", "exit"]


def test_nobody_enters_with_edges_off():
    log = EventLog("cam", lost_after=1.0)
    log.update(0.0, {})
    log.update(3.0, {1: STANDING})
    log.update(5.0, {})
    log.update(7.0, {})
    assert not [e for e in log.events if e.kind == "change"]


def test_interval_brackets_the_score_and_narrows_with_more_recordings():
    rows = {i: {"sit down": [1, 1, 1]} if i % 2 else {"sit down": [3, 1, 0]} for i in range(10)}
    (low, high), _, _ = intervals(rows, ("sit down",), samples=500)
    assert low <= 0.667 <= high
    many = {i: rows[i % 10] for i in range(200)}
    (low2, high2), _, _ = intervals(many, ("sit down",), samples=500)
    assert high2 - low2 < high - low


def test_two_cameras_must_agree_when_asked():
    dets = [("sit down", 10.0, "id00"), ("sit down", 11.0, "id02"), ("sit down", 40.0, "id01")]
    assert merge(dets, 2.0, min_cameras=2) == [("sit down", 10.0)]
    assert sorted(merge(dets, 2.0)) == [("sit down", 10.0), ("sit down", 40.0)]


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
