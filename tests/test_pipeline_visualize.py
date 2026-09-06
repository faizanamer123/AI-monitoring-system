"""
Tests for pipeline/visualize.py -- the overlay stage.

Headless by construction: this suite never opens a camera and never calls a GUI
function (there is a test that greps the module to keep it that way). Everything is
checked by reading pixels back out of the array that was drawn on, which is the only
honest way to assert that a drawing function drew.

The bug this file exists to catch is the coordinate mix-up. HandMask.mask is the one
ROI-local array in the whole pipeline; every other coordinate is full-frame. Blend it
at (0, 0) instead of at its origin and the overlay lands in the corner of the screen
while the box and the skeleton land on the hand -- so several tests below assert on
exactly where the pixels changed, not merely that they changed.

Run with:   .venv/bin/python -m pytest tests/test_pipeline_visualize.py -v
"""

import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import REPO, needs_dataset

sys.path.insert(0, str(REPO))

from pipeline.types import CONNECTIONS, FINGERTIPS, HandDetection, HandMask, HandPose, TrackedHand
from pipeline import visualize as viz


# ---------------------------------------------------------------- builders

def blank(height=240, width=320, value=0):
    return np.full((height, width, 3), value, np.uint8)


def make_pose(x=100.0, y=100.0, spread=40.0, count=21):
    """21 landmarks laid out on a ring so no two land on the same pixel."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.stack([x + spread * np.cos(angles), y + spread * np.sin(angles)], axis=1)
    return HandPose(points=points.astype(np.float32), score=0.9)


def make_mask(origin=(60, 40), size=50, fill=True):
    roi = np.zeros((size, size), np.uint8)
    if fill:
        roi[5:size - 5, 5:size - 5] = 255
    return HandMask(mask=roi, origin=origin)


def make_track(track_id=1, box=(50, 50, 150, 150), pose=True, mask=True,
               trajectory=True, gesture="point", velocity=(45.0, -12.0),
               handedness="Right", score=0.98):
    detection = HandDetection(box=box, score=score, handedness=handedness)
    return TrackedHand(
        track_id=track_id,
        detection=detection,
        pose=make_pose(100, 100) if pose else None,
        mask=make_mask() if mask else None,
        trajectory=deque([(60 + i * 2, 120 - i) for i in range(20)], maxlen=64)
        if trajectory else deque(maxlen=64),
        gesture=gesture,
        gesture_confidence=0.87,
        velocity=velocity,
        age=30,
    )


def changed_pixels(before, after):
    """Boolean (h, w) map of where the frame differs."""
    return np.any(before != after, axis=2)


# ---------------------------------------------------------------- basic contract

def test_render_preserves_shape_and_dtype():
    frame = blank()
    out = viz.render(frame, [make_track()], 30.0)
    assert out.shape == (240, 320, 3)
    assert out.dtype == np.uint8


def test_render_draws_in_place_and_returns_the_same_array():
    """Documented as in-place: a copy per frame is 2.7 MB of garbage at 1280x720."""
    frame = blank()
    out = viz.render(frame, [make_track()], 30.0)
    assert out is frame


def test_render_actually_modifies_the_frame():
    frame = blank()
    before = frame.copy()
    viz.render(frame, [make_track()], 30.0)
    assert changed_pixels(before, frame).any()


def test_draw_hand_returns_the_same_array():
    frame = blank()
    assert viz.draw_hand(frame, make_track()) is frame


def test_empty_show_draws_nothing():
    """`show` is a whitelist, so an empty one has to be a genuine no-op."""
    frame = blank()
    before = frame.copy()
    viz.render(frame, [make_track()], 30.0, show=())
    assert np.array_equal(before, frame)


def test_hud_is_drawn_for_the_layer_names_the_runner_passes():
    """run_pipeline.py builds --show from its own list of the five hand layers and
    never passes a "hud" token. If the HUD were opt-in, the runner would lose its fps
    readout and nothing here would have noticed."""
    runner_show = ("mask", "skeleton", "box", "trail", "label")
    frame = blank()
    viz.render(frame, [], 30.0, show=runner_show)   # no hands: only the HUD can draw
    assert frame.any(), "no HUD for the runner's own show list"


def test_hud_can_be_switched_off_explicitly():
    frame = blank()
    viz.render(frame, [], 30.0, show=("mask", "box", viz.NO_HUD))
    assert not frame.any()


def test_runner_show_names_are_all_understood():
    """Parsed out of run_pipeline.py rather than imported, so this cannot drag torch
    and mediapipe into a drawing test -- and skipped if the runner is not written yet."""
    runner = REPO / "run_pipeline.py"
    if not runner.is_file():
        pytest.skip("run_pipeline.py not present")
    match = re.search(r"ALL_OVERLAYS\s*=\s*\(([^)]*)\)", runner.read_text())
    if not match:
        pytest.skip("run_pipeline.py has no ALL_OVERLAYS")
    names = re.findall(r'"([^"]+)"', match.group(1))
    assert names, "could not parse ALL_OVERLAYS"
    assert set(names) <= set(viz.DEFAULT_SHOW), (
        f"runner asks for overlays visualize.py does not draw: "
        f"{sorted(set(names) - set(viz.DEFAULT_SHOW))}")

    # and every one of them must actually put pixels on the frame
    for name in names:
        frame = blank()
        viz.draw_hand(frame, make_track(), show=(name,))
        assert changed_pixels(blank(), frame).any(), f"layer {name!r} drew nothing"


def test_show_selects_layers_independently():
    box_only = blank()
    viz.render(box_only, [make_track()], 30.0, show=("box",))
    mask_only = blank()
    viz.render(mask_only, [make_track()], 30.0, show=("mask",))
    assert changed_pixels(blank(), box_only).any()
    assert changed_pixels(blank(), mask_only).any()
    assert not np.array_equal(box_only, mask_only)


# ---------------------------------------------------------------- colours

def test_same_track_id_gives_the_same_colour_across_calls():
    first = viz.color_for_track(7)
    for _ in range(50):
        assert viz.color_for_track(7) == first


def test_same_track_id_paints_the_same_pixels_across_frames():
    """The property that actually matters: hand 1 does not strobe between frames."""
    frames = []
    for _ in range(3):
        frame = blank()
        viz.draw_hand(frame, make_track(track_id=3), show=("box",))
        frames.append(frame)
    assert np.array_equal(frames[0], frames[1])
    assert np.array_equal(frames[1], frames[2])


def test_different_track_ids_get_distinct_colours():
    colours = [viz.color_for_track(i) for i in range(12)]
    assert len(set(colours)) == len(colours)


def test_colours_are_valid_bgr_triples():
    for track_id in (0, 1, 5, 99, 100000):
        colour = viz.color_for_track(track_id)
        assert len(colour) == 3
        assert all(isinstance(c, int) and 0 <= c <= 255 for c in colour)


def test_neighbouring_ids_are_visually_far_apart():
    """Golden-ratio hue stepping exists so hand 1 and hand 2 are not two greens."""
    for track_id in range(8):
        a = np.array(viz.color_for_track(track_id), float)
        b = np.array(viz.color_for_track(track_id + 1), float)
        assert np.abs(a - b).sum() > 120, f"{track_id} and {track_id + 1} too close"


def test_colour_is_deterministic_across_processes():
    """Salted hash() would make the same id a different colour in another process."""
    snippet = (
        "import sys; sys.path.insert(0, %r);"
        "from pipeline.visualize import color_for_track;"
        "print([color_for_track(i) for i in (0, 1, 2, 41)],"
        "      [color_for_track(s) for s in ('a', 'left-hand')])" % str(REPO)
    )
    outputs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                                text=True, env=env, cwd=str(REPO))
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())
    assert len(set(outputs)) == 1, outputs


def test_string_and_huge_track_ids_do_not_collapse_to_one_colour():
    """(key * 0.618) % 1.0 returns exactly 0.0 once key is around 2**63, so every
    string id would come out the same red. Ids are ints today; this keeps the door
    shut if a tracker ever starts naming them."""
    ids = ["left-hand", "right-hand", "a", "b", 0, 10 ** 15, 10 ** 15 + 1, 2 ** 62]
    colours = [viz.color_for_track(i) for i in ids]
    assert len(set(colours)) == len(ids), dict(zip(ids, colours))


def test_colour_cache_eviction_keeps_colours_stable():
    """The cache is a pure-function memo, so overflowing it must not change answers."""
    expected = viz.color_for_track(11)
    for track_id in range(viz._COLOR_CACHE_MAX + 50):
        viz.color_for_track(1000 + track_id)
    assert viz.color_for_track(11) == expected


# ---------------------------------------------------------------- the coordinate rule

def test_mask_is_composited_at_its_origin_not_at_zero_zero():
    """THE bug this module could have. ROI-local mask + full-frame everything else."""
    frame = blank(240, 320, value=30)
    before = frame.copy()
    mask = make_mask(origin=(200, 150), size=40)
    viz.draw_mask(frame, mask, (0, 255, 0))

    diff = changed_pixels(before, frame)
    rows, cols = np.nonzero(diff)
    assert diff.any(), "mask drew nothing at all"
    assert rows.min() >= 150 and rows.max() < 190, (rows.min(), rows.max())
    assert cols.min() >= 200 and cols.max() < 240, (cols.min(), cols.max())
    assert not diff[:100, :100].any(), "mask leaked into the top-left corner"


def test_mask_pixels_match_to_frame():
    """draw_mask and HandMask.to_frame must agree about which pixels are hand."""
    frame = blank(240, 320, value=30)
    before = frame.copy()
    mask = make_mask(origin=(120, 80), size=60)
    viz.draw_mask(frame, mask, (0, 255, 0))

    full = mask.to_frame(frame.shape)
    diff = changed_pixels(before, frame)
    assert diff.any()
    assert np.all(full[diff] > 0), "painted a pixel that the mask says is not hand"


def test_skeleton_is_drawn_in_full_frame_pixels():
    frame = blank(240, 320)
    pose = make_pose(x=250.0, y=60.0, spread=20.0)
    viz.draw_skeleton(frame, pose, (255, 0, 0))
    diff = changed_pixels(blank(240, 320), frame)
    rows, cols = np.nonzero(diff)
    assert 30 <= rows.mean() <= 90, rows.mean()
    assert 220 <= cols.mean() <= 280, cols.mean()


def test_inverted_box_renders_the_same_as_the_normal_one():
    """Detectors emit (x2, y2, x1, y1) for a hand leaving the frame. cv2.rectangle
    tolerates that, but the label anchor assumes corner 0 is the top-left, so an
    unsorted box silently parks the text in the wrong corner."""
    normal, inverted = blank(240, 320), blank(240, 320)
    viz.draw_hand(normal, make_track(box=(50, 60, 150, 160)), show=("box", "label"))
    viz.draw_hand(inverted, make_track(box=(150, 160, 50, 60)), show=("box", "label"))
    assert changed_pixels(blank(240, 320), normal).any()
    assert np.array_equal(normal, inverted)


def test_box_is_drawn_in_full_frame_pixels():
    frame = blank(240, 320)
    viz.draw_box(frame, HandDetection(box=(200, 40, 260, 100), score=0.9), (0, 0, 255))
    diff = changed_pixels(blank(240, 320), frame)
    rows, cols = np.nonzero(diff)
    assert rows.min() >= 38 and rows.max() <= 102
    assert cols.min() >= 198 and cols.max() <= 262


# ---------------------------------------------------------------- mask appearance

def test_mask_overlay_is_semi_transparent():
    """A solid fill hides the hand, which is the thing the user is checking."""
    frame = blank(120, 120, value=200)
    colour = (0, 255, 0)
    viz.draw_mask(frame, make_mask(origin=(20, 20), size=60), colour, alpha=0.4)

    inside = frame[45, 45]                       # well away from the contour outline
    assert not np.array_equal(inside, [200, 200, 200]), "overlay did not draw"
    assert not np.array_equal(inside, colour), "overlay is opaque, hand is hidden"
    assert 0 < inside[1] - 200 * 0.6 < 255


def test_hand_texture_survives_under_the_overlay():
    """Alpha blending must preserve contrast, not flatten the ROI to one colour."""
    rng = np.random.default_rng(0)
    frame = (rng.random((120, 120, 3)) * 255).astype(np.uint8)
    roi_before = frame[30:70, 30:70].copy()
    viz.draw_mask(frame, make_mask(origin=(20, 20), size=60), (0, 255, 0), alpha=0.4)
    roi_after = frame[30:70, 30:70]
    assert roi_after.std() > 0.5 * roi_before.std(), "overlay flattened the hand away"


def test_default_mask_alpha_is_semi_transparent():
    """The explicit-alpha test above would pass against MASK_ALPHA = 1.0, because it
    never uses the default. This one goes through draw_hand with no alpha argument,
    which is the path the runner actually takes."""
    frame = blank(200, 200, value=200)
    track = make_track(track_id=1)
    track.mask = make_mask(origin=(20, 20), size=80)
    viz.draw_hand(frame, track, show=("mask",))

    inside = frame[60, 60]
    colour = viz.color_for_track(1)
    assert not np.array_equal(inside, [200, 200, 200]), "overlay did not draw"
    assert not np.array_equal(inside, colour), "default overlay is opaque"
    assert 0.0 < viz.MASK_ALPHA < 1.0


def test_mask_alpha_zero_leaves_the_frame_alone():
    frame = blank(120, 120, value=200)
    before = frame.copy()
    viz.draw_mask(frame, make_mask(origin=(20, 20), size=60), (0, 255, 0),
                  alpha=0.0, outline=False)
    assert np.array_equal(before, frame)


# ---------------------------------------------------------------- skeleton

def test_fingertips_are_a_different_colour_from_other_joints():
    frame = blank(240, 320)
    pose = make_pose(x=160.0, y=120.0, spread=80.0)
    colour = viz.color_for_track(1)
    viz.draw_skeleton(frame, pose, colour)

    points = pose.points.astype(int)
    tip = frame[points[8][1], points[8][0]]
    knuckle = frame[points[6][1], points[6][0]]
    assert tuple(int(c) for c in tip) == viz.TIP_COLOR
    assert tuple(int(c) for c in knuckle) == colour
    assert not np.array_equal(tip, knuckle)


def test_every_bone_in_connections_is_drawn():
    """A missing pair in the loop shows up as one skeleton line silently absent."""
    frame = blank(400, 400)
    pose = make_pose(x=200.0, y=200.0, spread=150.0)
    viz.draw_skeleton(frame, pose, (255, 255, 255))
    points = pose.points.astype(int)
    for start, end in CONNECTIONS:
        midpoint = (points[start] + points[end]) // 2
        window = frame[midpoint[1] - 3:midpoint[1] + 4, midpoint[0] - 3:midpoint[0] + 4]
        assert window.any(), f"no bone drawn between {start} and {end}"


def test_all_21_joints_are_drawn():
    frame = blank(400, 400)
    pose = make_pose(x=200.0, y=200.0, spread=150.0)
    viz.draw_skeleton(frame, pose, (200, 100, 50))
    points = pose.points.astype(int)
    for index in range(21):
        assert frame[points[index][1], points[index][0]].any(), f"joint {index} missing"
    assert set(FINGERTIPS).issubset(range(21))


# ---------------------------------------------------------------- trail

def test_trail_fades_with_age(monkeypatch):
    """Older points dimmer: that is what encodes direction into a still frame.

    Thickness is pinned to 1 for this test. Measuring a trail that fades AND thickens
    at once cannot tell the two apart -- a thicker antialiased line has a brighter
    peak pixel on its own -- and the version of this test that did not pin it passed
    happily against a build with the fade removed entirely.
    """
    monkeypatch.setattr(viz, "TRAIL_MAX_THICKNESS", 1)
    frame = blank(200, 400)
    trajectory = deque([(20 + i * 5, 100) for i in range(64)], maxlen=64)
    viz.draw_trail(frame, trajectory, (255, 255, 255))

    old = int(frame[95:106, 30:60].max())
    new = int(frame[95:106, 250:280].max())
    assert old > 0, "the old end of the trail vanished entirely"
    assert new > old + 30, f"trail does not fade: old={old} new={new}"


def test_trail_thickness_grows_toward_the_present(monkeypatch):
    """And the mirror image: fade pinned flat, so this measures thickness only."""
    monkeypatch.setattr(viz, "TRAIL_MIN_FADE", 1.0)
    frame = blank(200, 400)
    trajectory = deque([(20 + i * 5, 100) for i in range(64)], maxlen=64)
    viz.draw_trail(frame, trajectory, (255, 255, 255))
    old_column = np.count_nonzero(frame[:, 40].any(axis=1))
    new_column = np.count_nonzero(frame[:, 290].any(axis=1))
    assert new_column > old_column, (old_column, new_column)


def test_trail_with_one_point_draws_a_dot():
    frame = blank(200, 200)
    viz.draw_trail(frame, deque([(100, 100)]), (0, 255, 0))
    assert frame[100, 100].any()


def test_trail_spans_the_whole_trajectory_without_gaps():
    """Banding into six polylines must not drop a segment or leave a seam.

    Asserted column by column rather than by the overall extent: the bright head dot
    sits at the newest point, so an extent check still passes when the last band was
    quietly truncated and only the dot is holding the right-hand end up.
    """
    frame = blank(200, 400)
    trajectory = deque([(20 + i * 5, 100) for i in range(64)], maxlen=64)
    viz.draw_trail(frame, trajectory, (255, 255, 255))

    lit = frame.any(axis=(0, 2))
    span = lit[25:330]
    assert span.all(), f"{np.count_nonzero(~span)} empty columns inside the trail"


# ---------------------------------------------------------------- text and HUD

def test_hud_text_has_a_dark_outline():
    """Readable on a white wall: there must be near-black pixels around the glyphs."""
    frame = blank(240, 320, value=255)
    viz.draw_hud(frame, [make_track()], 30.0)
    top = frame[:80, :]
    assert (top.max(axis=2) < 40).any(), "no dark outline drawn behind the HUD text"
    assert (top.min(axis=2) > 200).any(), "HUD blacked out everything"


def test_hud_text_readable_on_black_too():
    frame = blank(240, 320, value=0)
    viz.draw_hud(frame, [make_track()], 30.0)
    assert (frame[:80, :].max(axis=2) > 180).any(), "no bright pass on a dark frame"


def test_hud_lines_report_fps_and_hand_count():
    lines = viz.hud_lines([make_track(1), make_track(2)], fps=31.7)
    assert "31.7 fps" in lines[0]
    assert "hands: 2" in lines[0]
    assert len(lines) == 3


def test_hud_lines_report_per_hand_movement():
    track = make_track(1, gesture="point")
    line = viz.hud_lines([track], fps=30.0)[1]
    assert "#1" in line and "Right" in line
    assert "point" in line and "87%" in line
    assert "px" in line and "px/s" in line


def test_hud_extra_dict_and_sequence():
    lines = viz.hud_lines([], fps=30.0, extra={"latency": "18 ms"})
    assert "latency: 18 ms" in lines
    assert "note" in viz.hud_lines([], fps=30.0, extra=["note"])
    frame = blank()
    before = frame.copy()
    viz.draw_hud(frame, [], 30.0, extra={"latency": "18 ms"})
    assert changed_pixels(before, frame).any()


def test_hud_without_fps():
    lines = viz.hud_lines([], fps=None)
    assert "fps" not in lines[0]
    assert "hands: 0" in lines[0]


def test_label_lines_carry_id_handedness_gesture_confidence_and_speed():
    track = make_track(track_id=7, handedness="Left", score=0.98,
                       gesture="pinch", velocity=(45.0, 0.0))
    lines = viz.label_lines(track)
    joined = " ".join(lines)
    assert "#7" in joined
    assert "Left" in joined
    assert "98%" in joined
    assert "pinch" in joined and "87%" in joined
    assert "45 px/s" in joined


def test_label_omits_gesture_when_there_is_none():
    lines = viz.label_lines(make_track(gesture="none"))
    assert "none" not in " ".join(lines)
    assert "px/s" in " ".join(lines)


def test_label_for_unknown_handedness():
    track = make_track(handedness="unknown")
    assert "hand" in viz.label_lines(track)[0]


def test_label_stays_on_screen_for_a_hand_at_the_top_edge():
    """A label at y=2 would be drawn off the top and never seen."""
    frame = blank(240, 320)
    viz.draw_hand(frame, make_track(box=(10, 0, 90, 60)), show=("label",))
    assert changed_pixels(blank(240, 320), frame).any()


# ---------------------------------------------------------------- movement wording

@pytest.mark.parametrize("path, expected", [
    ([(100, 200), (100, 150)], "up"),
    ([(100, 100), (100, 160)], "down"),
    ([(100, 100), (160, 100)], "right"),
    ([(160, 100), (100, 100)], "left"),
    ([(100, 100), (140, 60)], "up-right"),
])
def test_movement_direction_words(path, expected):
    assert viz.movement_text(deque(path)).startswith(expected)


def test_movement_reports_distance_in_pixels():
    assert viz.movement_text(deque([(100, 200), (100, 188)])) == "up 12 px"


def test_movement_deadzone_calls_jitter_still():
    assert viz.movement_text(deque([(100, 100), (101, 100)])) == "still"
    assert viz.movement_text(deque()) == "still"
    assert viz.movement_text(None) == "still"
    assert viz.movement_text(deque([(100, 100)])) == "still"


# ---------------------------------------------------------------- degenerate input

def test_no_pose_no_mask_no_trajectory():
    frame = blank()
    track = make_track(pose=False, mask=False, trajectory=False)
    out = viz.render(frame, [track], 30.0)
    assert out.shape == (240, 320, 3)


def test_track_with_nothing_but_an_id():
    frame = blank()
    track = TrackedHand(track_id=1, detection=None)
    viz.render(frame, [track], 30.0)
    assert frame.shape == (240, 320, 3)


def test_empty_and_none_hand_lists():
    frame = blank()
    viz.render(frame, [], 30.0)
    viz.render(frame, None, None)
    viz.draw_hud(frame, None, None)
    viz.draw_hand(frame, None)
    assert frame.shape == (240, 320, 3)


def test_zero_sized_mask():
    frame = blank()
    empty = HandMask(mask=np.zeros((0, 0), np.uint8), origin=(10, 10))
    viz.draw_mask(frame, empty, (0, 255, 0))
    assert not changed_pixels(blank(), frame).any()


def test_all_zero_mask_paints_nothing():
    frame = blank(120, 120, value=50)
    before = frame.copy()
    viz.draw_mask(frame, make_mask(origin=(10, 10), size=40, fill=False), (0, 255, 0))
    assert np.array_equal(before, frame)


def test_bool_mask_is_tolerated():
    frame = blank(120, 120, value=50)
    roi = np.zeros((40, 40), bool)
    roi[10:30, 10:30] = True
    viz.draw_mask(frame, HandMask(mask=roi, origin=(20, 20)), (0, 255, 0))
    assert changed_pixels(blank(120, 120, value=50), frame).any()


def test_three_channel_mask_is_tolerated():
    """get_segmentation_mask() in this repo returns (720, 1280, 3), so a segmenter
    built on it will hand one straight through."""
    frame = blank(120, 120, value=50)
    roi = np.zeros((40, 40, 3), np.uint8)
    roi[10:30, 10:30] = 255
    viz.draw_mask(frame, HandMask(mask=roi, origin=(20, 20)), (0, 255, 0))
    diff = changed_pixels(blank(120, 120, value=50), frame)
    assert diff.any()
    rows, cols = np.nonzero(diff)
    assert rows.min() >= 30 and cols.min() >= 30


def test_colour_given_as_a_list_does_not_break_the_plate_cache():
    frame = blank(120, 120, value=50)
    viz.draw_mask(frame, make_mask(origin=(20, 20), size=60), [0, 255, 0])
    assert changed_pixels(blank(120, 120, value=50), frame).any()


def test_empty_trajectory_deque():
    frame = blank()
    viz.draw_trail(frame, deque(maxlen=64), (0, 255, 0))
    assert not changed_pixels(blank(), frame).any()


def test_pose_with_fewer_than_21_points():
    frame = blank()
    viz.draw_skeleton(frame, HandPose(points=np.zeros((5, 2), np.float32)), (0, 255, 0))
    assert frame.shape == (240, 320, 3)


def test_pose_with_empty_points():
    frame = blank()
    viz.draw_skeleton(frame, HandPose(points=np.zeros((0, 2), np.float32)), (0, 255, 0))
    viz.draw_skeleton(frame, HandPose(points=None), (0, 255, 0))
    assert not changed_pixels(blank(), frame).any()


# ---------------------------------------------------------------- out of frame

@pytest.mark.parametrize("box", [
    (-500, -500, -100, -100),           # entirely off the top-left
    (400, 300, 900, 800),               # entirely off the bottom-right
    (-50, -50, 400, 400),               # bigger than the frame
    (150, 150, 50, 50),                 # inverted corners
    (10 ** 9, 10 ** 9, 10 ** 9 + 5, 10 ** 9 + 5),
    (-10 ** 12, -10 ** 12, 10 ** 12, 10 ** 12),
])
def test_out_of_frame_boxes_do_not_raise(box):
    frame = blank()
    viz.draw_hand(frame, make_track(box=box))
    assert frame.shape == (240, 320, 3)


@pytest.mark.parametrize("origin", [(-50, -50), (-500, -500), (300, 220), (5000, 5000),
                                    (-20, 200), (310, -20)])
def test_mask_origins_outside_the_frame_do_not_raise(origin):
    frame = blank()
    viz.draw_mask(frame, make_mask(origin=origin, size=60), (0, 255, 0))
    assert frame.shape == (240, 320, 3)


def test_negative_mask_origin_does_not_wrap_to_the_far_edge():
    """frame[-20:...] is a legal numpy slice that paints the wrong side of the image."""
    frame = blank(240, 320)
    viz.draw_mask(frame, make_mask(origin=(-40, -40), size=50), (0, 255, 0))
    diff = changed_pixels(blank(240, 320), frame)
    assert not diff[200:, :].any(), "mask wrapped around to the bottom of the frame"
    assert not diff[:, 250:].any(), "mask wrapped around to the right of the frame"


def test_pose_points_outside_the_frame_do_not_raise():
    frame = blank()
    for value in (-10000.0, 1e6, 1e12, -1e12):
        points = np.full((21, 2), value, np.float32)
        viz.draw_skeleton(frame, HandPose(points=points), (0, 255, 0))
    assert frame.shape == (240, 320, 3)


def test_non_finite_pose_points_do_not_raise():
    """int(nan) raises ValueError; a pose that failed to converge must not kill a frame."""
    frame = blank()
    points = make_pose(100, 100).points.copy()
    points[3] = np.nan
    points[7] = np.inf
    points[11] = -np.inf
    viz.draw_skeleton(frame, HandPose(points=points), (0, 255, 0))
    assert changed_pixels(blank(), frame).any(), "one bad point wiped out the skeleton"


def test_non_finite_pose_points_are_dropped_not_clamped():
    """NaN must be discarded, not squeezed through the coordinate clamp.

    min(100000, nan) is 100000 in Python -- no exception -- so a missing isfinite()
    check does not crash, it quietly draws a bone from the hand to a point far off the
    bottom-right corner, which paints a bright diagonal streak across the whole frame.
    Checking only that nothing raised would never catch it.
    """
    frame = blank(240, 320)
    points = make_pose(60.0, 60.0, spread=20.0).points.copy()
    points[3] = np.nan
    points[7] = np.inf
    viz.draw_skeleton(frame, HandPose(points=points), (0, 255, 0))

    diff = changed_pixels(blank(240, 320), frame)
    assert diff.any()
    rows, cols = np.nonzero(diff)
    assert rows.max() < 120 and cols.max() < 160, (
        "a non-finite landmark was clamped and drawn as a streak across the frame")


def test_non_finite_trajectory_points_do_not_raise():
    frame = blank()
    viz.draw_trail(frame, deque([(10, 10), (np.nan, 5), (50, 50), (np.inf, np.inf)]),
                   (0, 255, 0))
    assert frame.shape == (240, 320, 3)


def test_everything_off_screen_at_once():
    frame = blank()
    before = frame.copy()
    track = TrackedHand(
        track_id=4,
        detection=HandDetection(box=(-900, -900, -800, -800), score=0.5),
        pose=HandPose(points=np.full((21, 2), -5000.0, np.float32)),
        mask=make_mask(origin=(-500, -500), size=40),
        trajectory=deque([(-900, -900), (-880, -910)]),
    )
    viz.render(frame, [track], 30.0,
               show=("mask", "skeleton", "box", "trail", viz.NO_HUD))
    assert np.array_equal(before, frame)


def test_tiny_frame():
    frame = blank(4, 4)
    viz.render(frame, [make_track()], 30.0)
    assert frame.shape == (4, 4, 3)


# ---------------------------------------------------------------- headless discipline

def test_module_never_touches_the_camera_or_a_gui():
    """Another process owns the camera, and imshow hangs a headless run."""
    source = (REPO / "pipeline" / "visualize.py").read_text()
    for forbidden in ("VideoCapture", "imshow", "namedWindow", "waitKey",
                      "destroyAllWindows", "startWindowThread"):
        assert forbidden not in source, f"visualize.py references {forbidden}"


def test_module_does_not_write_files():
    source = (REPO / "pipeline" / "visualize.py").read_text()
    assert "imwrite" not in source
    assert not re.search(r"\bopen\s*\(", source)


# ---------------------------------------------------------------- real frames

@needs_dataset
def test_render_on_a_real_egohands_frame(sample_frame_path):
    """1280x720 JPEG of a real hand: shape, dtype and the hand still visible."""
    frame = cv2.imread(str(sample_frame_path))
    assert frame is not None and frame.shape == (720, 1280, 3)
    before = frame.copy()

    track = make_track(track_id=2, box=(400, 200, 700, 500))
    track.mask = make_mask(origin=(420, 220), size=200)
    track.pose = make_pose(x=550.0, y=350.0, spread=90.0)
    track.trajectory = deque([(500 + i * 3, 400 - i) for i in range(64)], maxlen=64)

    out = viz.render(frame, [track], 29.4)
    assert out.shape == (720, 1280, 3) and out.dtype == np.uint8

    under = out[260:400, 460:600]
    assert under.std() > 15, "the hand is not visible under the overlay"
    assert np.array_equal(before[:100, 1000:], out[:100, 1000:]), "painted far from the hand"


@needs_dataset
def test_render_across_several_real_frames_is_stable(sample_frame_path):
    """Same track id over consecutive real frames keeps the same colour."""
    folder = Path(sample_frame_path).parent
    frames = sorted(folder.glob("frame_*.jpg"))[:3]
    if len(frames) < 3:
        pytest.skip("need three frames")
    swatches = []
    for path in frames:
        frame = cv2.imread(str(path))
        viz.draw_hand(frame, make_track(track_id=5, box=(100, 100, 300, 300)),
                      show=("box",))
        swatches.append(tuple(int(c) for c in frame[100, 200]))
    assert len(set(swatches)) == 1, swatches


# ---------------------------------------------------------------- performance

def test_render_stays_inside_the_frame_budget():
    """Budget: under 3 ms for two hands at 1280x720. This runs every single frame."""
    rng = np.random.default_rng(0)
    base = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)

    hands = []
    for index, (ox, oy) in enumerate(((200, 200), (800, 300))):
        size = 260
        roi = np.zeros((size, size), np.uint8)
        cv2.circle(roi, (size // 2, size // 2), size // 3, 255, -1)
        for finger in range(5):
            cv2.line(roi, (size // 2, size // 2), (30 + finger * 45, 20), 255, 22)
        hands.append(TrackedHand(
            track_id=index + 1,
            detection=HandDetection(box=(ox, oy, ox + size, oy + size), score=0.93,
                                    handedness="Right"),
            pose=make_pose(ox + size / 2, oy + size / 2, spread=110.0),
            mask=HandMask(mask=roi, origin=(ox, oy)),
            trajectory=deque([(ox + i * 2, oy + int(20 * np.sin(i / 5)))
                              for i in range(64)], maxlen=64),
            gesture="point", gesture_confidence=0.87, velocity=(45.0, -12.0), age=100,
        ))

    for _ in range(20):                                     # warm caches and OpenCV
        viz.render(base.copy(), hands, 30.0)

    timings = []
    for _ in range(120):
        frame = base.copy()
        start = time.perf_counter()
        viz.render(frame, hands, 30.0)
        timings.append((time.perf_counter() - start) * 1000.0)

    median = float(np.median(timings))
    print(f"\nrender: {median:.3f} ms/frame median for 2 hands at 1280x720 "
          f"({median / 2:.3f} ms/hand, p95 {np.percentile(timings, 95):.3f} ms)")
    assert median < 3.0, f"{median:.3f} ms/frame blows the 3 ms budget"


def test_mask_cost_scales_with_the_roi_not_the_frame():
    """Compositing full-frame would make a small hand as expensive as a huge one."""
    rng = np.random.default_rng(1)
    frame = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)
    small = HandMask(mask=np.full((60, 60), 255, np.uint8), origin=(100, 100))
    large = HandMask(mask=np.full((600, 600), 255, np.uint8), origin=(100, 100))

    def cost(mask):
        for _ in range(10):
            viz.draw_mask(frame, mask, (0, 255, 0))
        samples = []
        for _ in range(40):
            start = time.perf_counter()
            viz.draw_mask(frame, mask, (0, 255, 0))
            samples.append(time.perf_counter() - start)
        return float(np.median(samples))

    small_cost, large_cost = cost(small), cost(large)
    print(f"\nmask 60x60: {small_cost * 1000:.3f} ms   600x600: {large_cost * 1000:.3f} ms")
    assert small_cost < large_cost / 4, (small_cost, large_cost)
