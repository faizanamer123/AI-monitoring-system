"""Behaviour tests for webcam_detect_mp.py -- the MediaPipe Tasks path.

NOTHING here opens the camera. Everything runs on still frames from
_LABELLED_SAMPLES, on synthetic arrays, or on stub landmark objects.

The interesting invariants:
  * build_detector() must fail *cleanly* (SystemExit + curl hint) on a bad model path,
    because that is the single most common first-run failure.
  * landmarks_to_box() is pure math and is the only place a bad box can be born.
    Its docstring promises clamping for hands running off the edge of frame; that
    promise is tested here in both directions.
  * VIDEO mode is a state machine keyed on a millisecond clock. Feeding it a
    non-increasing timestamp is a hard error, so the camera loop's clock matters.

Run:  .venv/bin/python -m pytest tests/test_webcam_detect_mp.py -v
"""

import subprocess

import cv2
import numpy as np
import pytest

from conftest import needs_dataset, needs_mp_model

import webcam_detect_mp as mpd

MODEL = "models/hand_landmarker.task"


# --------------------------------------------------------------------------- #
# stubs / helpers
# --------------------------------------------------------------------------- #

class StubLandmark:
    """The only thing landmarks_to_box touches is .x / .y."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


def stub_hand(x_lo, x_hi, y_lo, y_hi, n=21):
    """21 landmarks spanning exactly [x_lo, x_hi] x [y_lo, y_hi]."""
    pts = [StubLandmark(x_lo, y_lo), StubLandmark(x_hi, y_hi)]
    for i in range(n - 2):
        t = (i + 1) / (n - 1)
        pts.append(StubLandmark(x_lo + t * (x_hi - x_lo), y_lo + t * (y_hi - y_lo)))
    return pts


def frames_from(repo, folders=3, per_folder=2):
    """A deterministic handful of real EgoHands frames."""
    root = repo / "_LABELLED_SAMPLES"
    picked = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir())[:folders]:
        picked.extend(sorted(folder.glob("frame_*.jpg"))[:per_folder])
    return picked


def loop_timestamp(started, tick):
    """The exact stamping expression used inside run_on_camera().

        detections = detect(detector, frame, int((tick - started) * 1000))
    """
    return int((tick - started) * 1000)


@pytest.fixture(scope="module")
def image_detector(repo):
    if not (repo / MODEL).is_file():
        pytest.skip("no mediapipe model file")
    return mpd.build_detector(MODEL, max_hands=4, confidence=0.4, video_mode=False)


@pytest.fixture
def video_detector(repo):
    """Function-scoped: VIDEO mode carries timestamp state that must not leak."""
    if not (repo / MODEL).is_file():
        pytest.skip("no mediapipe model file")
    return mpd.build_detector(MODEL, max_hands=4, confidence=0.4, video_mode=True)


@pytest.fixture(scope="module")
def hand_frame(repo):
    """A real frame that MediaPipe reliably finds two hands in."""
    path = repo / "_LABELLED_SAMPLES" / "CARDS_COURTYARD_B_T" / "frame_0011.jpg"
    if not path.is_file():
        pytest.skip("reference frame not present")
    frame = cv2.imread(str(path))
    assert frame is not None
    return frame


# --------------------------------------------------------------------------- #
# build_detector
# --------------------------------------------------------------------------- #

@needs_mp_model
def test_build_detector_image_mode_is_stateless_detector():
    detector = mpd.build_detector(MODEL, max_hands=2, confidence=0.4, video_mode=False)
    assert hasattr(detector, "detect")
    # a stateless IMAGE detector may be called repeatedly with no clock at all
    blank = np.zeros((120, 160, 3), np.uint8)
    assert mpd.detect(detector, blank) == []
    assert mpd.detect(detector, blank) == []


@needs_mp_model
def test_build_detector_video_mode_accepts_a_clock(video_detector):
    blank = np.zeros((120, 160, 3), np.uint8)
    assert hasattr(video_detector, "detect_for_video")
    assert mpd.detect(video_detector, blank, 0) == []
    assert mpd.detect(video_detector, blank, 33) == []


def test_build_detector_missing_model_raises_clean_systemexit(tmp_path):
    """A missing .task file is the #1 first-run failure; it must not be a traceback."""
    missing = tmp_path / "definitely_not_here.task"
    with pytest.raises(SystemExit) as excinfo:
        mpd.build_detector(str(missing), max_hands=2, confidence=0.4, video_mode=False)

    message = str(excinfo.value)
    assert str(missing) in message, "the error should name the path it tried"
    assert "curl" in message, "the error should hand the user a download command"
    assert "hand_landmarker.task" in message
    # __cause__ suppressed with `from None` -- the user sees the hint, not mediapipe guts
    assert excinfo.value.__cause__ is None


def test_build_detector_missing_model_is_systemexit_in_video_mode_too(tmp_path):
    with pytest.raises(SystemExit):
        mpd.build_detector(str(tmp_path / "nope.task"), 2, 0.4, video_mode=True)


# --------------------------------------------------------------------------- #
# landmarks_to_box -- pure math, no model needed
# --------------------------------------------------------------------------- #

def test_box_covers_the_landmark_extent_and_is_in_pixels():
    # span x in [0.2, 0.6] -> pad 0.048 ; span y in [0.3, 0.5] -> pad 0.024
    box = mpd.landmarks_to_box(stub_hand(0.2, 0.6, 0.3, 0.5), 1000, 500)
    x1, y1, x2, y2 = box
    assert all(isinstance(v, int) for v in box), "boxes must be integer pixel coords"
    assert abs(x1 - 152) <= 1 and abs(x2 - 648) <= 1
    assert abs(y1 - 138) <= 1 and abs(y2 - 262) <= 1


def test_box_margin_pads_outward_by_the_documented_fraction():
    width = height = 1000
    raw_lo, raw_hi = 0.30, 0.70
    x1, y1, x2, y2 = mpd.landmarks_to_box(
        stub_hand(raw_lo, raw_hi, raw_lo, raw_hi), width, height)

    raw_span = (raw_hi - raw_lo) * width
    assert (x2 - x1) > raw_span, "BOX_MARGIN must expand the raw landmark extent"
    # padded on BOTH sides -> span grows by 2 * BOX_MARGIN
    assert (x2 - x1) / raw_span == pytest.approx(1 + 2 * mpd.BOX_MARGIN, abs=0.01)
    assert (y2 - y1) / raw_span == pytest.approx(1 + 2 * mpd.BOX_MARGIN, abs=0.01)
    # padding is symmetric: the centre does not move
    assert (x1 + x2) / 2 == pytest.approx((raw_lo + raw_hi) / 2 * width, abs=1.5)


def test_box_scales_with_frame_size():
    """The same normalised hand must map to the same fraction of any frame.

    Tolerance is relative: each coordinate is truncated to a whole pixel
    independently, so the small frame's span carries up to 2px of rounding.
    """
    small = mpd.landmarks_to_box(stub_hand(0.25, 0.75, 0.25, 0.75), 320, 240)
    large = mpd.landmarks_to_box(stub_hand(0.25, 0.75, 0.25, 0.75), 1280, 960)
    assert (large[2] - large[0]) / 1280 == pytest.approx((small[2] - small[0]) / 320,
                                                        abs=2 / 320)
    assert (large[3] - large[1]) / 960 == pytest.approx((small[3] - small[1]) / 240,
                                                        abs=2 / 240)
    # 0.25..0.75 padded by 12% of the 0.5 span -> 0.19 .. 0.81 of the frame
    assert large[0] / 1280 == pytest.approx(0.19, abs=0.005)
    assert large[2] / 1280 == pytest.approx(0.81, abs=0.005)


def test_box_clamps_at_the_low_edge_when_a_hand_runs_off_left_and_top():
    """Landmarks partly below 0 must not produce negative pixel coords."""
    x1, y1, x2, y2 = mpd.landmarks_to_box(stub_hand(-0.20, 0.30, -0.15, 0.40), 640, 480)
    assert x1 == 0 and y1 == 0
    assert x2 > x1 and y2 > y1


def test_box_clamps_at_the_high_edge_when_a_hand_runs_off_right_and_bottom():
    """Landmarks partly above 1 must not exceed the last addressable pixel."""
    width, height = 640, 480
    x1, y1, x2, y2 = mpd.landmarks_to_box(
        stub_hand(0.70, 1.40, 0.60, 1.30), width, height)
    assert x2 == width - 1 and y2 == height - 1
    assert x2 > x1 and y2 > y1


def test_box_stays_inside_the_frame_for_a_hand_entirely_off_frame():
    """The docstring promises clamping for out-of-range landmarks. Both ends.

    A hand that has walked fully past an edge yields landmarks entirely outside
    [0, 1]. The box must still be a usable in-frame rectangle: no negative
    coordinate, nothing past width-1 / height-1, and x2 > x1 / y2 > y1.
    """
    width, height = 640, 480
    cases = {
        "off the right edge": stub_hand(1.05, 1.30, 0.40, 0.60),
        "off the left edge": stub_hand(-0.50, -0.20, 0.40, 0.60),
        "off the bottom edge": stub_hand(0.40, 0.60, 1.05, 1.30),
        "off the top edge": stub_hand(0.40, 0.60, -0.50, -0.20),
    }
    broken = {}
    for name, hand in cases.items():
        x1, y1, x2, y2 = mpd.landmarks_to_box(hand, width, height)
        if not (0 <= x1 <= width - 1 and 0 <= x2 <= width - 1
                and 0 <= y1 <= height - 1 and 0 <= y2 <= height - 1
                and x2 >= x1 and y2 >= y1):
            broken[name] = (x1, y1, x2, y2)
    assert not broken, (
        f"landmarks_to_box left the frame for {width}x{height}: {broken} -- "
        "max(0, ...) only guards the low end of x1/y1 and min(w-1, ...) only the "
        "high end of x2/y2, so x1/y1 can exceed the frame and x2/y2 can go negative"
    )


# --------------------------------------------------------------------------- #
# detect() -- real model, real frames, no camera
# --------------------------------------------------------------------------- #

@needs_mp_model
@needs_dataset
def test_detect_finds_hands_in_a_real_egohands_frame(image_detector, hand_frame):
    detections = mpd.detect(image_detector, hand_frame)
    assert len(detections) >= 1, "MediaPipe found no hands in a two-hand EgoHands frame"
    box, label, score, landmarks = detections[0]
    assert len(box) == 4
    assert isinstance(label, str)
    assert len(landmarks) == 21, "MediaPipe hand landmarker returns 21 joints"


@needs_mp_model
def test_detect_finds_nothing_in_blank_frames(image_detector):
    for name, frame in [
        ("black", np.zeros((480, 640, 3), np.uint8)),
        ("white", np.full((480, 640, 3), 255, np.uint8)),
        ("mid grey", np.full((480, 640, 3), 127, np.uint8)),
    ]:
        assert mpd.detect(image_detector, frame) == [], f"phantom hand in a {name} frame"


@needs_mp_model
@needs_dataset
def test_handedness_label_and_score_are_sane(image_detector, repo):
    seen_labels = set()
    for path in frames_from(repo, folders=4, per_folder=2):
        frame = cv2.imread(str(path))
        for _, label, score, _ in mpd.detect(image_detector, frame):
            assert label in ("Left", "Right"), f"unexpected handedness {label!r}"
            assert 0.0 <= float(score) <= 1.0, f"score {score} out of [0, 1]"
            seen_labels.add(label)
    assert seen_labels, "no detections at all across the sampled frames"


@needs_mp_model
@needs_dataset
def test_boxes_are_valid_in_frame_pixel_rectangles(image_detector, repo):
    checked = 0
    for path in frames_from(repo, folders=5, per_folder=2):
        frame = cv2.imread(str(path))
        height, width = frame.shape[:2]
        for (x1, y1, x2, y2), _, _, _ in mpd.detect(image_detector, frame):
            assert 0 <= x1 < width and 0 <= x2 < width, f"{path.name}: x out of frame"
            assert 0 <= y1 < height and 0 <= y2 < height, f"{path.name}: y out of frame"
            assert x2 > x1, f"{path.name}: degenerate/inverted box in x"
            assert y2 > y1, f"{path.name}: degenerate/inverted box in y"
            checked += 1
    assert checked >= 5, f"only {checked} boxes checked -- test would prove nothing"


@needs_mp_model
@needs_dataset
def test_box_actually_wraps_its_own_landmarks(image_detector, hand_frame):
    """The returned box must contain the landmarks it was derived from."""
    height, width = hand_frame.shape[:2]
    detections = mpd.detect(image_detector, hand_frame)
    assert detections
    for (x1, y1, x2, y2), _, _, landmarks in detections:
        for point in landmarks:
            px, py = point.x * width, point.y * height
            assert x1 - 1 <= px <= x2 + 1, "landmark falls outside its own box in x"
            assert y1 - 1 <= py <= y2 + 1, "landmark falls outside its own box in y"


@needs_mp_model
@needs_dataset
def test_max_hands_caps_the_number_of_detections(hand_frame):
    counts = {}
    for max_hands in (1, 2, 4):
        detector = mpd.build_detector(MODEL, max_hands, 0.4, video_mode=False)
        counts[max_hands] = len(mpd.detect(detector, hand_frame))
        assert counts[max_hands] <= max_hands, f"--max-hands {max_hands} not honoured"
    assert counts[2] >= counts[1], "a two-hand frame should give more hands at max-hands 2"


@needs_mp_model
@needs_dataset
def test_confidence_threshold_is_wired_through(repo):
    """--confidence must reach MediaPipe, not be silently ignored."""
    lenient = mpd.build_detector(MODEL, 4, 0.1, video_mode=False)
    strict = mpd.build_detector(MODEL, 4, 0.95, video_mode=False)
    n_lenient = n_strict = 0
    for path in frames_from(repo, folders=5, per_folder=2):
        frame = cv2.imread(str(path))
        n_lenient += len(mpd.detect(lenient, frame))
        n_strict += len(mpd.detect(strict, frame))
    assert n_lenient > 0
    assert n_strict <= n_lenient, "a stricter threshold produced MORE detections"
    assert n_strict < n_lenient, "the confidence argument had no effect at all"


# --------------------------------------------------------------------------- #
# VIDEO mode timestamps
# --------------------------------------------------------------------------- #

@needs_mp_model
@needs_dataset
def test_video_mode_accepts_strictly_increasing_timestamps(video_detector, hand_frame):
    for stamp in (0, 33, 66, 100):
        mpd.detect(video_detector, hand_frame, stamp)  # must not raise


@needs_mp_model
@needs_dataset
def test_video_mode_rejects_a_decreasing_timestamp(video_detector, hand_frame):
    mpd.detect(video_detector, hand_frame, 500)
    with pytest.raises(ValueError, match="monotonically increasing"):
        mpd.detect(video_detector, hand_frame, 200)


@needs_mp_model
@needs_dataset
def test_video_mode_rejects_a_repeated_timestamp(video_detector, hand_frame):
    """Not merely non-decreasing: MediaPipe wants STRICTLY increasing."""
    mpd.detect(video_detector, hand_frame, 500)
    with pytest.raises(ValueError, match="monotonically increasing"):
        mpd.detect(video_detector, hand_frame, 500)


def test_camera_loop_timestamp_is_strictly_increasing_for_fast_frames():
    """The stamps fed to detect_for_video must strictly increase, even at high fps.

    This exercises webcam_detect_mp.monotonic_timestamp_ms directly rather than
    re-implementing the arithmetic, so the guard in run_on_camera is what is under
    test. Ticks below are deliberately close enough that the naive
    int((tick - started) * 1000) collapses two frames onto the same millisecond.
    """
    from webcam_detect_mp import monotonic_timestamp_ms

    started = 100.0
    ticks = [100.0000, 100.0003, 100.0011, 100.0012, 100.0029]

    stamps, last = [], -1
    for tick in ticks:
        last = monotonic_timestamp_ms(started, tick, last)
        stamps.append(last)

    assert all(b > a for a, b in zip(stamps, stamps[1:])), (
        f"stamps must strictly increase, got {stamps}"
    )
    # and it must not drift far from real time just to stay monotonic
    assert stamps[-1] <= int((ticks[-1] - started) * 1000) + len(ticks)


@pytest.mark.parametrize("show_landmarks", [False, True])
def test_draw_preserves_shape_and_paints_something(image_detector, hand_frame,
                                                   show_landmarks):
    detections = mpd.detect(image_detector, hand_frame)
    assert detections, "need at least one detection to test drawing"

    canvas = hand_frame.copy()
    out = mpd.draw(canvas, detections, show_landmarks, fps=27.4)

    assert out.shape == hand_frame.shape
    assert out.dtype == hand_frame.dtype
    assert not np.array_equal(out, hand_frame), "draw() painted nothing"


@needs_mp_model
@needs_dataset
def test_draw_with_landmarks_paints_more_than_without(image_detector, hand_frame):
    detections = mpd.detect(image_detector, hand_frame)
    plain = mpd.draw(hand_frame.copy(), detections, False)
    joints = mpd.draw(hand_frame.copy(), detections, True)
    changed_plain = int(np.count_nonzero(np.any(plain != hand_frame, axis=2)))
    changed_joints = int(np.count_nonzero(np.any(joints != hand_frame, axis=2)))
    assert changed_joints > changed_plain, "--landmarks drew no extra pixels"


def test_draw_on_empty_detections_is_a_no_op_apart_from_the_hud():
    frame = np.full((240, 320, 3), 40, np.uint8)
    untouched = mpd.draw(frame.copy(), [])
    assert np.array_equal(untouched, frame), "no detections should mean no drawing"
    with_hud = mpd.draw(frame.copy(), [], fps=12.0)
    assert with_hud.shape == frame.shape
    assert not np.array_equal(with_hud, frame), "the fps HUD should be drawn"


def test_draw_survives_a_box_flush_against_the_frame_edge():
    """The label chip is drawn ABOVE y1, so a box at y1=0 goes negative."""
    frame = np.zeros((240, 320, 3), np.uint8)
    landmarks = stub_hand(0.0, 0.2, 0.0, 0.2)
    detections = [
        ((0, 0, 60, 40), "Left", 0.91, landmarks),
        ((260, 200, 319, 239), "Right", 0.88, landmarks),
    ]
    out = mpd.draw(frame, detections, show_landmarks=True, fps=30.0)
    assert out.shape == (240, 320, 3)


def test_draw_colours_left_and_right_differently():
    frame = np.zeros((200, 200, 3), np.uint8)
    landmarks = stub_hand(0.3, 0.6, 0.3, 0.6)
    left = mpd.draw(frame.copy(), [((20, 60, 120, 160), "Left", 0.9, landmarks)])
    right = mpd.draw(frame.copy(), [((20, 60, 120, 160), "Right", 0.9, landmarks)])
    assert not np.array_equal(left, right), "handedness must be visually distinguishable"
    assert mpd.LEFT_COLOR != mpd.RIGHT_COLOR


# --------------------------------------------------------------------------- #
# CLI entry point (subprocess, no camera: --image only)
# --------------------------------------------------------------------------- #

def run_cli(python_bin, repo, tmp_path, *args):
    """Run the --image path from a scratch cwd so the repo stays clean."""
    return subprocess.run(
        [python_bin, str(repo / "webcam_detect_mp.py"), "--model", str(repo / MODEL), *args],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=180,
    )


@needs_mp_model
@needs_dataset
def test_cli_image_mode_reports_hands_and_writes_a_render(python_bin, repo, tmp_path):
    frame = repo / "_LABELLED_SAMPLES" / "CARDS_COURTYARD_B_T" / "frame_0011.jpg"
    result = run_cli(python_bin, repo, tmp_path, "--image", str(frame))
    assert result.returncode == 0, result.stderr[-2000:]
    assert "hand(s) in" in result.stdout, result.stdout
    count = int(result.stdout.split()[0])
    assert count >= 1, f"CLI reported {count} hands in a two-hand frame"
    rendered = tmp_path / "mp_detect_result.png"
    assert rendered.is_file(), "CLI promised mp_detect_result.png but wrote nothing"
    assert cv2.imread(str(rendered)) is not None, "the render is not a readable image"


def test_cli_rejects_an_unreadable_image_cleanly(python_bin, repo, tmp_path):
    result = run_cli(python_bin, repo, tmp_path, "--image", str(tmp_path / "ghost.png"))
    assert result.returncode != 0
    assert "could not read" in (result.stdout + result.stderr)
    assert "Traceback" not in result.stderr, "should be a clean SystemExit, not a traceback"
