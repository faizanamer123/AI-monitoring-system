"""Behaviour tests for webcam_detect_cv.py -- the classical skin+contour detector.

This module is DELIBERATELY not accuracy-tested: webcam_detect_cv boxes faces and
sleeves and says so in its own docstring. What is tested here is that it is correct
and robust *on its own terms*:

  - skin_mask produces a well-formed binary mask for any 8-bit BGR input
  - finger_defects survives every degenerate input and both OpenCV output layouts
  - score_contour honours its documented area / aspect / solidity gates and never
    emits a score outside [0, 1]
  - detect is bounded, sorted, deterministic and mirror-equivariant
  - the --image CLI is headless (never touches the camera) and writes its two PNGs

NOTHING here opens a camera or a GUI window.

Run:  .venv/bin/python -m pytest tests/test_webcam_detect_cv.py -v
"""

import hashlib
import re
import subprocess
import sys

import cv2
import numpy as np
import pytest

from conftest import needs_dataset

import webcam_detect_cv as cvdet


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_args(**overrides):
    """Real CLI defaults from parse_args(), with selected fields overridden.

    Built through the module's own parser rather than a hand-rolled namespace so
    that a change to a default is exercised here instead of silently diverging.
    """
    saved = sys.argv
    sys.argv = ["webcam_detect_cv.py"]
    try:
        args = cvdet.parse_args()
    finally:
        sys.argv = saved
    for key, value in overrides.items():
        assert hasattr(args, key), f"parse_args() has no {key!r}"
        setattr(args, key, value)
    return args


def solid(bgr, h=60, w=60):
    return np.full((h, w, 3), bgr, np.uint8)


def hand_contour():
    """A four-fingered hand silhouette, painted then re-extracted with findContours
    so it is shaped exactly like something detect() would really hand to
    score_contour (CHAIN_APPROX_SIMPLE, int32, closed)."""
    canvas = np.zeros((400, 400), np.uint8)
    cv2.fillPoly(canvas, [np.array([[120, 380], [280, 380], [280, 240], [120, 240]], np.int32)], 255)
    for i in range(4):
        x = 135 + i * 40
        cv2.rectangle(canvas, (x, 90), (x + 22, 250), 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def disc_contour(radius=90, centre=(200, 200)):
    canvas = np.zeros((400, 400), np.uint8)
    cv2.circle(canvas, centre, radius, 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def star_contour(points=8, inner=0.18, radius=100.0, centre=(200.0, 200.0)):
    """A spiky star: area is in range and aspect is ~1, but solidity is far below
    the 0.30 floor, so only the solidity gate can reject it."""
    theta = np.linspace(0.0, 2 * np.pi, 2 * points, endpoint=False)
    r = np.where(np.arange(2 * points) % 2 == 0, radius, radius * inner)
    pts = np.stack([centre[0] + r * np.cos(theta), centre[1] + r * np.sin(theta)], axis=1)
    return pts.astype(np.int32).reshape(-1, 1, 2)


def rect_contour(w, h, x=20, y=20):
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], np.int32)


def defects_rows(depths_px):
    """A convexityDefects payload with the given valley depths, in OpenCV's
    fixed-point 1/256-px units. Columns are (start, end, farthest, depth)."""
    return np.array([[0, 1, 2, int(round(d * 256))] for d in depths_px], np.int32)


FRAME_AREA_400 = 400 * 400
SQUARE_HULL = cv2.convexHull(rect_contour(80, 80), returnPoints=False)  # 4 indices, passes the guard

CLI = "webcam_detect_cv.py"


@pytest.fixture(scope="session")
def frames(repo):
    """Up to five real EgoHands frames, decoded once."""
    dataset = repo / "_LABELLED_SAMPLES"
    if not dataset.is_dir():
        pytest.skip("_LABELLED_SAMPLES not present")
    paths = []
    for folder in sorted(p for p in dataset.iterdir() if p.is_dir()):
        found = sorted(folder.glob("frame_*.jpg"))
        if found:
            paths.append(found[0])
        if len(paths) == 5:
            break
    if not paths:
        pytest.skip("no frames found under _LABELLED_SAMPLES")
    images = [cv2.imread(str(p)) for p in paths]
    assert all(img is not None for img in images)
    return list(zip(paths, images))


@pytest.fixture(scope="session")
def cli_run(repo, python_bin, sample_frame_path, tmp_path_factory):
    """One --image CLI invocation in a scratch cwd. Session-scoped: the process
    spends most of its life importing cv2, and several tests read the same result."""
    workdir = tmp_path_factory.mktemp("cli_image")
    proc = subprocess.run(
        [python_bin, str(repo / CLI), "--image", str(sample_frame_path)],
        cwd=str(workdir), capture_output=True, text=True, timeout=120,
    )
    return proc, workdir


# --------------------------------------------------------------------------
# skin_mask
# --------------------------------------------------------------------------

@needs_dataset
def test_skin_mask_is_single_channel_uint8_binary_and_same_size(frames):
    for path, img in frames:
        mask = cvdet.skin_mask(img)
        assert mask.dtype == np.uint8, path
        assert mask.ndim == 2, f"{path}: mask must be single channel, got {mask.shape}"
        assert mask.shape == img.shape[:2], f"{path}: {mask.shape} != {img.shape[:2]}"
        assert set(np.unique(mask).tolist()) <= {0, 255}, f"{path}: {np.unique(mask)}"


@pytest.mark.parametrize("name,img", [
    ("black", np.zeros((240, 320, 3), np.uint8)),
    ("white", np.full((240, 320, 3), 255, np.uint8)),
    ("mid-grey", np.full((240, 320, 3), 128, np.uint8)),
])
def test_skin_mask_on_achromatic_extremes_is_empty_not_a_crash(name, img):
    """Black, white and grey carry no chroma, so nothing may be called skin.
    They are also the classic crash inputs (all-zero / saturated) for the
    threshold-then-morphology pipeline."""
    mask = cvdet.skin_mask(img)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert np.count_nonzero(mask) == 0, f"{name} was classified as skin"


def test_skin_mask_on_random_noise_stays_binary():
    rng = np.random.default_rng(20240902)
    img = rng.integers(0, 256, (200, 260, 3), dtype=np.uint8)
    mask = cvdet.skin_mask(img)
    assert mask.shape == (200, 260)
    assert set(np.unique(mask).tolist()) <= {0, 255}


@pytest.mark.parametrize("bgr", [(140, 170, 210), (100, 140, 190), (60, 90, 140)])
def test_skin_mask_accepts_plausible_skin_tones(bgr):
    """The gate must actually be open: a flat patch inside both the YCrCb and HSV
    envelopes has to survive the two thresholds *and* the morphology."""
    mask = cvdet.skin_mask(solid(bgr))
    assert mask.mean() / 255.0 > 0.9, f"{bgr} rejected, mean={mask.mean()}"


@pytest.mark.parametrize("bgr", [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 200, 200)])
def test_skin_mask_rejects_saturated_non_skin(bgr):
    assert np.count_nonzero(cvdet.skin_mask(solid(bgr))) == 0


@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 4, 3), (9, 9, 3), (11, 7, 3)])
def test_skin_mask_survives_frames_smaller_than_its_kernels(shape):
    """The 5px median blur and 9x9 ellipse are bigger than these frames."""
    mask = cvdet.skin_mask(np.full(shape, 130, np.uint8))
    assert mask.shape == shape[:2]
    assert mask.dtype == np.uint8


# --------------------------------------------------------------------------
# finger_defects -- the OpenCV-version compatibility shim
# --------------------------------------------------------------------------

def test_finger_defects_counts_valleys_on_a_notched_contour():
    contour = hand_contour()
    hull = cv2.convexHull(contour, returnPoints=False)
    assert cvdet.finger_defects(contour, hull) > 0, \
        "a four-fingered silhouette must register at least one deep valley"


def test_finger_defects_is_zero_for_a_convex_contour():
    """A 60-gon has no defects at all; OpenCV signals that with None."""
    theta = np.linspace(0.0, 2 * np.pi, 60, endpoint=False)
    circle = np.stack([200 + 100 * np.cos(theta), 200 + 100 * np.sin(theta)], axis=1)
    circle = circle.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(circle, returnPoints=False)
    assert cvdet.finger_defects(circle, hull) == 0


@pytest.mark.parametrize("pts", [
    [[10, 10], [100, 10], [50, 90]],   # triangle: hull has 3 indices
    [[10, 10], [100, 10]],             # two points
    [[10, 10]],                        # one point
])
def test_finger_defects_is_zero_for_contours_with_too_few_points(pts):
    contour = np.array(pts, np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(contour, returnPoints=False)
    assert cvdet.finger_defects(contour, hull) == 0


def test_finger_defects_is_zero_for_a_none_hull():
    assert cvdet.finger_defects(hand_contour(), None) == 0


@pytest.mark.parametrize("shape,label", [((-1, 1, 4), "OpenCV 4.x (N,1,4)"),
                                         ((-1, 4), "OpenCV 5.x (N,4)")])
def test_finger_defects_reads_both_opencv_output_layouts(monkeypatch, shape, label):
    """The shim's whole reason to exist: identical depths must yield an identical
    count whether cv2 hands back (N,1,4) or (N,4)."""
    payload = defects_rows([5.0, 20.0, 40.0, 100.0]).reshape(*shape)
    monkeypatch.setattr(cv2, "convexityDefects", lambda c, h: payload)
    assert cvdet.finger_defects(hand_contour(), SQUARE_HULL) == 3, label


@pytest.mark.parametrize("shape", [(0, 1, 4), (0, 4)])
def test_finger_defects_is_zero_for_an_empty_defects_array(monkeypatch, shape):
    monkeypatch.setattr(cv2, "convexityDefects", lambda c, h: np.empty(shape, np.int32))
    assert cvdet.finger_defects(hand_contour(), SQUARE_HULL) == 0


def test_finger_defects_is_zero_when_cv2_returns_none(monkeypatch):
    monkeypatch.setattr(cv2, "convexityDefects", lambda c, h: None)
    assert cvdet.finger_defects(hand_contour(), SQUARE_HULL) == 0


def test_finger_defects_swallows_cv2_error(monkeypatch):
    def boom(contour, hull):
        raise cv2.error("simulated non-monotonous hull")
    monkeypatch.setattr(cv2, "convexityDefects", boom)
    assert cvdet.finger_defects(hand_contour(), SQUARE_HULL) == 0


def test_finger_defects_applies_the_depth_threshold(monkeypatch):
    """Depth is fixed-point 1/256 px and only valleys deeper than ~12 px count,
    which is what stops contour jitter from being read as fingers."""
    monkeypatch.setattr(cv2, "convexityDefects",
                        lambda c, h: defects_rows([1.0, 5.0, 11.0]).reshape(-1, 1, 4))
    assert cvdet.finger_defects(hand_contour(), SQUARE_HULL) == 0, "shallow jitter counted"

    monkeypatch.setattr(cv2, "convexityDefects",
                        lambda c, h: defects_rows([1.0, 13.0, 60.0]).reshape(-1, 1, 4))
    assert cvdet.finger_defects(hand_contour(), SQUARE_HULL) == 2, "deep valleys missed"


# --------------------------------------------------------------------------
# score_contour
# --------------------------------------------------------------------------

def test_score_contour_rejects_blobs_below_min_area():
    args = make_args()
    tiny = rect_contour(20, 20)                       # 400 px^2 < 0.004 * 160000
    assert cv2.contourArea(tiny) < args.min_area * FRAME_AREA_400
    assert cvdet.score_contour(tiny, FRAME_AREA_400, args) is None


def test_score_contour_rejects_blobs_above_max_area():
    args = make_args()
    huge = rect_contour(250, 250, x=10, y=10)         # 62500 px^2 > 0.25 * 160000
    assert cv2.contourArea(huge) > args.max_area * FRAME_AREA_400
    assert cvdet.score_contour(huge, FRAME_AREA_400, args) is None


def test_score_contour_accepts_a_blob_just_inside_the_area_band():
    """Proves the two rejections above are the area gate and not a blanket 'no'."""
    args = make_args()
    ok = rect_contour(30, 30)                         # 900 px^2, inside [640, 40000]
    assert args.min_area * FRAME_AREA_400 < cv2.contourArea(ok) < args.max_area * FRAME_AREA_400
    assert cvdet.score_contour(ok, FRAME_AREA_400, args) is not None


@pytest.mark.parametrize("w,h,label", [(300, 10, "wide streak"), (10, 300, "tall streak")])
def test_score_contour_rejects_extreme_aspect_ratios(w, h, label):
    args = make_args()
    contour = rect_contour(w, h, x=10, y=10)
    area = cv2.contourArea(contour)
    assert args.min_area * FRAME_AREA_400 <= area <= args.max_area * FRAME_AREA_400, \
        f"{label} must be rejected by aspect, not area"
    assert cvdet.score_contour(contour, FRAME_AREA_400, args) is None, label


def test_score_contour_rejects_a_ragged_low_solidity_blob():
    """An 8-point star sits inside the area band with aspect ~1, so only the
    solidity floor (0.30) can turn it away."""
    args = make_args()
    star = star_contour()
    area = cv2.contourArea(star)
    x, y, w, h = cv2.boundingRect(star)
    assert args.min_area * FRAME_AREA_400 <= area <= args.max_area * FRAME_AREA_400
    assert 0.3 <= w / float(h) <= 3.0
    assert area / cv2.contourArea(cv2.convexHull(star)) < 0.30
    assert cvdet.score_contour(star, FRAME_AREA_400, args) is None


def test_score_contour_returns_a_well_formed_quadruple():
    args = make_args()
    result = cvdet.score_contour(hand_contour(), FRAME_AREA_400, args)
    assert result is not None
    assert len(result) == 4
    score, box, fingers, solidity = result
    assert isinstance(score, float) and 0.0 <= score <= 1.0
    assert len(box) == 4 and all(isinstance(v, (int, np.integer)) for v in box)
    assert isinstance(fingers, int) and fingers >= 0
    assert 0.30 <= solidity <= 1.0 + 1e-9


def test_score_contour_score_never_leaves_the_unit_interval():
    """Randomised blobs of every solidity/finger combination the pipeline can
    produce. The score formula sums two weighted terms that can exceed 1 before
    clamping, so this is the guard on that clamp."""
    args = make_args()
    rng = np.random.default_rng(7)
    scored = 0
    for _ in range(300):
        canvas = np.zeros((300, 300), np.uint8)
        for _ in range(int(rng.integers(1, 5))):
            centre = tuple(int(v) for v in rng.integers(0, 300, 2))
            cv2.circle(canvas, centre, int(rng.integers(3, 70)), 255, -1)
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            result = cvdet.score_contour(contour, 300 * 300, args)
            if result is None:
                continue
            scored += 1
            score, box, fingers, solidity = result
            assert 0.0 <= score <= 1.0, f"score {score} out of range (f={fingers}, s={solidity})"
            assert box[2] > 0 and box[3] > 0
    assert scored > 50, f"fuzz produced only {scored} scored blobs -- test is not exercising much"


def test_score_contour_prefers_notches_over_a_solid_blob():
    """The module's central claim: notches score above face-like solidity."""
    args = make_args()
    hand = cvdet.score_contour(hand_contour(), FRAME_AREA_400, args)
    disc = cvdet.score_contour(disc_contour(), FRAME_AREA_400, args)
    assert hand is not None and disc is not None
    assert hand[0] > disc[0], f"notched {hand[0]:.3f} !> solid {disc[0]:.3f}"


# --------------------------------------------------------------------------
# detect
# --------------------------------------------------------------------------

@needs_dataset
def test_detect_returns_scored_list_and_the_same_mask_skin_mask_would(frames):
    path, img = frames[0]
    detections, mask = cvdet.detect(img, make_args())
    assert isinstance(detections, list)
    assert np.array_equal(mask, cvdet.skin_mask(img))
    for score, box, fingers, solidity in detections:
        assert 0.0 <= score <= 1.0
        assert len(box) == 4


@needs_dataset
@pytest.mark.parametrize("max_hands", [1, 2, 4])
def test_detect_honours_max_hands_and_keeps_the_top_scorers(frames, max_hands):
    for path, img in frames:
        everything, _ = cvdet.detect(img, make_args(max_hands=10_000))
        limited, _ = cvdet.detect(img, make_args(max_hands=max_hands))
        assert len(limited) <= max_hands, f"{path}: {len(limited)} > {max_hands}"
        assert limited == everything[:max_hands], f"{path}: truncation dropped a top scorer"


@needs_dataset
def test_detect_results_are_sorted_by_score_descending(frames):
    saw_multiple = False
    for path, img in frames:
        detections, _ = cvdet.detect(img, make_args(max_hands=10_000))
        scores = [d[0] for d in detections]
        assert scores == sorted(scores, reverse=True), f"{path}: {scores}"
        saw_multiple |= len(scores) > 1
    assert saw_multiple, "no frame produced 2+ detections -- ordering was never exercised"


@needs_dataset
def test_detect_boxes_are_inside_the_frame_with_positive_extent(frames):
    for path, img in frames:
        height, width = img.shape[:2]
        detections, _ = cvdet.detect(img, make_args(max_hands=10_000))
        for score, (x, y, w, h), fingers, solidity in detections:
            assert w > 0 and h > 0, f"{path}: degenerate box {(x, y, w, h)}"
            assert 0 <= x and 0 <= y, f"{path}: negative origin {(x, y)}"
            assert x + w <= width, f"{path}: box {(x, y, w, h)} runs off width {width}"
            assert y + h <= height, f"{path}: box {(x, y, w, h)} runs off height {height}"


@needs_dataset
def test_detect_is_deterministic(frames):
    path, img = frames[0]
    first_dets, first_mask = cvdet.detect(img, make_args())
    second_dets, second_mask = cvdet.detect(img.copy(), make_args())
    assert first_dets == second_dets
    assert np.array_equal(first_mask, second_mask)


@needs_dataset
def test_detect_is_mirror_equivariant(frames):
    """Every stage is per-pixel or uses a symmetric kernel, so flipping the frame
    must flip the boxes and change nothing else. Catches any left/right bias
    sneaking into the mask or the contour scoring."""
    for path, img in frames:
        height, width = img.shape[:2]
        straight, _ = cvdet.detect(img, make_args(max_hands=10_000))
        flipped, _ = cvdet.detect(cv2.flip(img, 1), make_args(max_hands=10_000))
        as_mirrored = sorted((round(s, 6), width - b[0] - b[2], b[1], b[2], b[3])
                             for s, b, f, sol in straight)
        actual = sorted((round(s, 6), b[0], b[1], b[2], b[3]) for s, b, f, sol in flipped)
        assert as_mirrored == actual, f"{path}: mirrored detections differ"


@needs_dataset
def test_detect_finds_real_finger_valleys_on_real_frames(frames):
    """finger_defects fails closed (returns 0 on any cv2.error), so a broken
    convexity path would look like 'no hands' rather than an exception. At least
    one real frame must report a nonzero finger count."""
    total = sum(d[2] for path, img in frames
                for d in cvdet.detect(img, make_args(max_hands=10_000))[0])
    assert total > 0, "the finger heuristic reported 0 valleys across every frame"


@pytest.mark.parametrize("name,img", [
    ("black", np.zeros((360, 480, 3), np.uint8)),
    ("white", np.full((360, 480, 3), 255, np.uint8)),
])
def test_detect_finds_nothing_in_a_blank_frame(name, img):
    detections, mask = cvdet.detect(img, make_args())
    assert detections == [], name
    assert np.count_nonzero(mask) == 0, name


def test_detect_finds_nothing_when_the_whole_frame_is_skin():
    """A frame that is entirely skin-coloured has one blob above max_area, so the
    honest answer is 'no candidates', not one giant box."""
    detections, mask = cvdet.detect(solid((120, 150, 200), 360, 480), make_args())
    assert np.count_nonzero(mask) > 0, "precondition: the frame should mask as skin"
    assert detections == []


# --------------------------------------------------------------------------
# --image CLI
# --------------------------------------------------------------------------

@needs_dataset
def test_image_cli_exits_clean_and_writes_two_pngs(cli_run, sample_frame_path):
    proc, workdir = cli_run
    assert proc.returncode == 0, proc.stderr
    result = workdir / "cv_detect_result.png"
    mask = workdir / "cv_detect_mask.png"
    assert result.is_file() and mask.is_file(), sorted(p.name for p in workdir.iterdir())

    source = cv2.imread(str(sample_frame_path))
    written = cv2.imread(str(result))
    written_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    assert written is not None and written_mask is not None, "output PNGs are not decodable"
    assert written.shape == source.shape
    assert written_mask.shape == source.shape[:2]
    assert set(np.unique(written_mask).tolist()) <= {0, 255}


@needs_dataset
def test_image_cli_prints_one_line_per_reported_candidate(cli_run):
    proc, _ = cli_run
    header = re.search(r"^(\d+) candidate\(s\)", proc.stdout, re.M)
    assert header, proc.stdout
    claimed = int(header.group(1))
    printed = re.findall(r"^\s+score \d+\.\d+\s+box \(", proc.stdout, re.M)
    assert len(printed) == claimed, f"claimed {claimed}, printed {len(printed)}\n{proc.stdout}"
    assert claimed <= 4, "default --max-hands is 4"


@needs_dataset
def test_image_cli_max_hands_limits_the_report(repo, python_bin, sample_frame_path, tmp_path):
    proc = subprocess.run(
        [python_bin, str(repo / CLI), "--image", str(sample_frame_path), "--max-hands", "1"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert len(re.findall(r"^\s+score ", proc.stdout, re.M)) <= 1, proc.stdout


@needs_dataset
def test_image_cli_is_deterministic(repo, python_bin, sample_frame_path, tmp_path, cli_run):
    """Same image, fresh process, byte-identical PNGs."""
    proc, first_dir = cli_run
    again = subprocess.run(
        [python_bin, str(repo / CLI), "--image", str(sample_frame_path)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert again.returncode == 0, again.stderr
    for name in ("cv_detect_result.png", "cv_detect_mask.png"):
        assert (first_dir / name).read_bytes() == (tmp_path / name).read_bytes(), name
    assert again.stdout == proc.stdout


def test_image_cli_rejects_an_unreadable_image(repo, python_bin, tmp_path):
    proc = subprocess.run(
        [python_bin, str(repo / CLI), "--image", str(tmp_path / "nope.png")],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "could not read" in (proc.stdout + proc.stderr)
    assert not list(tmp_path.glob("*.png")), "wrote output for an image it could not read"


@needs_dataset
def test_image_mode_never_touches_the_camera_or_a_gui(monkeypatch, tmp_path, sample_frame_path):
    """--image must be completely headless. Run main() in-process with the camera
    and window entry points booby-trapped."""
    def forbidden(name):
        def stub(*a, **k):
            raise AssertionError(f"--image mode called cv2.{name}")
        return stub

    for name in ("VideoCapture", "imshow", "waitKey", "namedWindow", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, forbidden(name))
    monkeypatch.setattr(sys, "argv", [CLI, "--image", str(sample_frame_path)])
    monkeypatch.chdir(tmp_path)

    cvdet.main()

    assert (tmp_path / "cv_detect_result.png").is_file()
    assert (tmp_path / "cv_detect_mask.png").is_file()


# --------------------------------------------------------------------------
# Known defects -- these fail on purpose. See the findings report.
# --------------------------------------------------------------------------

@needs_dataset
def test_image_cli_does_not_claim_success_when_the_write_fails(repo, python_bin,
                                                               sample_frame_path, tmp_path):
    """FINDING: cv2.imwrite's return value is discarded, so a failed write is
    reported as 'wrote cv_detect_result.png and cv_detect_mask.png' with exit 0."""
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o555)
    try:
        proc = subprocess.run(
            [python_bin, str(repo / CLI), "--image", str(sample_frame_path)],
            cwd=str(readonly), capture_output=True, text=True, timeout=120,
        )
        wrote = sorted(p.name for p in readonly.glob("*.png"))
    finally:
        readonly.chmod(0o755)

    assert wrote == [], "precondition: the directory should have been unwritable"
    assert proc.returncode != 0 or "wrote " not in proc.stdout, (
        "claimed to have written files that do not exist\n"
        f"exit={proc.returncode}\nstdout={proc.stdout}"
    )


@needs_dataset
def test_show_mask_changes_the_output_in_image_mode(repo, python_bin, sample_frame_path,
                                                    tmp_path, cli_run):
    """FINDING: --show-mask is documented as 'show the skin mask side by side with
    the result' but main() only honours it on the camera branch, so in --image mode
    it is a silent no-op."""
    _, plain_dir = cli_run
    proc = subprocess.run(
        [python_bin, str(repo / CLI), "--image", str(sample_frame_path), "--show-mask"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    plain = hashlib.sha256((plain_dir / "cv_detect_result.png").read_bytes()).hexdigest()
    with_mask = hashlib.sha256((tmp_path / "cv_detect_result.png").read_bytes()).hexdigest()
    assert plain != with_mask, (
        "--show-mask had no effect at all in --image mode; cv_detect_result.png is "
        f"byte-identical (sha256 {plain[:16]})"
    )


@needs_dataset
def test_negative_max_hands_is_not_silently_accepted(frames):
    """FINDING: detect() slices with scored[:args.max_hands] and argparse puts no
    floor on --max-hands, so --max-hands -1 quietly discards the weakest
    detection instead of erroring (or returning nothing)."""
    path, img = frames[0]
    everything, _ = cvdet.detect(img, make_args(max_hands=10_000))
    if len(everything) < 2:
        pytest.skip(f"{path} produced {len(everything)} detections; need 2+")
    negative, _ = cvdet.detect(img, make_args(max_hands=-1))
    kept, total = len(negative), len(everything)
    assert kept in (0, total), (
        f"max_hands=-1 kept {kept} of {total} detections -- a negative count is "
        "neither rejected nor treated as 'none', it silently drops the weakest"
    )


# --------------------------------------------------------------------------
# draw -- appended after the findings block so the failing tests stay grouped
# --------------------------------------------------------------------------

def test_draw_survives_boxes_flush_against_every_edge():
    """The label plate is drawn at y - text_height - 6, which goes negative for any
    box touching the top of the frame -- and real detections do sit at y=0."""
    frame = np.zeros((120, 160, 3), np.uint8)
    detections = [
        (0.95, (0, 0, 40, 30), 4, 0.5),          # top-left corner, negative label plate
        (0.10, (120, 90, 40, 30), 0, 0.95),      # flush with the bottom-right corner
        (0.50, (159, 119, 1, 1), 1, 0.8),        # 1x1 box at the last pixel
    ]
    out = cvdet.draw(frame, detections)
    assert out is frame, "draw() should annotate in place, not return a new buffer"
    assert out.shape == (120, 160, 3)
    assert np.count_nonzero(out) > 0, "draw() painted nothing"


def test_draw_colours_strong_and_weak_detections_differently():
    """Scores at/above 0.4 are drawn in BOX_COLOR, below it in WEAK_COLOR -- the
    only cue that a candidate is a guess."""
    def painted(score):
        frame = np.zeros((120, 160, 3), np.uint8)
        cvdet.draw(frame, [(score, (20, 40, 60, 50), 2, 0.7)])
        return {tuple(int(c) for c in px) for px in frame.reshape(-1, 3)} - {(0, 0, 0)}

    assert cvdet.BOX_COLOR in painted(0.40)
    assert cvdet.WEAK_COLOR in painted(0.39)
    assert cvdet.BOX_COLOR not in painted(0.39)
