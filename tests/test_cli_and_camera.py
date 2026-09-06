"""The command-line surface of every entry point, plus the real camera.

Scope of this module:

  * every script answers --help and exits 0
  * bad input (bad --device, negative --threshold, missing file, missing camera)
    fails cleanly -- non-zero exit, a message a human can act on, no traceback
  * the documented still-image runbook actually produces the PNGs it promises
  * train_detector.py --eval-only reports arithmetically self-consistent metrics
  * the physical camera opens, warms up, and delivers real pixels
  * importing any script is side-effect free, so the suite itself cannot be
    hijacked by an entry point running main() at import time

Nothing here opens a GUI window: no subprocess started below ever reaches
cv2.imshow(), because every invocation exits during argument handling, during
model/asset loading, or inside the --image branch that returns before the camera
loop. The camera is opened only in-process, for a handful of frames, and is
always released.

Run with:   .venv/bin/python -m pytest tests/test_cli_and_camera.py -v
"""

import re
import subprocess
import sys

import pytest

from conftest import needs_checkpoint, needs_dataset, needs_mp_model

SCRIPTS = [
    "train_detector.py",
    "webcam_detect.py",
    "webcam_detect_mp.py",
    "webcam_detect_cv.py",
]

# a path that cannot exist, for the "missing file" arguments
NOPE = "/nonexistent-egohands-path/missing.file"

# an index no laptop has; used to exercise the "camera will not open" branch
BAD_CAMERA = "99"


# --------------------------------------------------------------------------- helpers


def run(python_bin, repo, *args, timeout=120):
    """Run one entry point from the repo root and capture everything it said."""
    proc = subprocess.run(
        [python_bin, *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    proc.output = proc.stdout + proc.stderr
    return proc


def assert_no_traceback(proc, context=""):
    assert "Traceback (most recent call last)" not in proc.output, (
        f"{context} died with a raw Python traceback instead of a message:\n{proc.output}"
    )


@pytest.fixture(scope="session")
def help_text(python_bin, repo):
    """--help output for all four entry points, collected once (imports are slow)."""
    collected = {}
    for script in SCRIPTS:
        collected[script] = run(python_bin, repo, script, "--help")
    return collected


@pytest.fixture
def repo_output(repo):
    """Claim a filename in the repo root, and delete it afterwards.

    Refuses to claim -- and therefore never deletes -- a file that already exists.
    """
    claimed = []

    def _claim(name):
        path = repo / name
        if path.exists():
            pytest.skip(f"{name} already exists in the repo root; refusing to touch it")
        claimed.append(path)
        return path

    yield _claim

    for path in claimed:
        if path.exists():
            path.unlink()


@pytest.fixture(scope="session")
def camera_capture():
    """Open camera 0 once, read a burst of frames, release, and hand back the burst.

    Session-scoped on purpose: opening the device costs ~1.7 s and macOS does not
    like several processes fighting over it. Returns a list of frames (numpy arrays
    or None), in read order, so tests can reason about warm-up.
    """
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            pytest.skip("no camera on this machine (or access not granted)")
        frames = []
        for _ in range(20):
            ok, frame = cap.read()
            frames.append(frame if ok else None)
    finally:
        cap.release()
    return frames


def blank(frame):
    """A frame that carries no image at all -- the macOS warm-up artefact."""
    return frame is None or not frame.any()


# --------------------------------------------------------------------------- --help


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_exits_zero(help_text, script):
    proc = help_text[script]
    assert proc.returncode == 0, f"{script} --help exited {proc.returncode}:\n{proc.output}"
    assert_no_traceback(proc, f"{script} --help")
    assert proc.stdout.startswith("usage:"), f"{script} --help printed no usage line"
    assert script in proc.stdout, f"{script} --help does not name itself in its usage line"


@pytest.mark.parametrize(
    "script,flags",
    [
        ("train_detector.py", ["--epochs", "--limit-videos", "--eval-only", "--checkpoint"]),
        ("webcam_detect.py", ["--camera", "--threshold", "--device", "--checkpoint"]),
        ("webcam_detect_mp.py", ["--image", "--max-hands", "--landmarks", "--model"]),
        ("webcam_detect_cv.py", ["--image", "--show-mask", "--min-area", "--max-area"]),
    ],
)
def test_help_lists_the_flags_the_docstring_advertises(help_text, script, flags):
    """Guards against doc/flag drift: every flag the module docstring shows off in
    its usage examples has to still be a real flag."""
    text = help_text[script].stdout
    missing = [flag for flag in flags if flag not in text]
    assert not missing, f"{script} --help never mentions {missing}"


# --------------------------------------------------------------------------- bad args


@pytest.mark.parametrize("script", ["train_detector.py", "webcam_detect.py"])
def test_bad_device_is_rejected_by_the_parser(python_bin, repo, script):
    proc = run(python_bin, repo, script, "--device", "gpu")
    assert proc.returncode == 2, f"expected argparse exit 2, got {proc.returncode}"
    assert_no_traceback(proc, f"{script} --device gpu")
    assert "--device" in proc.stderr and "invalid choice" in proc.stderr
    # the message has to say what IS allowed, or it is not actionable
    assert "cpu" in proc.stderr and "mps" in proc.stderr


@pytest.mark.parametrize("script", ["webcam_detect_mp.py", "webcam_detect_cv.py"])
def test_missing_image_fails_cleanly(python_bin, repo, script):
    proc = run(python_bin, repo, script, "--image", NOPE)
    assert proc.returncode != 0, f"{script} accepted a nonexistent --image"
    assert_no_traceback(proc, f"{script} --image {NOPE}")
    assert "could not read" in proc.output
    assert NOPE in proc.output, "the error does not say which path failed"


def test_missing_checkpoint_fails_cleanly_in_webcam_detect(python_bin, repo):
    proc = run(python_bin, repo, "webcam_detect.py", "--checkpoint", NOPE)
    assert proc.returncode != 0
    assert_no_traceback(proc, "webcam_detect.py --checkpoint <missing>")
    assert NOPE in proc.output
    assert "train_detector.py" in proc.output, "no hint about how to produce a checkpoint"


@needs_dataset
def test_missing_checkpoint_fails_cleanly_in_eval_only(python_bin, repo):
    proc = run(python_bin, repo, "train_detector.py", "--eval-only",
               "--checkpoint", NOPE, "--limit-videos", "2", "--workers", "0")
    assert proc.returncode != 0
    assert_no_traceback(proc, "train_detector.py --eval-only --checkpoint <missing>")
    assert NOPE in proc.output
    assert "train_detector.py" in proc.output


@needs_dataset
@needs_mp_model
def test_missing_mediapipe_model_fails_cleanly(python_bin, repo, sample_frame_path):
    proc = run(python_bin, repo, "webcam_detect_mp.py",
               "--model", NOPE, "--image", str(sample_frame_path))
    assert proc.returncode != 0
    assert_no_traceback(proc, "webcam_detect_mp.py --model <missing>")
    assert NOPE in proc.output
    assert "curl" in proc.output, "no download instructions for the missing model"


@pytest.mark.parametrize("script", ["webcam_detect_cv.py", "webcam_detect.py"])
def test_unopenable_camera_index_fails_cleanly(python_bin, repo, script):
    """A camera index that does not exist must stop the program with a message,
    not with an OpenCV assertion or an empty-frame loop."""
    proc = run(python_bin, repo, script, "--camera", BAD_CAMERA)
    assert proc.returncode != 0, f"{script} kept going with camera {BAD_CAMERA}"
    assert_no_traceback(proc, f"{script} --camera {BAD_CAMERA}")
    assert "could not open camera" in proc.output
    assert BAD_CAMERA in proc.output, "the error does not say which index failed"


@pytest.mark.parametrize(
    "script,flag,extra",
    [
        ("webcam_detect.py", "--threshold", []),
        ("train_detector.py", "--score-threshold",
         ["--eval-only", "--limit-videos", "2", "--workers", "0"]),
    ],
)
def test_negative_confidence_threshold_is_rejected(python_bin, repo, script, flag, extra):
    """A confidence threshold is a probability; -0.5 is not one.

    The run is paired with a checkpoint that cannot exist. If the value were
    validated, the parser would say so and never reach the checkpoint; if it is
    not validated, the run gets all the way to the missing-checkpoint error, which
    is exactly how this test detects the gap.
    """
    proc = run(python_bin, repo, script, flag, "-0.5", "--checkpoint", NOPE, *extra)
    assert proc.returncode != 0
    assert_no_traceback(proc, f"{script} {flag} -0.5")
    assert "threshold" in proc.output.lower(), (
        f"{script} silently accepted {flag} -0.5 and failed later on something else "
        f"instead:\n{proc.output}"
    )


@needs_dataset
def test_limit_videos_zero_is_not_silently_the_whole_dataset(python_bin, repo):
    """--limit-videos 0 must not mean 'all 48 videos'.

    The flag exists for smoke tests; a user who types 0 is asking for less work,
    and must not be handed a full-length run without a word.
    """
    proc = run(python_bin, repo, "train_detector.py", "--eval-only", "--limit-videos", "0",
               "--checkpoint", NOPE, "--workers", "0")
    assert_no_traceback(proc, "train_detector.py --limit-videos 0")
    if proc.returncode == 2:
        return  # rejected by the parser, which is a fine answer
    match = re.search(r"^videos\s+(\d+)", proc.output, re.MULTILINE)
    assert match, f"could not find the 'videos N' line in:\n{proc.output}"
    selected = int(match.group(1))
    assert selected != 48, (
        "--limit-videos 0 was ignored and the run selected all 48 videos "
        "(a full 17-minute training run) instead of complaining"
    )


@needs_dataset
def test_zero_epochs_does_not_claim_a_checkpoint_it_never_wrote(python_bin, repo, tmp_path):
    """If the run exits 0 and tells the user 'now run webcam_detect.py --checkpoint X',
    then X has to be on disk."""
    checkpoint = tmp_path / "smoke.pth"
    proc = run(python_bin, repo, "train_detector.py", "--epochs", "0",
               "--limit-videos", "2", "--workers", "0", "--checkpoint", str(checkpoint))
    assert_no_traceback(proc, "train_detector.py --epochs 0")
    if proc.returncode != 0:
        return  # rejecting --epochs 0 outright is also acceptable
    assert checkpoint.is_file(), (
        "--epochs 0 exited 0 and printed "
        f"{[l for l in proc.stdout.splitlines() if 'now run' in l]}, "
        f"but {checkpoint} was never written"
    )


# ------------------------------------------------------------------- documented runbook


@needs_dataset
def test_cv_still_image_runbook(python_bin, repo, sample_frame_path, repo_output):
    """python webcam_detect_cv.py --image <frame>  ->  two PNGs in the repo root."""
    cv2 = pytest.importorskip("cv2")
    result = repo_output("cv_detect_result.png")
    mask = repo_output("cv_detect_mask.png")

    proc = run(python_bin, repo, "webcam_detect_cv.py", "--image", str(sample_frame_path))
    assert proc.returncode == 0, proc.output
    assert_no_traceback(proc, "webcam_detect_cv.py --image")
    assert "candidate(s) in" in proc.stdout
    assert result.is_file() and mask.is_file(), f"promised PNGs missing:\n{proc.output}"

    source = cv2.imread(str(sample_frame_path))
    drawn = cv2.imread(str(result))
    assert drawn is not None and drawn.shape == source.shape, "annotated PNG is not the input size"
    written_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    assert written_mask.shape == source.shape[:2]
    assert set(written_mask.flatten().tolist()) <= {0, 255}, "skin mask is not binary"

    # printed boxes are (x, y, w, h) here -- note the sibling tool prints (x1,y1,x2,y2)
    height, width = source.shape[:2]
    boxes = re.findall(r"box \((\d+), (\d+), (\d+), (\d+)\)", proc.stdout)
    assert boxes, f"no boxes printed for a real hands frame:\n{proc.stdout}"
    for x, y, w, h in ((int(a), int(b), int(c), int(d)) for a, b, c, d in boxes):
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= width, f"box runs off the frame: {(x, y, w, h)}"
        assert 0 <= y and y + h <= height, f"box runs off the frame: {(x, y, w, h)}"


@needs_dataset
@needs_mp_model
def test_mp_still_image_runbook(python_bin, repo, sample_frame_path, repo_output):
    """python webcam_detect_mp.py --image <frame>  ->  mp_detect_result.png."""
    cv2 = pytest.importorskip("cv2")
    result = repo_output("mp_detect_result.png")

    proc = run(python_bin, repo, "webcam_detect_mp.py", "--image", str(sample_frame_path),
               "--landmarks", "--max-hands", "2")
    assert proc.returncode == 0, proc.output
    assert_no_traceback(proc, "webcam_detect_mp.py --image")
    assert "hand(s) in" in proc.stdout
    assert result.is_file(), f"mp_detect_result.png was promised but not written:\n{proc.output}"

    source = cv2.imread(str(sample_frame_path))
    drawn = cv2.imread(str(result))
    assert drawn is not None and drawn.shape == source.shape
    assert (drawn != source).any(), "the 'annotated' PNG is byte-identical to the input"

    # a labelled EgoHands frame contains hands, so a hand detector must find one
    count = int(re.search(r"(\d+) hand\(s\) in", proc.stdout).group(1))
    assert count >= 1, f"MediaPipe found no hands in a hand-labelled frame:\n{proc.stdout}"

    height, width = source.shape[:2]
    boxes = re.findall(r"box \((\d+), (\d+), (\d+), (\d+)\)", proc.stdout)
    assert len(boxes) == count
    for x1, y1, x2, y2 in ((int(a), int(b), int(c), int(d)) for a, b, c, d in boxes):
        assert x1 < x2 and y1 < y2, f"degenerate box {(x1, y1, x2, y2)}"
        assert 0 <= x1 and x2 <= width and 0 <= y1 and y2 <= height, "box outside the frame"


@needs_dataset
def test_image_mode_never_touches_the_camera(python_bin, repo, sample_frame_path, repo_output):
    """--image is the headless path: it must succeed even when --camera is nonsense."""
    repo_output("cv_detect_result.png")
    repo_output("cv_detect_mask.png")
    proc = run(python_bin, repo, "webcam_detect_cv.py",
               "--image", str(sample_frame_path), "--camera", BAD_CAMERA)
    assert proc.returncode == 0, (
        f"--image opened the camera anyway:\n{proc.output}"
    )
    assert "could not open camera" not in proc.output


@needs_mp_model
@needs_dataset
def test_bad_argument_is_not_blamed_on_the_model_file(python_bin, repo, sample_frame_path):
    """--confidence 5.0 is a bad argument, not a missing model.

    The model file is present and valid here, so the failure message must not send
    the user off to re-download it.
    """
    proc = run(python_bin, repo, "webcam_detect_mp.py",
               "--image", str(sample_frame_path), "--confidence", "5.0")
    assert proc.returncode != 0, "a confidence of 5.0 was accepted"
    assert_no_traceback(proc, "webcam_detect_mp.py --confidence 5.0")
    assert "curl" not in proc.output, (
        "a bad --confidence is reported as a broken/missing model file and the user "
        f"is told to re-download a model that is perfectly fine:\n{proc.output}"
    )


def test_readme_only_tells_you_to_run_scripts_that_exist(repo):
    """Every `something.py` the README names has to be a file you can run."""
    text = (repo / "README.md").read_text()
    named = sorted(set(re.findall(r"[A-Za-z0-9_]+\.py", text)))
    missing = [name for name in named if not ((repo / name).exists() or (repo / 'tests' / name).exists())]
    assert not missing, (
        f"README.md points the reader at scripts that are not in the repo: {missing} "
        f"(present names use snake_case: get_meta_by.py, visualize_dataset.py, ...)"
    )


# --------------------------------------------------------------------------- eval-only


@needs_dataset
@needs_checkpoint
def test_eval_only_reports_self_consistent_metrics(python_bin, repo):
    """--eval-only must print precision/recall/F1 that agree with its own tp/fp/fn."""
    proc = run(python_bin, repo, "train_detector.py", "--eval-only",
               "--limit-videos", "4", "--workers", "0", timeout=240)
    assert proc.returncode == 0, proc.output
    assert_no_traceback(proc, "train_detector.py --eval-only")

    def number(label, cast=float):
        match = re.search(rf"^{label}\s+([-\d.]+)", proc.stdout, re.MULTILINE)
        assert match, f"--eval-only never printed '{label}':\n{proc.stdout}"
        return cast(match.group(1))

    tp = number(r"true positives", int)
    fp = number(r"false positives", int)
    fn = number(r"false negatives", int)
    precision = number(r"precision")
    recall = number(r"recall")
    f1 = number(r"F1")

    assert tp > 0, "a trained checkpoint scored zero true positives on its own val split"
    assert min(tp, fp, fn) >= 0
    for name, value in (("precision", precision), ("recall", recall), ("F1", f1)):
        assert 0.0 <= value <= 1.0, f"{name} = {value} is not a rate"

    assert precision == pytest.approx(tp / (tp + fp), abs=1e-3), "precision != tp/(tp+fp)"
    assert recall == pytest.approx(tp / (tp + fn), abs=1e-3), "recall != tp/(tp+fn)"
    harmonic = 2 * precision * recall / (precision + recall)
    assert f1 == pytest.approx(harmonic, abs=2e-3), "F1 is not the harmonic mean of P and R"


# --------------------------------------------------------------------------- the camera


def test_camera_opens_and_delivers_real_pixels(camera_capture):
    """The camera hands back real content -- eventually.

    macOS returns a couple of all-black frames with ok=True before auto-exposure
    settles, so asserting on frame 0 would report a false 'permission denied'.
    This asserts on the burst instead.
    """
    assert camera_capture, "no frames were read at all"
    assert camera_capture[0] is not None, "the very first read failed outright"

    warmup = next((i for i, f in enumerate(camera_capture) if not blank(f)), None)
    assert warmup is not None, (
        f"all {len(camera_capture)} frames were empty -- camera access is probably "
        "denied in System Settings -> Privacy & Security -> Camera"
    )
    assert warmup <= 10, f"took {warmup} frames before any content appeared"

    live = camera_capture[warmup]
    assert live.ndim == 3 and live.shape[2] == 3, f"unexpected frame shape {live.shape}"
    assert live.max() > 0
    # once warm, the feed must stay warm
    tail = camera_capture[warmup:]
    assert all(not blank(f) for f in tail), "the feed went blank again after warming up"


def test_camera_reports_the_resolution_it_actually_delivers():
    """open_camera() prints a resolution; frames must actually arrive at that size,
    because the detector is fed the raw frame and its aspect ratio matters."""
    cv2 = pytest.importorskip("cv2")
    from webcam_detect import open_camera

    cap = open_camera(0)
    try:
        assert cap.isOpened()
        reported = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        shapes = []
        for _ in range(6):
            ok, frame = cap.read()
            if ok and frame is not None:
                shapes.append((frame.shape[1], frame.shape[0]))
    finally:
        cap.release()

    assert shapes, "camera opened but produced no frames"
    assert set(shapes) == {reported}, (
        f"open_camera() announced {reported} but delivered {set(shapes)}"
    )


def test_camera_frame_flows_through_the_classical_detector(camera_capture, monkeypatch):
    """A live frame must survive the whole webcam_detect_cv pipeline, with every
    box inside the frame -- no GUI involved."""
    import webcam_detect_cv as cv_detect

    frame = next((f for f in camera_capture if not blank(f)), None)
    if frame is None:
        pytest.skip("camera never produced a non-blank frame")

    monkeypatch.setattr(sys, "argv", ["webcam_detect_cv.py"])
    args = cv_detect.parse_args()

    detections, mask = cv_detect.detect(frame, args)
    height, width = frame.shape[:2]
    assert mask.shape == (height, width)
    assert len(detections) <= args.max_hands
    for score, (x, y, w, h), fingers, solidity in detections:
        assert 0.0 <= score <= 1.0, f"score {score} out of range"
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= width and 0 <= y and y + h <= height
        assert fingers >= 0
        assert 0.0 < solidity <= 1.0


# --------------------------------------------------------------------- import hygiene


@pytest.mark.parametrize("script", SCRIPTS)
def test_main_guard_is_present(repo, script):
    source = (repo / script).read_text()
    assert 'if __name__ == "__main__":' in source, f"{script} has no main guard"
    tail = [line for line in source.splitlines() if line.strip()][-2:]
    assert tail[0].strip() == 'if __name__ == "__main__":' and tail[1].strip() == "main()", (
        f"{script} ends with {tail}, not a plain main() guard"
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_import_has_no_side_effects(python_bin, repo, script):
    """Importing an entry point must not run it.

    If any of these executed main() at import time, this very test suite would open
    a camera or a GUI window the moment it imported the module.
    """
    module = script[:-3]
    proc = run(
        python_bin, repo, "-c",
        f"import {module} as m; print('IMPORTED', callable(getattr(m, 'main', None)))",
        timeout=90,
    )
    assert proc.returncode == 0, f"importing {module} failed:\n{proc.output}"
    assert "IMPORTED True" in proc.stdout, f"{module} exposes no main():\n{proc.output}"
    for marker in ("usage:", "camera 0 open", "could not open camera", "device            "):
        assert marker not in proc.output, f"importing {module} ran main() ({marker!r} printed)"
