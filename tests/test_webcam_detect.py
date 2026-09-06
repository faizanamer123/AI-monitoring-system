"""Behavioural tests for webcam_detect.py -- the trained-model inference path.

The camera is deliberately never opened. Everything webcam_detect.py does between
`cap.read()` and `cv2.imshow()` is exercised on real _LABELLED_SAMPLES stills:

    load_model  -> checkpoint handling, eval mode, reported metrics
    detect      -> box geometry, accuracy, thresholding, determinism
    draw        -> rendering contract

The geometry test is the important one. detection_dataset.py trains on frames
squashed to 320x320, but detect() feeds the *full* 1280x720 frame and trusts SSD's
internal GeneralizedRCNNTransform (fixed_size=(320, 320)) to squash it the same way
and map the boxes back to original pixels. If that ever stops being true the demo
draws boxes in the top-left eighth of the screen and nothing else notices.

Run:  .venv/bin/python -m pytest tests/test_webcam_detect.py -v
"""

import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
from torchvision.ops import box_iou

from conftest import needs_checkpoint, needs_dataset

import webcam_detect as W
from detection_dataset import FRAME_HEIGHT, FRAME_WIDTH, HAND_LABEL, IMAGE_SIZE, split_videos
from get_bounding_boxes import get_bounding_boxes
from get_frame_path import get_frame_path
from train_detector import DEFAULT_CHECKPOINT

# Everything runs on CPU: MPS was checked to agree with CPU to ~1e-4 px, and CPU is
# the only backend that is guaranteed present wherever this suite runs.
DEVICE = torch.device("cpu")

# Two frames from each of the 12 held-out videos. Frame 0 and frame 50 are ~45s
# apart in a 90s clip, so they are not near-duplicates of each other.
VAL_FRAME_INDICES = (0, 50)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def model():
    """The real fine-tuned detector, loaded once."""
    return W.load_model(DEFAULT_CHECKPOINT, DEVICE)


@pytest.fixture(scope="session")
def val_frame_specs(videos):
    """(video_row, frame_index, path) for two frames of every held-out video."""
    _train_idx, val_idx = split_videos(videos)
    specs = []
    for video_num in val_idx:
        video = videos.iloc[video_num]
        for frame_num in VAL_FRAME_INDICES:
            specs.append((video, frame_num, get_frame_path(video, frame_num)))
    return specs


@pytest.fixture(scope="session")
def val_predictions(model, val_frame_specs):
    """detect() run once over the held-out stills at the default 0.5 threshold.

    Returns [(shape, boxes, scores, ground_truth_xyxy), ...]. Shared because a
    single 1280x720 forward pass costs ~0.16 s and four tests need the same one.
    """
    results = []
    for video, frame_num, path in val_frame_specs:
        frame = cv2.imread(path)
        assert frame is not None, f"could not read {path}"
        boxes, scores = W.detect(model, frame, DEVICE, 0.5)
        results.append((frame.shape, boxes, scores, ground_truth_xyxy(video, frame_num)))
    return results


@pytest.fixture(scope="session")
def one_frame(val_frame_specs):
    frame = cv2.imread(val_frame_specs[0][2])
    assert frame is not None
    return frame


# ---------------------------------------------------------------- helpers


def ground_truth_xyxy(video, frame_num):
    """The same [x, y, w, h] -> xyxy conversion detection_dataset trained against."""
    boxes = []
    for x, y, w, h in get_bounding_boxes(video, frame_num):
        if w <= 1 or h <= 1:      # absent hands come back as a row of zeros
            continue
        boxes.append([x, y, x + w - 1, y + h - 1])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def greedy_match(boxes, scores, truth, iou_threshold=0.5):
    """train_detector.evaluate's matching rule: tp, fp, fn for one frame."""
    order = np.argsort(-scores)
    claimed, true_pos, false_pos = set(), 0, 0
    for i in order:
        if len(truth) == 0:
            false_pos += 1
            continue
        ious = box_iou(torch.from_numpy(boxes[i][None]), torch.from_numpy(truth))[0]
        best = int(ious.argmax())
        if ious[best] >= iou_threshold and best not in claimed:
            claimed.add(best)
            true_pos += 1
        else:
            false_pos += 1
    return true_pos, false_pos, len(truth) - len(claimed)


# ---------------------------------------------------------------- load_model


@needs_checkpoint
def test_load_model_returns_an_eval_mode_model_and_reports_its_metrics(capsys):
    model = W.load_model(DEFAULT_CHECKPOINT, DEVICE)

    assert not model.training, "detector must come back in eval mode"
    assert not any(m.training for m in model.modules()), "a submodule is still in train mode"
    assert next(model.parameters()).device.type == DEVICE.type

    printed = capsys.readouterr().out
    assert DEFAULT_CHECKPOINT in printed, f"load_model said nothing about the checkpoint: {printed!r}"
    assert "F1" in printed, f"load_model did not report the checkpoint's metrics: {printed!r}"
    # the reported F1 must be the number actually stored in the checkpoint
    state = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    assert f"{state['metrics']['f1']:.3f}" in printed


@needs_checkpoint
def test_model_only_ever_emits_the_hand_class(model, one_frame):
    """draw() labels every box "hand" without consulting output["labels"].

    That is only safe while the head really is background + hand.
    """
    assert model.head.classification_head.num_columns == 2
    rgb = cv2.cvtColor(one_frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0)
    with torch.no_grad():
        output = model([tensor])[0]
    assert len(output["labels"]) > 0
    assert set(output["labels"].tolist()) <= {HAND_LABEL}


def test_load_model_on_a_missing_checkpoint_exits_cleanly(tmp_path):
    missing = tmp_path / "not_trained_yet.pth"
    with pytest.raises(SystemExit) as exc:
        W.load_model(str(missing), DEVICE)

    message = str(exc.value)
    assert str(missing) in message, "the error must name the path that was missing"
    assert "train_detector" in message, "the error must say how to produce a checkpoint"
    assert not isinstance(exc.value.__cause__, FileNotFoundError), "raw traceback leaked as __cause__"


@pytest.mark.parametrize("kind", ["directory", "not_a_checkpoint"])
def test_load_model_on_an_unreadable_checkpoint_exits_cleanly(tmp_path, kind):
    """A --checkpoint that exists but cannot be loaded should fail like a missing one.

    `--checkpoint checkpoints` (shell tab-completion drops the filename) and a
    truncated/half-copied .pth are both ordinary user mistakes; both currently
    escape as a raw IsADirectoryError / UnpicklingError traceback.
    """
    if kind == "directory":
        target = tmp_path / "checkpoints"
        target.mkdir()
    else:
        target = tmp_path / "hand_detector.pth"
        target.write_bytes(b"# this is not a torch checkpoint\n")

    with pytest.raises(SystemExit):
        W.load_model(str(target), DEVICE)


@needs_checkpoint
def test_load_model_does_not_need_to_fetch_the_coco_weights(monkeypatch):
    """Loading a local checkpoint should not depend on the network.

    build_model() instantiates ssdlite320 with COCO_V1 weights, and load_model then
    overwrites all of them from the checkpoint (0 missing / 0 unexpected keys), so
    the 14 MB fetch is pure waste -- and on a machine that has the checkpoint but no
    torchvision cache and no internet, the demo cannot start at all.
    """
    import torchvision.models._api as tv_api

    fetched = []
    real = tv_api.load_state_dict_from_url

    def spy(url, *args, **kwargs):
        fetched.append(url)
        return real(url, *args, **kwargs)

    monkeypatch.setattr(tv_api, "load_state_dict_from_url", spy)
    W.load_model(DEFAULT_CHECKPOINT, DEVICE)
    assert fetched == [], f"load_model fetched pretrained weights it immediately discards: {fetched}"


@needs_checkpoint
def test_cli_refuses_a_missing_checkpoint_before_touching_the_camera(python_bin, repo):
    """The CLI must exit non-zero with the friendly message and no traceback."""
    result = subprocess.run(
        [python_bin, "webcam_detect.py", "--checkpoint", "no/such/checkpoint.pth", "--device", "cpu"],
        cwd=str(repo), capture_output=True, text=True, timeout=120,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "Traceback" not in combined, f"raw traceback shown to the user:\n{combined}"
    assert "no checkpoint at no/such/checkpoint.pth" in combined
    assert "open at" not in combined, "open_camera ran before the checkpoint was validated"


# ---------------------------------------------------------------- detect: geometry


@needs_checkpoint
@needs_dataset
def test_detect_returns_original_frame_pixel_coordinates(val_predictions):
    """Boxes must live in 1280x720 space, not the model's internal 320x320 space."""
    total = 0
    beyond_320 = 0
    for shape, boxes, scores, _truth in val_predictions:
        height, width = shape[:2]
        assert (height, width) == (FRAME_HEIGHT, FRAME_WIDTH)
        assert boxes.shape[1:] == (4,), f"boxes must be (N, 4), got {boxes.shape}"
        assert scores.shape == (len(boxes),)
        assert boxes.dtype == np.float32 and scores.dtype == np.float32
        for x1, y1, x2, y2 in boxes:
            assert 0 <= x1 < x2 <= width, f"box x range {x1}..{x2} outside 0..{width}"
            assert 0 <= y1 < y2 <= height, f"box y range {y1}..{y2} outside 0..{height}"
        total += len(boxes)
        beyond_320 += int(((boxes[:, 2] > IMAGE_SIZE) | (boxes[:, 3] > IMAGE_SIZE)).sum())

    assert total > 0, "no detections at all on held-out frames -- nothing was verified"
    # If the boxes were left in 320x320 space every coordinate would be <= 320.
    assert beyond_320 > 0, (
        f"all {total} boxes fit inside {IMAGE_SIZE}x{IMAGE_SIZE}: boxes look like they were "
        "never mapped back to the full frame"
    )


@needs_checkpoint
@needs_dataset
def test_detected_boxes_are_plausibly_hand_sized(val_predictions):
    """Ground-truth hands cover 0.18%-13.5% of the frame; predictions must be in that ballpark."""
    frame_area = FRAME_WIDTH * FRAME_HEIGHT
    for _shape, boxes, _scores, _truth in val_predictions:
        for x1, y1, x2, y2 in boxes:
            fraction = (x2 - x1) * (y2 - y1) / frame_area
            assert 0.0005 <= fraction <= 0.35, f"box {x1, y1, x2, y2} covers {fraction:.3%} of the frame"
            assert (x2 - x1) <= 0.8 * FRAME_WIDTH
            assert (y2 - y1) <= 0.9 * FRAME_HEIGHT


@needs_checkpoint
@needs_dataset
def test_full_resolution_inference_matches_the_training_geometry(model, val_frame_specs):
    """Feeding the raw frame must agree with feeding the 320x320 squash detection_dataset builds.

    Same frame, two routes into the model: straight in at 1280x720 (what detect does)
    versus pre-squashed to 320x320 with the boxes scaled back by hand (what training
    saw). If SSD's transform were ever configured with a different resize -- aspect-
    preserving with padding, say -- these two would stop lining up.
    """
    worst = 1.0
    compared = 0
    for video, frame_num, path in val_frame_specs[:12]:
        frame = cv2.imread(path)
        full_boxes, _ = W.detect(model, frame, DEVICE, 0.5)

        squashed = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        square_boxes, _ = W.detect(model, squashed, DEVICE, 0.5)
        square_boxes = square_boxes.copy()
        if len(square_boxes):
            square_boxes[:, [0, 2]] *= FRAME_WIDTH / IMAGE_SIZE
            square_boxes[:, [1, 3]] *= FRAME_HEIGHT / IMAGE_SIZE

        assert len(full_boxes) == len(square_boxes), (
            f"{path} frame {frame_num}: {len(full_boxes)} boxes at full res vs "
            f"{len(square_boxes)} through the training geometry"
        )
        if len(full_boxes) == 0:
            continue
        ious = box_iou(torch.from_numpy(full_boxes), torch.from_numpy(square_boxes))
        worst = min(worst, float(ious.max(dim=1).values.min()))
        compared += len(full_boxes)

    assert compared > 0
    assert worst >= 0.7, f"full-res and training-geometry boxes disagree (worst IoU {worst:.3f})"


@needs_checkpoint
@needs_dataset
def test_detect_maps_boxes_back_for_a_non_native_camera_resolution(model, one_frame):
    """open_camera only *asks* for 1280x720; a camera may hand back 640x480 anyway."""
    small = cv2.resize(one_frame, (640, 480))
    boxes, scores = W.detect(model, small, DEVICE, 0.5)
    assert len(boxes) > 0, "the same frame at 640x480 produced no detections at all"
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 640, f"x range {x1}..{x2} outside a 640-wide frame"
        assert 0 <= y1 < y2 <= 480, f"y range {y1}..{y2} outside a 480-tall frame"


# ---------------------------------------------------------------- detect: accuracy


@needs_checkpoint
@needs_dataset
def test_detector_actually_finds_hands_on_held_out_frames(val_predictions):
    """Accuracy sanity against ground truth, at the same IoU>=0.5 rule train_detector uses.

    The checkpoint claims val F1 0.848 over the whole val split; this 24-frame
    subsample should land near it and must not be near zero. Bounds are set well
    below what was measured (P 0.95 / R 0.68) so only a real regression trips them.
    """
    true_pos = false_pos = false_neg = 0
    frames_with_a_hit = 0
    for _shape, boxes, scores, truth in val_predictions:
        tp, fp, fn = greedy_match(boxes, scores, truth)
        true_pos, false_pos, false_neg = true_pos + tp, false_pos + fp, false_neg + fn
        frames_with_a_hit += int(tp > 0)

    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    report = (f"tp={true_pos} fp={false_pos} fn={false_neg} "
              f"P={precision:.3f} R={recall:.3f} on {len(val_predictions)} frames")

    assert true_pos > 0, f"not one correct detection on held-out data: {report}"
    assert recall >= 0.45, report
    assert precision >= 0.70, report
    assert frames_with_a_hit >= 0.7 * len(val_predictions), report


# ---------------------------------------------------------------- detect: contract


@needs_checkpoint
@needs_dataset
def test_raising_the_threshold_never_adds_boxes(model, one_frame):
    ladder = [0.05, 0.1, 0.25, 0.5, 0.75, 0.95]
    results = [W.detect(model, one_frame, DEVICE, t) for t in ladder]

    for (low_t, (low_boxes, low_scores)), (high_t, (high_boxes, high_scores)) in zip(
        zip(ladder, results), zip(ladder[1:], results[1:])
    ):
        assert len(high_boxes) <= len(low_boxes), (
            f"threshold {high_t} returned {len(high_boxes)} boxes, "
            f"more than {len(low_boxes)} at {low_t}"
        )
        # the stricter result must be a subset, not merely a smaller count
        for box in high_boxes:
            assert any(np.array_equal(box, other) for other in low_boxes), (
                f"box {box} appears at threshold {high_t} but not at {low_t}"
            )
        assert (high_scores >= high_t).all(), "a score below the threshold was returned"

    assert len(results[0][0]) > 0, "a 0.05 threshold found nothing -- the ladder proves nothing"


@needs_checkpoint
@pytest.mark.parametrize("fill", [0, 127, 255])
def test_uniform_frames_do_not_hallucinate_hands(model, fill):
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), fill, dtype=np.uint8)

    boxes, scores = W.detect(model, frame, DEVICE, 0.5)
    assert boxes.shape == (0, 4), f"found {len(boxes)} hands in a uniform {fill} frame: {scores}"

    loose_boxes, _ = W.detect(model, frame, DEVICE, 0.2)
    assert len(loose_boxes) <= 2, f"{len(loose_boxes)} boxes on a uniform frame even at 0.2"


@needs_checkpoint
@needs_dataset
def test_detect_is_deterministic_and_leaves_the_frame_alone(model, one_frame):
    original = one_frame.copy()

    first_boxes, first_scores = W.detect(model, one_frame, DEVICE, 0.5)
    second_boxes, second_scores = W.detect(model, one_frame, DEVICE, 0.5)
    third_boxes, third_scores = W.detect(model, one_frame.copy(), DEVICE, 0.5)

    assert len(first_boxes) > 0
    assert np.array_equal(first_boxes, second_boxes), "same frame twice gave different boxes"
    assert np.array_equal(first_scores, second_scores), "same frame twice gave different scores"
    assert np.array_equal(first_boxes, third_boxes), "a copy of the frame gave different boxes"
    assert np.array_equal(one_frame, original), "detect mutated the caller's frame"


# ---------------------------------------------------------------- draw


@needs_checkpoint
@needs_dataset
def test_draw_renders_without_touching_the_boxes(model, one_frame):
    boxes, scores = W.detect(model, one_frame, DEVICE, 0.5)
    assert len(boxes) > 0

    boxes_before, scores_before = boxes.copy(), scores.copy()
    canvas = one_frame.copy()
    out = W.draw(canvas, boxes, scores, 24.0)

    assert isinstance(out, np.ndarray)
    assert out.shape == one_frame.shape and out.dtype == one_frame.dtype
    assert np.array_equal(boxes, boxes_before), "draw mutated the box coordinates"
    assert np.array_equal(scores, scores_before), "draw mutated the scores"
    assert not np.array_equal(out, one_frame), "draw returned an untouched frame"

    # the box outline really landed where detect said it was
    x1, y1, x2, y2 = boxes[0].astype(int)
    edge = out[y1:y2 + 1, x1]
    assert (np.all(edge == W.BOX_COLOR, axis=-1)).any(), "no box edge drawn at x1"


def test_draw_survives_empty_and_edge_boxes():
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    empty = W.draw(frame.copy(), np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.0)
    assert empty.shape == frame.shape

    # a hand entering from the top: the label strip is drawn at y1 - text_h - 6 < 0,
    # and a hand running off the right edge pushes the strip past the frame width
    awkward = np.array([[4.0, 1.0, 90.0, 70.0],
                        [FRAME_WIDTH - 30.0, FRAME_HEIGHT - 20.0, FRAME_WIDTH - 1.0, FRAME_HEIGHT - 1.0]],
                       dtype=np.float32)
    out = W.draw(frame.copy(), awkward, np.array([0.91, 0.55], np.float32), 12.5)
    assert out.shape == frame.shape


