"""Behavioural tests for train_detector.py.

Four things have to hold for the detection pipeline to mean anything:

  1. build_model() really is a 2-class SSDlite that kept its COCO backbone.
  2. A training step actually moves the weights (a loop that runs but does not
     learn is the most expensive kind of silent bug).
  3. evaluate() counts TP/FP/FN the way a human counting by hand would, and in
     particular refuses to let two predictions claim the same ground-truth box.
  4. A saved checkpoint reloads into a fresh build_model() and produces bit-for-bit
     the same predictions -- this is the contract webcam_detect.py relies on.

Run with:   .venv/bin/python -m pytest tests/test_train_detector.py -v
"""

import re
import subprocess

import pytest
import torch

from conftest import CHECKPOINT, needs_checkpoint, needs_dataset

from detection_dataset import IMAGE_SIZE, NUM_CLASSES, EgoHandsDetection, collate_fn
from train_detector import build_model, evaluate, pick_device, train_one_epoch

CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _StubDetector:
    """Stands in for a trained model so evaluate() can be fed known predictions.

    evaluate() only ever calls .eval() and then the model on a list of images, so
    this is the whole surface area it touches.
    """

    def __init__(self, batched_outputs):
        self.batched_outputs = list(batched_outputs)
        self.eval_called = False
        self.calls = 0

    def eval(self):
        self.eval_called = True

    def __call__(self, images):
        outputs = self.batched_outputs[self.calls]
        self.calls += 1
        assert len(outputs) == len(images), "stub wired up wrong"
        return outputs


def _pred(boxes, scores):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32).reshape(-1),
        "labels": torch.ones(len(scores), dtype=torch.int64),
    }


def _truth(boxes):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.ones(len(boxes), dtype=torch.int64),
    }


def run_evaluate(predictions, truths, score_threshold=0.5, iou_threshold=0.5):
    """Score one hand-built batch: predictions[i] is judged against truths[i]."""
    assert len(predictions) == len(truths)
    images = [torch.zeros(3, 8, 8) for _ in predictions]
    loader = [(images, truths)]
    model = _StubDetector([predictions])
    metrics = evaluate(model, loader, CPU, score_threshold, iou_threshold)
    assert model.eval_called, "evaluate() must put the model in eval mode"
    return metrics


@pytest.fixture(scope="module")
def fresh_model():
    """One un-mutated build_model() shared by the read-only inspection tests."""
    return build_model()


@pytest.fixture(scope="module")
def coco_reference():
    """The stock COCO-pretrained SSDlite, to diff build_model() against."""
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
    return ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    )


@pytest.fixture(scope="module")
def real_batch(videos):
    """Two real EgoHands frames, collated exactly as the DataLoader would."""
    ds = EgoHandsDetection(videos, [0], train=False)
    return collate_fn([ds[0], ds[1]])


# --------------------------------------------------------------------------
# pick_device
# --------------------------------------------------------------------------

def test_pick_device_honours_an_explicit_request():
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_auto_picks_mps_on_this_machine():
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS on this machine")
    assert pick_device("auto").type == "mps"


def test_pick_device_auto_never_returns_an_unavailable_backend():
    """'auto' must degrade to something that actually exists."""
    device = pick_device("auto")
    if device.type == "mps":
        assert torch.backends.mps.is_available()
    elif device.type == "cuda":
        assert torch.cuda.is_available()
    else:
        assert device.type == "cpu"


def test_pick_device_rejects_a_backend_that_is_not_available():
    """Asking for a backend this machine does not have must fail loudly here,
    not thirty frames deep inside torch.load()."""
    unavailable = "cuda" if not torch.cuda.is_available() else None
    if unavailable is None:
        pytest.skip("this machine has CUDA; nothing unavailable to ask for")
    with pytest.raises((RuntimeError, SystemExit, ValueError)):
        pick_device(unavailable)


# --------------------------------------------------------------------------
# build_model
# --------------------------------------------------------------------------

def test_classification_head_predicts_exactly_two_classes(fresh_model):
    head = fresh_model.head.classification_head
    assert head.num_columns == NUM_CLASSES == 2, "background + hand, nothing else"

    anchors = fresh_model.anchor_generator.num_anchors_per_location()
    convs = [m for m in head.modules() if isinstance(m, torch.nn.Conv2d)]
    # the last conv of each of the 6 feature maps emits anchors * classes channels
    emitters = [c.out_channels for c in convs if c.out_channels in
                {a * NUM_CLASSES for a in anchors}]
    assert len(emitters) == len(anchors), (
        f"expected {len(anchors)} score-emitting convs, found {len(emitters)}"
    )
    for out_channels, num_anchors in zip(emitters, anchors):
        assert out_channels == num_anchors * NUM_CLASSES


def test_model_is_small_enough_to_run_live(fresh_model):
    """SSDlite was picked for speed; ~2.2M params is the whole reason."""
    total = sum(p.numel() for p in fresh_model.parameters())
    assert 2.0e6 < total < 2.4e6, f"{total} params -- not an SSDlite any more"


def test_backbone_and_box_regressor_keep_their_coco_weights(fresh_model, coco_reference):
    """The docstring promises only the classifier is swapped. Verify it."""
    ref_backbone = coco_reference.backbone.state_dict()
    new_backbone = fresh_model.backbone.state_dict()
    assert set(ref_backbone) == set(new_backbone)
    changed = [k for k in ref_backbone
               if not torch.equal(ref_backbone[k], new_backbone[k])]
    assert changed == [], f"backbone lost its COCO weights: {changed[:5]}"

    ref_reg = coco_reference.head.regression_head.state_dict()
    new_reg = fresh_model.head.regression_head.state_dict()
    changed = [k for k in ref_reg if not torch.equal(ref_reg[k], new_reg[k])]
    assert changed == [], f"box regression head lost its COCO weights: {changed[:5]}"


def test_classification_head_is_freshly_initialised(fresh_model, coco_reference):
    """The 91-class COCO classifier must be gone, not merely resized."""
    assert coco_reference.head.classification_head.num_columns == 91
    assert fresh_model.head.classification_head.num_columns == 2


# --------------------------------------------------------------------------
# one training step
# --------------------------------------------------------------------------

@needs_dataset
def test_training_step_produces_both_loss_terms(real_batch):
    images, targets = real_batch
    model = build_model().to(CPU)
    model.train()
    losses = model(list(images), [dict(t) for t in targets])

    assert set(losses) == {"bbox_regression", "classification"}, (
        f"SSD should return both loss terms, got {sorted(losses)}"
    )
    total = sum(losses.values())
    assert torch.isfinite(total), f"loss is not finite: {total}"
    assert total.item() > 0, "a randomly initialised classifier cannot have zero loss"
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} is not finite"
        assert value.item() >= 0, f"{name} is negative"


@needs_dataset
def test_training_step_actually_moves_the_weights(real_batch):
    """A loop that runs but does not learn is the bug this catches."""
    images, targets = real_batch
    loader = [(list(images), [dict(t) for t in targets])]

    model = build_model().to(CPU)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)

    head_name, head_param = next(
        (n, p) for n, p in model.named_parameters()
        if n.startswith("head.classification_head") and p.dim() > 1
    )
    body_name, body_param = next(
        (n, p) for n, p in model.named_parameters()
        if n.startswith("backbone") and p.dim() > 1
    )
    head_before = head_param.detach().clone()
    body_before = body_param.detach().clone()

    mean_loss = train_one_epoch(model, loader, optimizer, CPU, epoch=1)

    assert mean_loss > 0 and mean_loss == mean_loss, f"bad mean loss {mean_loss}"
    assert head_param.grad is not None, f"no gradient reached {head_name}"
    assert torch.isfinite(head_param.grad).all(), f"non-finite gradient on {head_name}"
    assert head_param.grad.abs().sum() > 0, f"gradient is all zeros on {head_name}"
    # AdamW's decoupled weight decay nudges every parameter by ~lr*wd*|p| (~5e-8)
    # even with a zero gradient, so require a step far larger than that.
    head_delta = (head_param.detach() - head_before).abs().max().item()
    body_delta = (body_param.detach() - body_before).abs().max().item()
    assert head_delta > 1e-6, (
        f"{head_name} moved by {head_delta} after optimizer.step() -- not learning"
    )
    assert body_delta > 1e-6, (
        f"{body_name} moved by {body_delta} -- the backbone is effectively frozen"
    )


@needs_dataset
def test_train_one_epoch_averages_over_batches(real_batch):
    """The reported loss is a per-batch mean, so feeding the same batch twice
    must not double it."""
    images, targets = real_batch
    batch = (list(images), [dict(t) for t in targets])

    torch.manual_seed(0)
    model = build_model().to(CPU)
    model.eval()  # train_one_epoch is responsible for switching to train mode
    frozen = torch.optim.SGD(model.parameters(), lr=0.0)
    once = train_one_epoch(model, [batch], frozen, CPU, epoch=1)
    twice = train_one_epoch(model, [batch, batch], frozen, CPU, epoch=1)

    assert model.training, "train_one_epoch must put the model in train mode"
    assert once == pytest.approx(twice, rel=0.05), (
        f"mean loss over 1 batch {once} vs the same batch twice {twice}"
    )


# --------------------------------------------------------------------------
# evaluate() -- counts checked by hand
# --------------------------------------------------------------------------

def test_evaluate_perfect_prediction():
    gt = [[10, 10, 50, 50], [100, 100, 160, 160]]
    m = run_evaluate([_pred(gt, [0.9, 0.8])], [_truth(gt)])
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_evaluate_completely_wrong_prediction():
    m = run_evaluate(
        [_pred([[200, 200, 240, 240]], [0.9])],
        [_truth([[10, 10, 50, 50]])],
    )
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_evaluate_empty_prediction_is_all_false_negatives():
    m = run_evaluate(
        [_pred([], [])],
        [_truth([[10, 10, 50, 50], [100, 100, 160, 160]])],
    )
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 2)
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_evaluate_prediction_with_no_ground_truth_is_a_false_positive():
    m = run_evaluate([_pred([[10, 10, 50, 50]], [0.9])], [_truth([])])
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 0)
    assert m["precision"] == 0.0


def test_evaluate_two_predictions_cannot_both_claim_one_box():
    """1 TP + 1 FP, never 2 TP. Greedy matching must not double-claim."""
    m = run_evaluate(
        [_pred([[10, 10, 50, 50], [12, 12, 52, 52]], [0.95, 0.90])],
        [_truth([[10, 10, 50, 50]])],
    )
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2 / 3)


def test_evaluate_ignores_predictions_below_the_score_threshold():
    args = ([_pred([[10, 10, 50, 50]], [0.40])], [_truth([[10, 10, 50, 50]])])
    kept = run_evaluate(*args, score_threshold=0.3)
    assert (kept["tp"], kept["fp"], kept["fn"]) == (1, 0, 0)

    dropped = run_evaluate(*args, score_threshold=0.5)
    assert (dropped["tp"], dropped["fp"], dropped["fn"]) == (0, 0, 1)


def test_evaluate_iou_threshold_is_inclusive_at_the_boundary():
    truth = _truth([[0, 0, 100, 100]])                       # area 10000
    exactly_half = run_evaluate([_pred([[0, 0, 100, 50]], [0.9])], [truth])
    assert (exactly_half["tp"], exactly_half["fp"]) == (1, 0), "IoU == 0.50 must match"

    just_under = run_evaluate([_pred([[0, 0, 100, 49]], [0.9])], [truth])
    assert (just_under["tp"], just_under["fp"], just_under["fn"]) == (0, 1, 1)


def test_evaluate_highest_confidence_prediction_claims_the_box_first():
    """The good box scores lower than the junk box; the junk box must not win."""
    truth = _truth([[10, 10, 50, 50]])
    m = run_evaluate(
        [_pred([[10, 10, 50, 50], [11, 11, 51, 51]], [0.60, 0.95])],
        [truth],
    )
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_evaluate_accumulates_across_images_and_batches():
    batch_one = [_pred([[10, 10, 50, 50]], [0.9]), _pred([], [])]
    batch_two = [_pred([[10, 10, 50, 50], [900, 900, 950, 950]], [0.9, 0.8])]
    truths_one = [_truth([[10, 10, 50, 50]]), _truth([[0, 0, 20, 20]])]
    truths_two = [_truth([[10, 10, 50, 50]])]

    images_one = [torch.zeros(3, 8, 8) for _ in batch_one]
    images_two = [torch.zeros(3, 8, 8) for _ in batch_two]
    model = _StubDetector([batch_one, batch_two])
    m = evaluate(model, [(images_one, truths_one), (images_two, truths_two)], CPU)

    # image 1: 1 TP.  image 2: nothing predicted, 1 FN.  image 3: 1 TP + 1 FP.
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_evaluate_metrics_are_internally_consistent():
    m = run_evaluate(
        [_pred([[10, 10, 50, 50], [12, 12, 52, 52], [900, 900, 950, 950]],
               [0.9, 0.8, 0.7])],
        [_truth([[10, 10, 50, 50], [100, 100, 160, 160]])],
    )
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    assert m["precision"] == pytest.approx(tp / (tp + fp))
    assert m["recall"] == pytest.approx(tp / (tp + fn))
    assert m["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))


def test_evaluate_greedy_matching_gives_up_on_overlapping_ground_truth():
    """Documents a known limitation of the argmax-then-check matching.

    Two heavily overlapping ground-truth hands, two good predictions. The second
    prediction's *best* IoU is with the already-claimed box, so it is written off
    as a false positive even though the other ground-truth box is unclaimed and
    well inside the IoU threshold. Standard greedy matching (COCO/VOC) assigns
    each prediction to its best *unmatched* box and would score 2 TP / 0 FP / 0 FN.
    """
    m = run_evaluate(
        [_pred([[0, 0, 100, 100], [0, 0, 102, 102]], [0.9, 0.8])],
        [_truth([[0, 0, 100, 100], [0, 0, 110, 110]])],
    )
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)  # ideal matching: (2, 0, 0)


# --------------------------------------------------------------------------
# checkpoint roundtrip -- the contract webcam_detect.py depends on
# --------------------------------------------------------------------------

@needs_checkpoint
@needs_dataset
def test_checkpoint_roundtrips_into_a_fresh_build_model(tmp_path, videos):
    """Save, reload into a brand new build_model(), get identical predictions."""
    state = torch.load(CHECKPOINT, map_location=CPU, weights_only=False)
    assert "model" in state, "checkpoint must carry a 'model' state_dict"

    original = build_model(state.get("num_classes", NUM_CLASSES))
    original.load_state_dict(state["model"])   # strict: no missing/unexpected keys
    original.eval()

    image, _ = EgoHandsDetection(videos, [0], train=False)[0]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        before = original([image])[0]
    assert len(before["boxes"]) > 0, "roundtrip test needs a non-empty prediction"

    saved = tmp_path / "roundtrip.pth"
    torch.save(
        {"model": original.state_dict(), "epoch": 1,
         "metrics": {}, "num_classes": NUM_CLASSES},
        saved,
    )

    reloaded_state = torch.load(saved, map_location=CPU, weights_only=False)
    reloaded = build_model(reloaded_state.get("num_classes", NUM_CLASSES))
    reloaded.load_state_dict(reloaded_state["model"])
    reloaded.eval()
    with torch.no_grad():
        after = reloaded([image])[0]

    assert len(after["boxes"]) == len(before["boxes"])
    torch.testing.assert_close(after["boxes"], before["boxes"], rtol=0, atol=1e-6)
    torch.testing.assert_close(after["scores"], before["scores"], rtol=0, atol=1e-6)
    assert torch.equal(after["labels"], before["labels"])
    assert set(after["labels"].tolist()) <= {1}, "only hands, label 1"


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------

def _run_cli(python_bin, repo, *args, timeout=600):
    return subprocess.run(
        [python_bin, "train_detector.py", *args],
        cwd=str(repo), capture_output=True, text=True, timeout=timeout,
    )


def test_cli_rejects_an_unknown_device(python_bin, repo):
    result = _run_cli(python_bin, repo, "--device", "banana", "--eval-only", timeout=120)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_eval_only_without_a_checkpoint_fails_cleanly(python_bin, repo, tmp_path):
    missing = tmp_path / "nope" / "hand_detector.pth"
    result = _run_cli(python_bin, repo, "--eval-only", "--workers", "0",
                      "--checkpoint", str(missing))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr, f"raw traceback:\n{result.stderr[-800:]}"
    assert "no checkpoint at" in result.stderr
    assert not missing.exists(), "a failed eval must not create the checkpoint"


@needs_checkpoint
@needs_dataset
def test_cli_eval_only_reproduces_the_checkpoint_f1(python_bin, repo):
    """Scoring the shipped checkpoint must give back the F1 it was saved with."""
    stored = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["metrics"]
    result = _run_cli(python_bin, repo, "--eval-only", "--workers", "0")
    assert result.returncode == 0, result.stderr[-2000:]

    out = result.stdout

    def number(label):
        match = re.search(rf"^{label}\s+([-\d.]+)$", out, re.MULTILINE)
        assert match, f"no '{label}' line in:\n{out[-1500:]}"
        return float(match.group(1))

    tp, fp, fn = (int(number(k)) for k in
                  ("true positives", "false positives", "false negatives"))
    precision, recall, f1 = (number(k) for k in ("precision", "recall", "F1"))

    assert f1 == pytest.approx(0.848, abs=0.01), f"F1 drifted to {f1}"
    assert f1 == pytest.approx(stored["f1"], abs=0.01)
    assert precision == pytest.approx(stored["precision"], abs=0.01)
    assert recall == pytest.approx(stored["recall"], abs=0.01)
    # printed counts must explain the printed rates
    assert precision == pytest.approx(tp / (tp + fp), abs=0.001)
    assert recall == pytest.approx(tp / (tp + fn), abs=0.001)
    assert tp == pytest.approx(stored["tp"], rel=0.02)


@needs_dataset
def test_cli_short_training_run_completes_and_writes_a_checkpoint(
    python_bin, repo, tmp_path
):
    """Plumbing only: 2 videos, 1 epoch. The numbers are expected to be bad."""
    out_path = tmp_path / "smoke" / "detector.pth"
    result = _run_cli(python_bin, repo, "--limit-videos", "2", "--epochs", "1",
                      "--batch-size", "8", "--workers", "0",
                      "--checkpoint", str(out_path))
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Traceback" not in result.stderr
    assert out_path.is_file(), "training finished but saved no checkpoint"

    assert re.search(r"epoch\s+1/1\s+loss\s+[\d.]+", result.stdout), result.stdout[-800:]
    assert "done. best F1" in result.stdout

    state = torch.load(out_path, map_location="cpu", weights_only=False)
    assert set(state) >= {"model", "epoch", "metrics", "num_classes"}
    assert state["epoch"] == 1
    assert state["num_classes"] == NUM_CLASSES
    # the checkpoint must be loadable by exactly the call webcam_detect.py makes
    model = build_model(state.get("num_classes", NUM_CLASSES))
    model.load_state_dict(state["model"])

    if CHECKPOINT.is_file():
        assert CHECKPOINT.stat().st_size > 0, "the shipped checkpoint must be untouched"
