"""
True end-to-end pipeline test.

The other test modules exercise components in isolation. This one runs the whole
journey a user actually takes, in order, in one process, and asserts at every seam:

    raw dataset on disk
        -> metadata agrees with the JPEGs
        -> dataset yields images + boxes
        -> training actually LEARNS (not merely "runs without crashing")
        -> checkpoint saves and reloads identically
        -> evaluation produces metrics in a sane range
        -> inference puts boxes on real hands
        -> the shipped checkpoint meets an accuracy floor
        -> all three detectors work on the same frame

The assertion that matters most is stage 3. A training loop that runs cleanly but
does not learn passes every shape check, every smoke test, and every CI job -- and
produces a useless model. So we train a real model on a small subset and require it
to beat its own random-initialised starting point on real data.

Run:  MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_e2e_pipeline.py -v -s
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import CHECKPOINT, MP_MODEL, REPO, needs_checkpoint, needs_dataset

pytestmark = pytest.mark.e2e

# A real but small training run: 2 videos, 1 epoch. Big enough to prove learning
# happens, small enough that the whole module stays near a minute.
E2E_VIDEOS = 2
E2E_EPOCHS = 1


# --------------------------------------------------------------------------- #
# stage 1: the data on disk is what the metadata claims                        #
# --------------------------------------------------------------------------- #

@needs_dataset
def test_stage1_every_annotated_video_has_its_frames_on_disk(videos):
    """metadata.mat and _LABELLED_SAMPLES must agree, or every later stage is a lie."""
    root = REPO / "_LABELLED_SAMPLES"
    missing, short = [], []
    for i in range(len(videos)):
        vid = str(videos.iloc[i].loc["video_id"][0])
        folder = root / vid
        if not folder.is_dir():
            missing.append(vid)
            continue
        n_jpg = len(list(folder.glob("frame_*.jpg")))
        n_annotated = len(videos.iloc[i].loc["labelled_frames"][0])
        if n_jpg < n_annotated:
            short.append(f"{vid}: {n_jpg} jpg < {n_annotated} annotated")
    assert not missing, f"videos in metadata with no folder: {missing}"
    assert not short, f"videos with fewer frames than annotations: {short}"


@needs_dataset
def test_stage1_frame_paths_resolve_and_decode(videos):
    """A path that resolves is not enough -- the JPEG has to actually decode."""
    import cv2
    from get_frame_path import get_frame_path

    for v in (0, 17, 47):
        for f in (0, 50, 99):
            path = get_frame_path(videos.iloc[v], f)
            assert Path(path).is_file(), f"missing frame: {path}"
            img = cv2.imread(str(path))
            assert img is not None, f"undecodable frame: {path}"
            assert img.shape == (720, 1280, 3), f"unexpected frame shape {img.shape} at {path}"


# --------------------------------------------------------------------------- #
# stage 2 + 3: the pipeline trains, and it learns                              #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a real model on a small subset and return everything needed to judge it.

    Module-scoped: this is the expensive fixture, and stages 3-6 all interrogate the
    same run rather than each paying for their own.
    """
    from detection_dataset import EgoHandsDetection, collate_fn, split_videos
    from get_meta_by import get_meta_by
    from torch.utils.data import DataLoader
    from train_detector import build_model, evaluate, pick_device, train_one_epoch

    if not (REPO / "_LABELLED_SAMPLES").is_dir():
        pytest.skip("_LABELLED_SAMPLES not present")

    device = pick_device("auto")
    videos = get_meta_by().iloc[:E2E_VIDEOS]
    train_idx, val_idx = split_videos(videos)

    train_ds = EgoHandsDetection(videos, train_idx, train=True)
    val_ds = EgoHandsDetection(videos, val_idx, train=False)
    loader_args = dict(batch_size=8, num_workers=0, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    torch.manual_seed(0)
    model = build_model().to(device)

    # score the untrained model first -- this is the baseline the run must beat
    before = evaluate(model, val_loader, device, 0.5)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4
    )
    losses = []
    for epoch in range(E2E_EPOCHS):
        losses.append(train_one_epoch(model, train_loader, optimizer, device, epoch + 1))

    after = evaluate(model, val_loader, device, 0.5)

    path = tmp_path_factory.mktemp("e2e") / "e2e_checkpoint.pth"
    torch.save({"model": model.state_dict(), "epoch": E2E_EPOCHS,
                "metrics": after, "num_classes": 2}, path)

    return {"model": model, "device": device, "path": path, "before": before,
            "after": after, "losses": losses, "val_ds": val_ds, "val_loader": val_loader}


@needs_dataset
def test_stage2_training_runs_and_produces_a_finite_loss(trained):
    assert trained["losses"], "no epoch completed"
    for i, loss in enumerate(trained["losses"]):
        assert np.isfinite(loss), f"epoch {i} loss was {loss}"
        assert loss > 0, f"epoch {i} loss was {loss}; a zero loss means no signal"


@needs_dataset
def test_stage3_training_actually_learns_not_just_runs(trained):
    """THE test. A pipeline that runs but does not learn passes everything else."""
    before, after = trained["before"], trained["after"]
    print(f"\n    untrained: P {before['precision']:.3f} R {before['recall']:.3f} "
          f"F1 {before['f1']:.3f}")
    print(f"    trained:   P {after['precision']:.3f} R {after['recall']:.3f} "
          f"F1 {after['f1']:.3f}")
    assert after["f1"] > before["f1"], (
        f"one epoch of training did not improve F1 ({before['f1']:.3f} -> "
        f"{after['f1']:.3f}). The loop runs but the model is not learning."
    )
    assert after["tp"] > 0, "trained model found no hands at all"


# --------------------------------------------------------------------------- #
# stage 4: the checkpoint survives a round trip                                #
# --------------------------------------------------------------------------- #

@needs_dataset
def test_stage4_checkpoint_reloads_to_identical_predictions(trained):
    """What gets saved must be what gets loaded, bit for bit on the same input."""
    from train_detector import build_model

    device = trained["device"]
    image, _ = trained["val_ds"][0]
    original = trained["model"].eval()
    with torch.no_grad():
        expected = original([image.to(device)])[0]

    state = torch.load(trained["path"], map_location=device, weights_only=False)
    restored = build_model(state["num_classes"]).to(device)
    restored.load_state_dict(state["model"])
    restored.eval()
    with torch.no_grad():
        actual = restored([image.to(device)])[0]

    assert len(expected["boxes"]) == len(actual["boxes"]), "box count changed after reload"
    torch.testing.assert_close(expected["boxes"].cpu(), actual["boxes"].cpu(),
                               rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(expected["scores"].cpu(), actual["scores"].cpu(),
                               rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# stage 5: the shipped checkpoint meets an accuracy floor                      #
# --------------------------------------------------------------------------- #

@needs_dataset
@needs_checkpoint
def test_stage5_shipped_checkpoint_meets_accuracy_floor():
    """A regression gate. If a future change drops AP below this, the suite says so.

    Measured on a 200-frame slice of the real validation split, so it runs in seconds
    while still being a genuine held-out measurement.
    """
    from evaluate_detector import average_precision, collect
    from detection_dataset import EgoHandsDetection, split_videos
    from get_meta_by import get_meta_by
    from train_detector import build_model, pick_device

    device = pick_device("auto")
    videos = get_meta_by()
    _, val_idx = split_videos(videos)
    dataset = EgoHandsDetection(videos, val_idx, train=False)

    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_model(state.get("num_classes", 2)).to(device)
    model.load_state_dict(state["model"])

    records = collect(model, dataset, device, limit=200)
    ap50, _, _ = average_precision(records, 0.5)
    print(f"\n    AP@0.50 on 200 held-out frames: {ap50:.4f}")
    assert ap50 >= 0.85, f"AP@0.50 regressed to {ap50:.4f} (floor 0.85)"


@needs_dataset
@needs_checkpoint
def test_stage5_predictions_land_on_real_hands():
    """End-to-end sanity in the most literal sense: do the boxes cover the hands?"""
    from torchvision.ops import box_iou
    from detection_dataset import EgoHandsDetection, split_videos
    from get_meta_by import get_meta_by
    from train_detector import build_model, pick_device

    device = pick_device("auto")
    videos = get_meta_by()
    _, val_idx = split_videos(videos)
    dataset = EgoHandsDetection(videos, val_idx, train=False)

    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_model(state.get("num_classes", 2)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    hit = total = 0
    for i in range(0, 600, 25):
        image, target = dataset[i]
        with torch.no_grad():
            out = model([image.to(device)])[0]
        keep = out["scores"] >= 0.2          # the threshold the sweep found best
        boxes = out["boxes"][keep].cpu()
        gt = target["boxes"]
        if len(gt) == 0:
            continue
        total += len(gt)
        if len(boxes):
            hit += int((box_iou(gt, boxes).max(dim=1).values >= 0.5).sum())

    recall = hit / max(total, 1)
    print(f"\n    recall @score0.2/IoU0.5 over {total} hands: {recall:.3f}")
    assert recall >= 0.75, f"only {recall:.3f} of ground-truth hands were detected"


# --------------------------------------------------------------------------- #
# stage 6: every shipped entry point works on the same real frame              #
# --------------------------------------------------------------------------- #

@needs_dataset
@needs_checkpoint
def test_stage6_all_three_detectors_run_on_the_same_frame(sample_frame_path, tmp_path):
    """The three demos are meant to be interchangeable. Prove they all produce output."""
    import cv2
    from train_detector import pick_device

    frame = cv2.imread(str(sample_frame_path))
    assert frame is not None
    device = pick_device("auto")
    results = {}

    from webcam_detect import detect as ssd_detect, load_model
    boxes, scores = ssd_detect(load_model(str(CHECKPOINT), device), frame, device, 0.2)
    results["ssdlite"] = len(boxes)
    for (x1, y1, x2, y2) in boxes:
        assert 0 <= x1 < x2 <= frame.shape[1], f"ssdlite box out of frame: {(x1,y1,x2,y2)}"
        assert 0 <= y1 < y2 <= frame.shape[0], f"ssdlite box out of frame: {(x1,y1,x2,y2)}"

    if MP_MODEL.is_file():
        from webcam_detect_mp import build_detector, detect as mp_detect
        det = build_detector(str(MP_MODEL), 4, 0.4, video_mode=False)
        results["mediapipe"] = len(mp_detect(det, frame))

    import argparse
    from webcam_detect_cv import detect as cv_detect
    cv_args = argparse.Namespace(min_area=0.004, max_area=0.25, max_hands=4)
    results["classical"] = len(cv_detect(frame, cv_args)[0])

    print(f"\n    detections on {sample_frame_path.name}: {results}")
    assert results["ssdlite"] > 0, "the trained detector found nothing on a real hand frame"


@needs_dataset
def test_stage6_cli_entry_points_complete_on_a_real_frame(python_bin, sample_frame_path, tmp_path):
    """Shell out exactly the way the README tells a user to, and require exit 0."""
    if not MP_MODEL.is_file():
        pytest.skip("no mediapipe model")
    proc = subprocess.run(
        [python_bin, "webcam_detect_mp.py", "--image", str(sample_frame_path)],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"webcam_detect_mp.py --image failed:\n{proc.stderr[-1500:]}"
    produced = REPO / "mp_detect_result.png"
    assert produced.is_file(), "documented output PNG was not written"
    produced.unlink()


# --------------------------------------------------------------------------- #
# stage 7: the camera, the one part that touches hardware                      #
# --------------------------------------------------------------------------- #

@pytest.mark.camera
def test_stage7_camera_delivers_real_video():
    """Open the camera and require actual image content.

    Deliberately tolerant of the first-frame-black warm-up on macOS: asserting on
    frame 0 gives a false 'permission denied' reading. We sample a window instead.
    """
    import cv2
    import time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pytest.skip("no camera available on this machine")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        best = 0
        deadline = time.time() + 6.0
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                best = max(best, int(frame.max()))
                if best > 0:
                    break
            time.sleep(0.1)
    finally:
        cap.release()
    assert best > 0, (
        "every frame was pure black. On macOS grant camera access to the terminal in "
        "System Settings -> Privacy & Security -> Camera."
    )
    print(f"\n    camera delivered video (peak pixel {best})")
