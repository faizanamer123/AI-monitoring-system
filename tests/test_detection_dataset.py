"""Behaviour tests for detection_dataset.py -- the lazy PyTorch detection dataset.

The invariants under test, in rough order of how much damage a violation does:

  1. no video appears in both the train and the val split (frame-level leakage would
     silently invalidate every precision/recall number train_detector.py prints);
  2. the horizontal flip moves boxes onto the SAME visual content it moves pixels to
     (a wrong flip poisons half of training while every shape assertion still passes);
  3. brightness/contrast jitter is a plausible lighting change, not an inversion;
  4. every emitted box is a valid, in-image xyxy rectangle, including the zero-hand
     case torchvision insists must be shaped (0, 4) / (0,);
  5. construction stays lazy -- 4800 JPEGs must not be decoded up front.

Run:  .venv/bin/python -m pytest tests/test_detection_dataset.py -v
"""

import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

import detection_dataset as dd
from conftest import needs_dataset
from detection_dataset import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HAND_LABEL,
    IMAGE_SIZE,
    EgoHandsDetection,
    collate_fn,
    split_videos,
    video_group,
)
from get_bounding_boxes import get_bounding_boxes

# A frame (video 3, frame 40) whose rightmost hand box reaches the right edge of the
# original 1280px frame. Found by scanning the whole dataset; see the repro in the
# findings. Flat index into an all-48-video EgoHandsDetection.
RIGHT_EDGE_INDEX = 340

# Flat indices of frames that carry no annotated hand at all (13 exist in total).
ZERO_HAND_INDICES = [2600, 2601, 2849, 2897, 2898, 2899, 2932, 3466, 3893, 4100, 4101, 4146, 4147]


class FixedRandom:
    """Stand-in for the `random` module that detection_dataset._augment reaches for.

    _augment consumes exactly two random() draws (flip?, jitter?) and, if it jitters,
    two uniform() draws (alpha, beta). The values cycle, so one instance drives any
    number of __getitem__ calls identically.
    """

    def __init__(self, flip=True, jitter=False, alpha=1.0, beta=0.0):
        self._draws = [0.0 if flip else 0.99, 0.0 if jitter else 0.99]
        self._uniforms = [alpha, beta]
        self._n = 0
        self._m = 0

    def randrange(self, stop):
        """_zoom_in picks which annotated hand to centre the crop on."""
        return 0

    def random(self):
        value = self._draws[self._n % 2]
        self._n += 1
        return value

    def uniform(self, _a, _b):
        value = self._uniforms[self._m % 2]
        self._m += 1
        return value


@pytest.fixture
def no_augment_randomness(monkeypatch):
    """Factory: install a deterministic stand-in for detection_dataset's `random`."""

    def install(zoom=False, **kwargs):
        monkeypatch.setattr(dd, "random", FixedRandom(**kwargs))
        # Zoom off by default. These tests isolate the flip and the jitter, and the
        # zoom crop runs BEFORE both: it resizes the frame, so interpolation reorders
        # the synthetic gradients the jitter test inspects and shifts the pixels the
        # flip test compares. Leaving it on would make them fail on correct code.
        if not zoom:
            monkeypatch.setattr(dd, "ZOOM_PROB", 0.0)

    return install


def covered_columns(x1, x2):
    """Half-open range of image columns a box spanning continuous [x1, x2] touches."""
    return int(np.floor(x1)), int(np.ceil(x2))


def as_uint8(image_tensor):
    return np.rint(image_tensor.numpy() * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------------
# split_videos: the split that every reported metric depends on
# --------------------------------------------------------------------------------


def test_split_is_36_train_12_val_and_covers_every_video(videos):
    train_idx, val_idx = split_videos(videos)
    assert len(videos) == 48
    assert len(train_idx) == 36
    assert len(val_idx) == 12
    assert sorted(train_idx + val_idx) == list(range(48))


def test_no_video_appears_in_both_splits(videos):
    """The one failure mode that would silently invalidate every reported metric."""
    train_idx, val_idx = split_videos(videos)
    assert set(train_idx).isdisjoint(val_idx)

    train_ids = {str(videos.iloc[i].loc["video_id"][0]) for i in train_idx}
    val_ids = {str(videos.iloc[i].loc["video_id"][0]) for i in val_idx}
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) == 36 and len(val_ids) == 12


def test_split_is_deterministic_and_seed_actually_matters(videos):
    first = split_videos(videos)

    # a different global random state must not move the split: split_videos owns its
    # own Random instance, and train_detector.py never seeds the global one
    import random as global_random

    global_random.seed(12345)
    [global_random.random() for _ in range(50)]
    assert split_videos(videos) == first

    assert split_videos(videos, seed=0) == first
    assert split_videos(videos, seed=7) != first, "seed argument has no effect"


def test_split_is_reproducible_in_a_fresh_process(videos, python_bin, repo):
    """A split that only holds within one process is not a reproducible split."""
    snippet = (
        "import sys; sys.path.insert(0, '.');"
        "from get_meta_by import get_meta_by;"
        "from detection_dataset import split_videos;"
        "print(split_videos(get_meta_by())[1])"
    )
    out = subprocess.run(
        [python_bin, "-c", snippet], cwd=str(repo), capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(split_videos(videos)[1])


def test_val_split_is_stratified_over_locations_and_activities(videos):
    train_idx, val_idx = split_videos(videos)

    val_groups = [video_group(videos.iloc[i]) for i in val_idx]
    assert len(set(val_groups)) == 12, "val must draw exactly one video per activity x location"

    val_activities = {a for a, _ in val_groups}
    val_locations = {loc for _, loc in val_groups}
    assert val_activities == {"CHESS", "JENGA", "PUZZLE", "CARDS"}
    assert val_locations == {"OFFICE", "COURTYARD", "LIVINGROOM"}

    train_groups = [video_group(videos.iloc[i]) for i in train_idx]
    assert len(set(train_groups)) == 12
    assert {a for a, _ in train_groups} == val_activities
    assert {loc for _, loc in train_groups} == val_locations


# --------------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------------


def test_len_matches_the_frames_actually_indexed(videos):
    dataset = EgoHandsDetection(videos)
    expected = sum(len(videos.iloc[i].loc["labelled_frames"][0]) for i in range(len(videos)))
    assert expected == 4800
    assert len(dataset) == expected == len(dataset.samples)


def test_indices_map_to_distinct_video_frame_pairs(videos):
    dataset = EgoHandsDetection(videos)
    assert len(set(dataset.samples)) == len(dataset), "a (video, frame) pair is indexed twice"


def test_subset_dataset_only_indexes_the_videos_it_was_given(videos):
    train_idx, val_idx = split_videos(videos)
    train_ds = EgoHandsDetection(videos, train_idx, train=True)
    val_ds = EgoHandsDetection(videos, val_idx, train=False)

    assert len(train_ds) == 3600
    assert len(val_ds) == 1200
    assert {v for v, _ in train_ds.samples} == set(train_idx)
    assert {v for v, _ in val_ds.samples} == set(val_idx)
    # frame-level leakage check, not just video-level
    assert set(train_ds.samples).isdisjoint(val_ds.samples)


# --------------------------------------------------------------------------------
# item contract
# --------------------------------------------------------------------------------


@needs_dataset
def test_getitem_returns_the_contract_torchvision_expects(videos):
    dataset = EgoHandsDetection(videos, train=False)
    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert tuple(image.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    assert float(image.max()) > 0.1, "image looks blank"

    assert set(target) == {"boxes", "labels"}
    boxes, labels = target["boxes"], target["labels"]
    assert boxes.dtype == torch.float32
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert labels.dtype == torch.int64
    assert labels.shape == (boxes.shape[0],)
    assert torch.all(labels == HAND_LABEL)
    assert HAND_LABEL != 0, "torchvision reserves 0 for background"


@needs_dataset
def test_boxes_are_valid_and_in_frame_across_300_frames(videos):
    """Box validity at scale: every box from every video must be a real rectangle
    inside the 320x320 image the detector sees."""
    dataset = EgoHandsDetection(videos, train=False)
    indices = list(range(0, len(dataset), 16))
    assert len(indices) >= 300

    bad = []
    total_boxes = 0
    for index in indices:
        _, target = dataset[index]
        boxes = target["boxes"].numpy()
        total_boxes += len(boxes)
        for row in boxes:
            x1, y1, x2, y2 = (float(v) for v in row)
            if not (x2 > x1 and y2 > y1 and 0.0 <= x1 and 0.0 <= y1
                    and x2 <= IMAGE_SIZE and y2 <= IMAGE_SIZE):
                bad.append((index, dataset.samples[index], row.tolist()))

    assert total_boxes > 600, "sampled frames produced suspiciously few boxes"
    assert not bad, f"{len(bad)} invalid boxes, first few: {bad[:5]}"


@needs_dataset
def test_frames_with_zero_hands_produce_empty_tensors_of_the_right_shape(videos):
    """torchvision needs exactly (0, 4) / (0,) -- an empty list or (0,) boxes crashes it."""
    dataset = EgoHandsDetection(videos, train=False)
    for index in ZERO_HAND_INDICES[:4]:
        image, target = dataset[index]
        boxes, labels = target["boxes"], target["labels"]
        assert tuple(boxes.shape) == (0, 4), f"index {index}: boxes {tuple(boxes.shape)}"
        assert tuple(labels.shape) == (0,), f"index {index}: labels {tuple(labels.shape)}"
        assert boxes.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert tuple(image.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_zero_hand_indices_really_are_hand_free(videos):
    """Guards the constant above: if the data changes, the test above must be re-derived."""
    dataset = EgoHandsDetection(videos)
    for index in ZERO_HAND_INDICES:
        video_num, frame_num = dataset.samples[index]
        boxes = EgoHandsDetection._boxes_xyxy(videos.iloc[video_num], frame_num)
        assert boxes.shape == (0, 4)


# --------------------------------------------------------------------------------
# _boxes_xyxy: undoing segmentation2box's w = x2 - x1 + 1
# --------------------------------------------------------------------------------


def test_boxes_xyxy_round_trips_the_polygon_extent_without_an_off_by_one(videos):
    """segmentation2box stores w = x2 - x1 + 1, so xyxy must be x + w - 1, not x + w.

    Checked against the raw annotation polygons rather than against get_bounding_boxes,
    so an off-by-one shared by both would still show up.
    """
    checked = 0
    for video_num in range(0, 48, 4):
        video = videos.iloc[video_num]
        for frame_num in range(0, 100, 20):
            polygons = video.loc["labelled_frames"][0][frame_num]
            expected = []
            for hand in range(1, 5):
                shape = np.int32(polygons[hand])
                if not np.any(shape):
                    continue
                x1 = max(1, int(shape[:, 0].min()))
                y1 = max(1, int(shape[:, 1].min()))
                x2 = min(FRAME_WIDTH, int(shape[:, 0].max()))
                y2 = min(FRAME_HEIGHT, int(shape[:, 1].max()))
                if x2 - x1 + 1 <= 1 or y2 - y1 + 1 <= 1:
                    continue
                expected.append([x1, y1, x2, y2])

            actual = EgoHandsDetection._boxes_xyxy(video, frame_num)
            assert actual.tolist() == [[float(v) for v in b] for b in expected], (
                f"video {video_num} frame {frame_num}"
            )
            checked += len(expected)
    assert checked > 100


def test_boxes_xyxy_inverts_get_bounding_boxes_width_convention(videos):
    video = videos.iloc[0]
    raw = get_bounding_boxes(video, 0)
    converted = EgoHandsDetection._boxes_xyxy(video, 0)
    kept = [row for row in raw if row[2] > 1 and row[3] > 1]
    assert len(kept) == len(converted) > 0
    for (x, y, w, h), (x1, y1, x2, y2) in zip(kept, converted):
        assert (x1, y1) == (x, y)
        assert x2 - x1 == w - 1, "width convention not undone (w = x2 - x1 + 1)"
        assert y2 - y1 == h - 1
        assert x2 > x1 and y2 > y1


def test_boxes_xyxy_drops_absent_hands_and_degenerate_slivers(videos):
    video = videos.iloc[0]
    # every video has frames where fewer than four hands are visible
    counts = {len(EgoHandsDetection._boxes_xyxy(video, f)) for f in range(100)}
    assert min(counts) < 4, "expected some frames with fewer than four hands"
    for frame_num in range(0, 100, 7):
        boxes = EgoHandsDetection._boxes_xyxy(video, frame_num)
        assert boxes.dtype == np.float32
        assert boxes.shape[1] == 4
        assert np.all(boxes[:, 2] > boxes[:, 0])
        assert np.all(boxes[:, 3] > boxes[:, 1])


@needs_dataset
def test_boxes_are_scaled_into_the_320_frame_not_left_in_1280x720(videos):
    dataset = EgoHandsDetection(videos, train=False)
    _, target = dataset[RIGHT_EDGE_INDEX]
    raw = EgoHandsDetection._boxes_xyxy(dataset.videos.iloc[3], 40)
    scaled = target["boxes"].numpy()
    assert len(scaled) == len(raw)
    np.testing.assert_allclose(scaled[:, [0, 2]], raw[:, [0, 2]] * IMAGE_SIZE / FRAME_WIDTH, rtol=1e-5)
    np.testing.assert_allclose(scaled[:, [1, 3]], raw[:, [1, 3]] * IMAGE_SIZE / FRAME_HEIGHT, rtol=1e-5)


# --------------------------------------------------------------------------------
# augmentation -- the part that fails silently
# --------------------------------------------------------------------------------


@needs_dataset
def test_flip_mirrors_the_image_pixels_exactly(videos, no_augment_randomness):
    plain = EgoHandsDetection(videos, train=False)
    before, _ = plain[RIGHT_EDGE_INDEX]

    no_augment_randomness(flip=True, jitter=False)
    flipped_ds = EgoHandsDetection(videos, train=True)
    after, _ = flipped_ds[RIGHT_EDGE_INDEX]

    assert np.array_equal(after.numpy()[:, :, ::-1], before.numpy())


@needs_dataset
def test_flip_moves_boxes_to_the_mirror_side_and_preserves_their_size(videos, no_augment_randomness):
    plain = EgoHandsDetection(videos, train=False)
    _, before = plain[RIGHT_EDGE_INDEX]

    no_augment_randomness(flip=True, jitter=False)
    _, after = EgoHandsDetection(videos, train=True)[RIGHT_EDGE_INDEX]

    b, a = before["boxes"].numpy(), after["boxes"].numpy()
    assert a.shape == b.shape and len(b) > 1
    np.testing.assert_allclose(a[:, [1, 3]], b[:, [1, 3]], atol=1e-5)  # y untouched
    np.testing.assert_allclose(a[:, 2] - a[:, 0], b[:, 2] - b[:, 0], atol=1e-5)  # width kept
    assert np.all(a[:, 2] > a[:, 0])
    # centres must land on the mirrored side of the image (catches "boxes never flipped")
    centre_before = (b[:, 0] + b[:, 2]) / 2
    centre_after = (a[:, 0] + a[:, 2]) / 2
    np.testing.assert_allclose(centre_after, IMAGE_SIZE - centre_before, atol=2.0)


@needs_dataset
def test_flipped_boxes_land_on_the_same_visual_content(videos, no_augment_randomness):
    """The highest-value check here: the flip must move boxes onto exactly the pixels
    it moved. The image flip maps column c -> IMAGE_SIZE - 1 - c, so the columns a box
    covers after the flip must be the mirror of the columns it covered before, and the
    pixels inside must be the mirrored crop -- byte for byte."""
    plain = EgoHandsDetection(videos, train=False)
    image_before, target_before = plain[RIGHT_EDGE_INDEX]

    no_augment_randomness(flip=True, jitter=False)
    image_after, target_after = EgoHandsDetection(videos, train=True)[RIGHT_EDGE_INDEX]

    before = as_uint8(image_before)
    after = as_uint8(image_after)
    b = target_before["boxes"].numpy()
    a = target_after["boxes"].numpy()

    problems = []
    for k in range(len(b)):
        lo, hi = covered_columns(b[k, 0], b[k, 2])
        flo, fhi = covered_columns(a[k, 0], a[k, 2])
        # mirror of the half-open column range [lo, hi) under c -> W-1-c
        want = (IMAGE_SIZE - hi, IMAGE_SIZE - lo)
        if (flo, fhi) != want:
            problems.append(
                f"box {k}: covered cols {lo}..{hi - 1} before, {flo}..{fhi - 1} after, "
                f"mirror says {want[0]}..{want[1] - 1}"
            )
            continue
        ylo, yhi = covered_columns(b[k, 1], b[k, 3])
        crop_before = before[:, ylo:yhi, lo:hi][:, :, ::-1]
        crop_after = after[:, ylo:yhi, flo:fhi]
        if not np.array_equal(crop_before, crop_after):
            problems.append(f"box {k}: pixel content inside the flipped box is not the mirror")

    assert not problems, "flip misplaces boxes relative to the image:\n  " + "\n  ".join(problems)


@needs_dataset
def test_flipped_boxes_stay_inside_the_image(videos, no_augment_randomness):
    """Boxes must remain valid xyxy inside [0, IMAGE_SIZE] after augmentation --
    torchvision's box ops and anchor matching assume image coordinates."""
    no_augment_randomness(flip=True, jitter=False)
    dataset = EgoHandsDetection(videos, train=True)
    # a stride that lands on RIGHT_EDGE_INDEX, so this cannot pass by lucky sampling
    indices = sorted({RIGHT_EDGE_INDEX} | set(range(0, len(dataset), 30)))

    bad = []
    for index in indices:
        _, target = dataset[index]
        for row in target["boxes"].numpy():
            x1, y1, x2, y2 = (float(v) for v in row)
            if not (x2 > x1 and y2 > y1 and 0.0 <= x1 and 0.0 <= y1
                    and x2 <= IMAGE_SIZE and y2 <= IMAGE_SIZE):
                bad.append((index, dataset.samples[index], row.tolist()))

    assert not bad, f"{len(bad)} out-of-image boxes after flip, first few: {bad[:5]}"


@needs_dataset
def test_flip_leaves_empty_boxes_alone(videos, no_augment_randomness):
    no_augment_randomness(flip=True, jitter=True, alpha=1.1, beta=5.0)
    dataset = EgoHandsDetection(videos, train=True)
    _, target = dataset[ZERO_HAND_INDICES[0]]
    assert tuple(target["boxes"].shape) == (0, 4)
    assert tuple(target["labels"].shape) == (0,)


@needs_dataset
def test_brightness_jitter_never_inverts_dark_pixels(videos, no_augment_randomness):
    """A lighting change must be monotone: a pixel that started darker must not end up
    brighter than one that started lighter. Otherwise shadows are turned inside out and
    the model is trained on scenes that cannot physically occur."""
    plain = EgoHandsDetection(videos, train=False)
    original = as_uint8(plain[0][0])

    # alpha/beta well inside the sampled ranges (0.7..1.3 and -30..30)
    no_augment_randomness(flip=False, jitter=True, alpha=1.0, beta=-20.0)
    jittered = as_uint8(EgoHandsDetection(videos, train=True)[0][0])
    assert jittered.shape == original.shape

    mapping = {}
    for value, out in zip(original.ravel().tolist(), jittered.ravel().tolist()):
        mapping.setdefault(value, out)

    levels = sorted(mapping)
    outputs = [mapping[v] for v in levels]
    inversions = [
        (levels[i], outputs[i], levels[i + 1], outputs[i + 1])
        for i in range(len(levels) - 1)
        if outputs[i] > outputs[i + 1]
    ]
    assert not inversions, (
        f"{len(inversions)} intensity inversions with alpha=1.0 beta=-20; "
        f"e.g. input {inversions[0][0]} -> {inversions[0][1]} but "
        f"input {inversions[0][2]} -> {inversions[0][3]}"
    )


@needs_dataset
def test_train_false_applies_no_augmentation(videos):
    """Validation must be deterministic, or the reported F1 wobbles run to run."""
    dataset = EgoHandsDetection(videos, train=False)
    import random as global_random

    global_random.seed(1)
    first_image, first_target = dataset[RIGHT_EDGE_INDEX]
    global_random.seed(999)
    second_image, second_target = dataset[RIGHT_EDGE_INDEX]
    assert torch.equal(first_image, second_image)
    assert torch.equal(first_target["boxes"], second_target["boxes"])


# --------------------------------------------------------------------------------
# collate_fn / DataLoader / torchvision integration
# --------------------------------------------------------------------------------


@needs_dataset
def test_collate_fn_feeds_a_dataloader_and_a_torchvision_detector(videos):
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    dataset = EgoHandsDetection(videos, train=False)
    # include a zero-hand frame: empty targets are where detectors usually blow up
    subset = Subset(dataset, [0, ZERO_HAND_INDICES[0]])
    loader = DataLoader(subset, batch_size=2, num_workers=0, collate_fn=collate_fn)

    batches = list(loader)
    assert len(batches) == 1
    images, targets = batches[0]
    assert isinstance(images, tuple) and isinstance(targets, tuple)
    assert len(images) == len(targets) == 2
    assert all(tuple(i.shape) == (3, IMAGE_SIZE, IMAGE_SIZE) for i in images)
    assert all(set(t) == {"boxes", "labels"} for t in targets)

    torch.manual_seed(0)
    model = ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=dd.NUM_CLASSES
    )

    model.train()
    for images, targets in loader:
        losses = model(list(images), [dict(t) for t in targets])
        assert set(losses) >= {"bbox_regression", "classification"}
        for name, value in losses.items():
            assert torch.isfinite(value), f"{name} is not finite"

    model.eval()
    with torch.no_grad():
        images, _ = batches[0]
        predictions = model(list(images))
    assert len(predictions) == 2
    assert set(predictions[0]) == {"boxes", "scores", "labels"}


# --------------------------------------------------------------------------------
# laziness
# --------------------------------------------------------------------------------


def test_construction_decodes_no_jpegs(videos, monkeypatch):
    """The whole reason this class exists instead of dataset.EgoHandsDataset."""
    calls = []

    def exploding_imread(*args, **kwargs):
        calls.append(args[:1])
        raise AssertionError("dataset construction decoded a JPEG")

    monkeypatch.setattr(dd.cv2, "imread", exploding_imread)

    train_idx, _ = split_videos(videos)
    dataset = EgoHandsDetection(videos, train_idx, train=True)
    assert len(dataset) == 3600
    assert calls == []


def test_construction_is_fast_and_holds_only_index_pairs(videos):
    import resource

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    train_idx, _ = split_videos(videos)
    dataset = EgoHandsDetection(videos, train_idx, train=True)
    elapsed = time.perf_counter() - started
    growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before

    assert elapsed < 3.0, f"constructing a 36-video dataset took {elapsed:.1f}s"
    # 3600 decoded 320x320 RGB frames would be ~1.1 GB; ru_maxrss is bytes on macOS
    scale = 1 if sys.platform == "darwin" else 1024
    assert growth * scale < 100 * 1024 * 1024, f"construction grew RSS by {growth * scale} bytes"

    assert all(isinstance(s, tuple) and len(s) == 2 for s in dataset.samples)
    assert all(isinstance(f, int) for _, f in dataset.samples)
