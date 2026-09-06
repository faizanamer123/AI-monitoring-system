"""
detection_dataset.py builds an object detection dataset from the EgoHands videos,
using the ground-truth bounding boxes that are already stored in metadata.mat.

This is deliberately separate from EgoHandsDataset in dataset.py. That one decodes
every frame up front, which costs ~527 MB of RAM per 100 frames -- fine for the one
video the UNet trains on, impossible for all 4800 labelled frames. This dataset is
lazy: it stores only (video, frame) index pairs and decodes each JPEG on demand.

Each item comes back in the format torchvision detection models expect:
    image  : FloatTensor[3, 320, 320] with values in [0, 1]
    target : {"boxes":  FloatTensor[N, 4] as xyxy pixel corners,
              "labels": Int64Tensor[N]}

See also train_detector.py, webcam_detect.py
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from get_bounding_boxes import get_bounding_boxes
from get_frame_path import get_frame_path

# ssdlite320_mobilenet_v3_large resizes everything to 320x320 internally, so we
# match that here and let the model's own transform become a no-op during training.
IMAGE_SIZE = 320

# torchvision reserves label 0 for background, so hands have to start at 1.
HAND_LABEL = 1
NUM_CLASSES = 2

# Smallest zoom-in window, as a fraction of the frame. Module-level so a training run
# can push it without touching the dataset: at 0.35 augmented hands land near 0.25 of
# the frame, while a webcam user's hand sits at ~0.44, so a smaller floor is what closes
# that gap. train_detector.py --zoom-min overrides it.
ZOOM_MIN_SCALE = 0.35
ZOOM_PROB = 0.5

# native EgoHands capture resolution, hardcoded the same way get_bounding_boxes does
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720


def video_group(video):
    """Return the (activity, location) key for a video, parsed from its video_id.

    video_id looks like PUZZLE_COURTYARD_B_S. We read the split key off the string
    rather than the activity_id/location_id columns because get_meta_by only cleans
    up some of those columns on the frame it returns.
    """
    activity, location, _viewer, _partner = str(video.loc['video_id'][0]).split('_')
    return activity, location


def split_videos(videos, val_per_group=1, seed=0):
    """Split into train/val at the VIDEO level, never at the frame level.

    The 100 frames inside one 90-second clip are near-duplicates of each other, so
    a random frame split would leak validation frames into training and report an
    accuracy that means nothing. The 48 videos form 12 groups of 4 (activity x
    location); holding out `val_per_group` from each keeps every location, activity
    and actor represented on both sides of the split.

    Returns (train_indices, val_indices) as positions into `videos`.
    """
    groups = {}
    for i in range(len(videos)):
        groups.setdefault(video_group(videos.iloc[i]), []).append(i)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for key in sorted(groups):
        members = sorted(groups[key])
        rng.shuffle(members)
        val_idx.extend(members[:val_per_group])
        train_idx.extend(members[val_per_group:])
    return sorted(train_idx), sorted(val_idx)


class EgoHandsDetection(Dataset):
    """Lazy (image, boxes) dataset over some subset of the EgoHands videos."""

    def __init__(self, videos, video_indices=None, train=False):
        self.videos = videos
        self.train = train
        if video_indices is None:
            video_indices = range(len(videos))

        # flat sample index -> (which video, which frame inside that video)
        self.samples = []
        for video_num in video_indices:
            frames = len(videos.iloc[video_num].loc['labelled_frames'][0])
            self.samples.extend((video_num, f) for f in range(frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_num, frame_num = self.samples[index]
        video = self.videos.iloc[video_num]

        path = get_frame_path(video, frame_num)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(
                f"could not read {path}. Is the _LABELLED_SAMPLES folder next to the code, "
                "and are you running from the repo root?"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = self._boxes_xyxy(video, frame_num)

        # zoom happens first, in original frame coordinates
        # ZOOM_PROB > 0 short-circuits so no random draw is consumed when zoom is
        # disabled -- otherwise turning it off still perturbs the RNG sequence that
        # the flip and jitter draws come from.
        if self.train and ZOOM_PROB > 0 and random.random() < ZOOM_PROB:
            img, boxes = self._zoom_in(img, boxes)

        # squash to the square the detector works in, and carry the boxes with it
        source_h, source_w = img.shape[:2]
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        boxes[:, [0, 2]] *= IMAGE_SIZE / source_w
        boxes[:, [1, 3]] *= IMAGE_SIZE / source_h

        if self.train:
            img, boxes = self._augment(img, boxes)

        image = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255.0)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.full((len(boxes),), HAND_LABEL, dtype=torch.int64),
        }
        return image, target

    @staticmethod
    def _boxes_xyxy(video, frame_num):
        """Convert get_bounding_boxes' 4x4 [x, y, w, h] matrix into xyxy corners.

        Rows for hands that are not in the frame come back as zeros, and
        segmentation2box defines w as x2 - x1 + 1, so both have to be undone here.
        Degenerate slivers are dropped -- torchvision raises on a box with x2 <= x1.
        """
        boxes = []
        for x, y, w, h in get_bounding_boxes(video, frame_num):
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w - 1, y + h - 1])
        return np.array(boxes, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def _zoom_in(img, boxes, min_scale=None):
        """Crop a random window around a hand, so that hand fills more of the frame.

        This is the augmentation that matters most for webcam use. EgoHands is shot
        from a head-mounted camera at table distance, so the median hand covers 0.207
        of the frame and the 90th percentile only 0.358. A webcam user holds a hand up
        close, at roughly 0.44 -- larger than 90% of anything in training.

        Asked to find a hand at twice the scale it knows, the detector fires on the
        PARTS instead: a palm at 0.44 looks like a whole hand at 0.22. That is what
        produces two boxes on one hand. Zooming in during training puts those scales
        in the training distribution.
        """
        if not len(boxes):
            return img, boxes
        if min_scale is None:
            min_scale = ZOOM_MIN_SCALE

        height, width = img.shape[:2]
        anchor = boxes[random.randrange(len(boxes))]
        cx = (anchor[0] + anchor[2]) / 2.0
        cy = (anchor[1] + anchor[3]) / 2.0

        # Log-uniform, not uniform. Uniform(min_scale, 1.0) puts most of its mass near
        # 1.0, so the median augmented hand barely moves however low the floor goes.
        # Sampling log-uniformly weights the small windows that actually produce
        # webcam-scale hands.
        span = 1.0 / max(min_scale, 1e-3)
        scale = min_scale * (span ** random.random())
        cw, ch = width * scale, height * scale
        # keep the chosen hand inside the crop, then clamp the window to the image
        x0 = min(max(cx - cw / 2.0, 0.0), width - cw)
        y0 = min(max(cy - ch / 2.0, 0.0), height - ch)
        x1, y1 = x0 + cw, y0 + ch

        cropped = img[int(y0):int(y1), int(x0):int(x1)]
        if cropped.size == 0:
            return img, boxes

        shifted = boxes.copy()
        shifted[:, [0, 2]] -= x0
        shifted[:, [1, 3]] -= y0
        # Assign the result back explicitly. `np.clip(a[:, [0, 2]], ..., out=a[:, [0, 2]])`
        # silently does nothing: fancy indexing returns a COPY, so out= writes into a
        # temporary and the boxes keep their out-of-range values.
        shifted[:, [0, 2]] = np.clip(shifted[:, [0, 2]], 0, cropped.shape[1] - 1)
        shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]], 0, cropped.shape[0] - 1)

        # drop hands the crop cut away to a sliver
        wide = (shifted[:, 2] - shifted[:, 0]) > 4
        tall = (shifted[:, 3] - shifted[:, 1]) > 4
        shifted = shifted[wide & tall]
        if not len(shifted):
            return img, boxes
        return cropped, shifted

    @staticmethod
    def _augment(img, boxes):
        """Horizontal flip plus brightness/contrast jitter.

        The jitter matters more than it looks. EgoHands was shot in three fixed
        settings; your webcam will be in a fourth one, with different lighting and
        a different skin-tone-to-background contrast, and that is the single
        biggest thing that breaks the live demo.

        There is deliberately no vertical flip and no large rotation, unlike the
        UNet's augmentation in train.py -- hands enter a first-person frame from
        the bottom, and upside-down hands only waste model capacity.
        """
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            if len(boxes):
                x1 = boxes[:, 0].copy()
                # Continuous-coordinate mirror: x -> W - x. The inclusive-pixel-index
                # form (W - 1 - x) shifts every flipped box one pixel left and can
                # drive x1 negative, which is invalid input to the detection loss.
                boxes[:, 0] = IMAGE_SIZE - boxes[:, 2]
                boxes[:, 2] = IMAGE_SIZE - x1
        if random.random() < 0.8:
            # Deliberately NOT cv2.convertScaleAbs: that computes |src*alpha + beta|,
            # so a negative beta reflects dark pixels back up the scale instead of
            # clamping them at 0 -- a pixel of 10 with beta=-30 came out as 20,
            # brighter than it started. Shadows inverted on ~40% of training images.
            alpha = random.uniform(0.7, 1.3)
            beta = random.uniform(-30, 30)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, boxes


def collate_fn(batch):
    """Detection targets hold a variable number of boxes, so they cannot be stacked
    into a tensor the way classification labels can. DataLoader needs to be told to
    hand the batch over as plain tuples instead."""
    return tuple(zip(*batch))


class NegativeFrames(Dataset):
    """Images that contain NO hands, yielded with empty targets.

    Why this exists: EgoHands is head-mounted footage of hands on tables. It contains
    no faces, no torsos and no front-facing views, so a detector trained purely on it
    never learns that a large skin-coloured oval is *not* a hand -- and then boxes a
    webcam user's face at 1.00 confidence. Feeding frames of exactly that kind with
    zero ground-truth boxes teaches the background class directly.

    torchvision detection models accept empty targets natively: SSD marks every anchor
    as background for such an image, which is precisely the signal we want.
    """

    def __init__(self, folder, train=True):
        self.paths = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
        self.train = train
        if not self.paths:
            raise FileNotFoundError(f"no images found in {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"could not read negative frame {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        if self.train:
            # same jitter the positives get, so the negatives are not trivially
            # distinguishable by colour statistics alone
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            if random.random() < 0.8:
                alpha = random.uniform(0.7, 1.3)
                beta = random.uniform(-30, 30)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        image = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255.0)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
        return image, target
