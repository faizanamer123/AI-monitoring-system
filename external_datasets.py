"""
Loaders for third-party hand-detection datasets, in the same format train_detector.py
already consumes.

WHY THESE EXIST. A detector trained only on EgoHands boxes faces, elbows and feet on a
webcam. EgoHands is head-mounted footage in which the ONLY skin-coloured objects are
hands, so the model never had to learn "hand vs other body part" -- it learned the
shortcut "skin-coloured blob of roughly this size = hand". That is 100% correct on
EgoHands and wrong on any frame containing a face.

The fix is data in which other body parts are visible but NOT labelled as hands:

    COCO-Hand   hands annotated on ordinary COCO photos, so whole people, faces and
                feet are in frame and unlabelled
    HaGRID      one person, frontal, upper body visible -- the webcam framing exactly
    Oxford Hand unconstrained third-person photos of people

Every loader yields the SAME contract as EgoHandsDetection so they can be mixed with
torch.utils.data.ConcatDataset:

    image  : FloatTensor[3, IMAGE_SIZE, IMAGE_SIZE] in [0, 1]
    target : {"boxes": FloatTensor[N, 4] xyxy, "labels": Int64Tensor[N] (all 1)}
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset as Subset_t

from detection_dataset import HAND_LABEL, IMAGE_SIZE, ZOOM_PROB, EgoHandsDetection


class BoxDataset(Dataset):
    """A list of (image_path, boxes_xyxy) rendered into the training contract.

    Every third-party loader below reduces to this. Keeping the parsing (which differs
    per dataset) separate from the tensor plumbing (which does not) means a new dataset
    is a parser function, not another Dataset subclass.

    Images are decoded lazily. These sets run to tens of thousands of images and
    decoding them up front would need tens of gigabytes.
    """

    def __init__(self, samples, train=True, name="external", normalised=False):
        self.samples = [s for s in samples if len(s[1])]      # frames with no hand teach nothing here
        self.train = train
        self.name = name
        # HaGRID stores boxes as fractions of the image; COCO-Hand stores pixels. The
        # flag is carried here so __getitem__ has one scaling rule instead of each
        # loader pre-multiplying by a size it would have to decode the image to learn.
        self.normalised = normalised
        if not self.samples:
            raise ValueError(f"{name}: no annotated samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, boxes = self.samples[index]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"{self.name}: could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()

        height, width = img.shape[:2]
        if self.normalised:
            # convert to pixels first, so zoom and resize share one coordinate space
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height

        # These sets are small-scale: COCO-Hand hands sit at 0.062 of the frame and
        # HaGRID at 0.163, against ~0.44 for a webcam user. Without the same zoom the
        # EgoHands frames get, 12.5k of the 22k training images would keep teaching the
        # detector to expect distant hands.
        if self.train and ZOOM_PROB > 0 and random.random() < ZOOM_PROB:
            img, boxes = EgoHandsDetection._zoom_in(img, boxes)
            height, width = img.shape[:2]

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        boxes[:, [0, 2]] *= IMAGE_SIZE / width
        boxes[:, [1, 3]] *= IMAGE_SIZE / height

        if self.train and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            x1 = boxes[:, 0].copy()
            boxes[:, 0] = IMAGE_SIZE - boxes[:, 2]
            boxes[:, 2] = IMAGE_SIZE - x1
        if self.train and random.random() < 0.8:
            # np.clip, not cv2.convertScaleAbs: that computes |src*a + b|, so a negative
            # beta reflects dark pixels back up instead of clamping them at zero
            alpha, beta = random.uniform(0.7, 1.3), random.uniform(-30, 30)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, IMAGE_SIZE)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, IMAGE_SIZE)
        keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes = boxes[keep]

        image = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().div_(255.0)
        return image, {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.full((len(boxes),), HAND_LABEL, dtype=torch.int64),
        }


def summarise(dataset, sample=300):
    """Box-count and hand-size statistics, for sanity-checking a freshly parsed set."""
    picks = random.Random(0).sample(range(len(dataset)), min(sample, len(dataset)))
    counts, sizes = [], []
    for i in picks:
        _, target = dataset[i]
        boxes = target["boxes"].numpy()
        counts.append(len(boxes))
        for x1, y1, x2, y2 in boxes:
            sizes.append(max((x2 - x1) / IMAGE_SIZE, (y2 - y1) / IMAGE_SIZE))
    sizes = np.array(sizes) if sizes else np.array([0.0])
    return {
        "images": len(dataset),
        "hands_per_image": float(np.mean(counts)),
        "hand_size_median": float(np.median(sizes)),
        "hand_size_p90": float(np.percentile(sizes, 90)),
    }


def load_coco_hand(root="datasets/COCO-Hand/COCO-Hand-S", train=True):
    """COCO-Hand-S -- 4,534 COCO photos, 10,845 hands.

    Hands annotated onto ordinary COCO images, so whole people, faces, feet and torsos
    are in frame and NOT labelled as hands. That is precisely the signal EgoHands lacks.

    We take COCO-Hand-S rather than COCO-Hand-Big deliberately. Per the dataset's own
    README, Big masks hands its automatic detector missed by painting black circles over
    them, and it keeps images with unreliable detections. Training on that teaches the
    model to expect black discs and, worse, shows it real hands sitting in unlabelled
    background. S is the hand-verified subset with "good and complete annotations".

    ANNOTATION ORDER IS A TRAP. The file is

        image_name, xmin, xmax, ymin, ymax, x1,y1, x2,y2, x3,y3, x4,y4, label

    -- xmin, XMAX, ymin, ymax, not the usual xmin, ymin, xmax, ymax. Read as xyxy it
    yields x2 < x1 on every single row. The four (xi, yi) pairs after it are the
    rotated quadrilateral; we use the axis-aligned extent, which is what the detector
    is trained to predict.
    """
    root = Path(root)
    annotations = root / f"{root.name}_annotations.txt"
    images_dir = root / f"{root.name}_Images"
    if not annotations.is_file():
        raise FileNotFoundError(f"missing {annotations}")

    by_image = {}
    skipped = 0
    for line in annotations.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        name = parts[0]
        try:
            xmin, xmax, ymin, ymax = (float(v) for v in parts[1:5])
        except ValueError:
            skipped += 1
            continue
        if xmax - xmin < 2 or ymax - ymin < 2:
            skipped += 1
            continue
        by_image.setdefault(name, []).append([xmin, ymin, xmax, ymax])

    samples = [(images_dir / name, boxes) for name, boxes in by_image.items()
               if (images_dir / name).is_file()]
    dataset = BoxDataset(samples, train=train, name="COCO-Hand-S")
    dataset.skipped = skipped
    return dataset


def load_hagrid(root="datasets/hagrid-sample-30k-384p", train=True, include_no_gesture=True):
    """HaGRID 30k sample at 384p -- one person, frontal, upper body in frame.

    This is the dataset whose framing IS the deployment shot. EgoHands is a head-mounted
    camera looking down at a table; HaGRID is a person facing a webcam with their head,
    torso and arms visible. That is the distribution our detector actually runs on, and
    it is why HaGRID attacks the face false positive more directly than anything else:
    a face is present in essentially every image and is never labelled a hand.

    Annotation format (per gesture class, keyed by image uuid):

        {"bboxes": [[x, y, w, h], ...],   <- NORMALISED 0-1, top-left + size
         "labels": ["call", "no_gesture", ...]}

    Two things to get right. The boxes are xywh in normalised coordinates, not xyxy in
    pixels, so they need both a conversion and a scale. And `no_gesture` is a real hand
    -- it is the person's other hand, resting -- so it is kept by default. Dropping it
    would put an unlabelled hand in the frame and teach the detector to suppress hands
    that are not gesturing, which is the opposite of what we want.
    """
    root = Path(root)
    ann_dir = root / "ann_train_val"
    img_root = root / "hagrid_30k"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"missing {ann_dir}")

    import json

    samples = []
    missing = 0
    for ann_file in sorted(ann_dir.glob("*.json")):
        gesture = ann_file.stem
        folder = img_root / f"train_val_{gesture}"
        if not folder.is_dir():
            continue
        # only a subset of each class's annotations is present in the 30k sample, so
        # index the folder once rather than stat-ing every annotation key
        present = {p.stem: p for p in folder.glob("*.jpg")}
        entries = json.loads(ann_file.read_text())
        for uuid, record in entries.items():
            path = present.get(uuid)
            if path is None:
                missing += 1
                continue
            labels = record.get("labels") or []
            boxes = []
            for index, (x, y, w, h) in enumerate(record.get("bboxes") or []):
                if not include_no_gesture and index < len(labels) and labels[index] == "no_gesture":
                    continue
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])       # still normalised
            if boxes:
                samples.append((path, boxes))

    dataset = BoxDataset(samples, train=train, name="HaGRID-30k", normalised=True)
    dataset.missing = missing
    return dataset


class GenericNegatives(Dataset):
    """Hand-free crops taken from COCO-Hand / HaGRID images.

    WHY NOT WEBCAM CAPTURES. A negative set recorded from one person at one desk teaches
    "this face, this wall, this lighting is not a hand" -- the model memorises that room
    and still fires on the next person who sits down. The lesson has to be about faces,
    torsos and arms in general, which means it has to come from many people.

    So the negatives are cut from the same third-party images used as positives, from
    regions that contain NO labelled hand. Every crop is a real photograph of a real
    person -- their face, shoulder, forearm, or the room behind them -- drawn from
    thousands of different subjects and settings. That is a lesson that transfers.

    A crop is accepted only if it overlaps no annotated hand box at all. COCO-Hand-S is
    safe for this because its README describes it as the hand-verified subset with
    "good and complete annotations"; COCO-Hand-Big is NOT, because it keeps images whose
    missed hands were painted over, so a "hand-free" crop there might contain a real hand.
    """

    def __init__(self, sources, per_image=1, min_frac=0.25, attempts=24, seed=0, train=True):
        self.items = []
        for source in sources:
            base = source.dataset if isinstance(source, Subset_t) else source
            indices = source.indices if isinstance(source, Subset_t) else range(len(base))
            for i in indices:
                self.items.append((base, i))
        self.per_image = per_image
        self.min_frac = min_frac
        self.attempts = attempts
        self.seed = seed
        self.train = train
        if not self.items:
            raise ValueError("GenericNegatives: no source images")

    def __len__(self):
        return len(self.items) * self.per_image

    def __getitem__(self, index):
        base, i = self.items[index % len(self.items)]
        path, boxes = base.samples[i]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"negative source unreadable: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        if getattr(base, "normalised", False):
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height

        rng = random.Random(self.seed * 1_000_003 + index)
        crop = self._hand_free_window(rng, width, height, boxes)
        if crop is None:
            # No hand-free window of usable size: the hands fill the frame. Returning a
            # blank would teach "flat grey is background", which is worthless, so fall
            # back to an empty ROI and let __len__ absorb the loss.
            patch = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
        else:
            x1, y1, x2, y2 = crop
            patch = cv2.resize(img[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE),
                               interpolation=cv2.INTER_AREA)

        if self.train:
            if rng.random() < 0.5:
                patch = np.ascontiguousarray(patch[:, ::-1])
            if rng.random() < 0.8:
                alpha, beta = rng.uniform(0.7, 1.3), rng.uniform(-30, 30)
                patch = np.clip(patch.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        image = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float().div_(255.0)
        return image, {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    def _hand_free_window(self, rng, width, height, boxes):
        """A random window overlapping no annotated hand, or None if there isn't one."""
        shortest = min(width, height)
        for _ in range(self.attempts):
            size = rng.uniform(self.min_frac, 0.6) * shortest
            size = max(32.0, min(size, shortest))
            x1 = rng.uniform(0, width - size)
            y1 = rng.uniform(0, height - size)
            x2, y2 = x1 + size, y1 + size
            clear = True
            for bx1, by1, bx2, by2 in boxes:
                if x1 < bx2 and bx1 < x2 and y1 < by2 and by1 < y2:
                    clear = False
                    break
            if clear:
                return int(x1), int(y1), int(x2), int(y2)
        return None
