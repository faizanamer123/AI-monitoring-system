"""
Build a YOLO-format hand-detection dataset from every source we have.

    python export_yolo.py                 # EgoHands + COCO-Hand + HaGRID
    python export_yolo.py --out yolo_hands --val-frac 0.15

WHY A SEPARATE EXPORT STEP. Ultralytics wants images and label files on disk in its own
layout, not a PyTorch Dataset. Rather than convert on the fly, this writes the layout
once and SYMLINKS the images, so nothing is copied -- the export costs a few megabytes
of text instead of ~10 GB of duplicated JPEGs.

    <out>/
      images/train/*.jpg -> symlinks
      images/val/*.jpg
      labels/train/*.txt      one line per hand: "0 cx cy w h", all normalised 0-1
      labels/val/*.txt
      data.yaml

THE SPLIT IS BY SOURCE GROUPING, NOT BY IMAGE. EgoHands frames are split by VIDEO: the
100 labelled frames in one clip are near-duplicates, so a random image split would leak
validation into training and report an accuracy that means nothing. HaGRID is split by
its gesture folders for the same reason -- the same person appears many times within a
class.

ONE CLASS. Everything is "hand". The point of mixing COCO-Hand, HaGRID and BodyHands in
is that those images contain faces, torsos, arms and feet which are NOT labelled -- so
the model is forced to learn hand vs. not-hand instead of the shortcut EgoHands alone
teaches it ("skin-coloured blob of about this size = hand").
"""

import argparse
import random
from pathlib import Path

import numpy as np


# HaGRID's 18 gestures, plus the resting hand. Order is fixed and alphabetical so a
# class index means the same thing across every export and every trained checkpoint --
# a reordered list would silently relabel an existing model's predictions.
GESTURES = (
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm",
    "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2",
    "two_up", "two_up_inverted",
)
NO_GESTURE = "no_gesture"
GESTURE_CLASSES = GESTURES + (NO_GESTURE,)
GESTURE_INDEX = {name: i for i, name in enumerate(GESTURE_CLASSES)}


def xyxy_to_yolo(box, width, height):
    """xyxy -> normalised centre-x, centre-y, width, height, clipped to the frame.

    Works for pixel coordinates (pass the real width/height) and for already-normalised
    ones (pass 1.0, 1.0). The degenerate-box threshold is expressed as a FRACTION of the
    frame rather than in pixels, because a fixed `< 1` test silently rejects every
    normalised box -- 0.05 is less than 1, so an entire dataset vanishes.
    """
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    min_w, min_h = 0.002 * width, 0.002 * height
    if x2 - x1 < min_w or y2 - y1 < min_h:
        return None
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
            (x2 - x1) / width, (y2 - y1) / height)


def egohands_samples():
    """(image_path, [xyxy...], group) for every EgoHands frame. Group = video id."""
    import cv2
    from get_bounding_boxes import get_bounding_boxes
    from get_frame_path import get_frame_path
    from get_meta_by import get_meta_by

    videos = get_meta_by()
    out = []
    for vi in range(len(videos)):
        video = videos.iloc[vi]
        group = str(video.loc["video_id"][0])
        frames = video.loc["labelled_frames"][0]
        for fi in range(len(frames)):
            path = Path(get_frame_path(video, fi))
            if not path.is_file():
                continue
            boxes = [[x, y, x + w - 1, y + h - 1]
                     for x, y, w, h in get_bounding_boxes(video, fi) if w > 1 and h > 1]
            if boxes:
                out.append((path, boxes, group))
    return out


def coco_hand_samples(root="datasets/COCO-Hand/COCO-Hand-S", gestures=False):
    """COCO-Hand-S. Columns are xmin,XMAX,ymin,ymax -- not the usual xyxy order."""
    root = Path(root)
    annotations = root / f"{root.name}_annotations.txt"
    images = root / f"{root.name}_Images"
    if not annotations.is_file():
        return []
    by_image = {}
    for line in annotations.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        try:
            xmin, xmax, ymin, ymax = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if xmax - xmin < 2 or ymax - ymin < 2:
            continue
        by_image.setdefault(parts[0], []).append([xmin, ymin, xmax, ymax])
    # One group PER IMAGE. These are independent COCO photos with no near-duplicates,
    # so a per-image split is sound -- and grouping them all under one label made the
    # holdout all-or-nothing, dumping 4,534 images into val in a single draw.
    if gestures:
        # COCO-Hand has no gesture labels, and its hands are not making any of the 18.
        # They become no_gesture -- which keeps the property that mattered most about
        # this dataset: faces, torsos and feet are in frame and stay unlabelled.
        idx = GESTURE_INDEX[NO_GESTURE]
        return [(images / name, [(b, idx) for b in boxes], f"coco-{name}")
                for name, boxes in by_image.items() if (images / name).is_file()]
    return [(images / name, boxes, f"coco-{name}")
            for name, boxes in by_image.items() if (images / name).is_file()]


def hagrid_samples(root="datasets/hagrid-30k/hagrid-sample-30k-384p", gestures=False):
    """HaGRID. Boxes are NORMALISED [x, y, w, h], so they need the image size."""
    import json

    root = Path(root)
    ann_dir = root / "ann_train_val"
    img_root = root / "hagrid_30k"
    if not ann_dir.is_dir():
        return []
    out = []
    for ann_file in sorted(ann_dir.glob("*.json")):
        gesture = ann_file.stem
        folder = img_root / f"train_val_{gesture}"
        if not folder.is_dir():
            continue
        present = {p.stem: p for p in folder.glob("*.jpg")}
        for uuid, record in json.loads(ann_file.read_text()).items():
            path = present.get(uuid)
            if path is None:
                continue
            # keep no_gesture boxes: that is the person's resting other hand, and
            # dropping it would leave a real hand unlabelled in the frame
            raw = record.get("bboxes") or []
            labels = record.get("labels") or []
            if gestures:
                # labels is index-aligned with bboxes, so each hand carries its own
                # gesture. A two-hand frame is typically ["call", "no_gesture"] -- the
                # resting hand MUST be kept and labelled, or the model learns that a
                # real hand is background.
                boxes = []
                for i, (x, y, w, h) in enumerate(raw):
                    if w <= 0 or h <= 0:
                        continue
                    name = labels[i] if i < len(labels) else NO_GESTURE
                    boxes.append(([x, y, x + w, y + h],
                                  GESTURE_INDEX.get(name, GESTURE_INDEX[NO_GESTURE])))
            else:
                boxes = [[x, y, x + w, y + h]
                         for x, y, w, h in raw if w > 0 and h > 0]
            if boxes:
                # Group by user_id, not by gesture folder: the same person appears
                # across many gestures, so a gesture split would put the same face on
                # both sides while a user split genuinely holds people out.
                user = record.get("user_id") or gesture
                out.append((path, boxes, f"hagrid-{user}", True))
    return out


def bodyhands_samples(root="datasets/bodyhands/Data/VOC2007", two_class=False):
    """BodyHands (CVPR 2022) -- Pascal VOC, and the only source that labels BODIES.

    Each object carries a class (`hand` or `body`) and a `body_id` linking a hand to the
    person it belongs to. That makes it the most direct attack on our failure mode: the
    detector fires on faces, elbows and feet because EgoHands never showed it a body that
    was not a hand. Here bodies are annotated explicitly.

    two_class=False -- export only hands, so bodies become unlabelled background and act
        as implicit hard negatives in the same frame as a real hand.
    two_class=True  -- export bodies as class 1 as well, forcing the model to actively
        DISCRIMINATE rather than merely suppress. At inference class 1 is ignored.

    Note the images are named train_*/test_* and the ImageSets/Main lists define the
    official split, so we group by that rather than inventing our own.
    """
    import xml.etree.ElementTree as ET

    root = Path(root)
    ann_dir, img_dir = root / "Annotations", root / "JPEGImages"
    if not ann_dir.is_dir():
        return []

    out = []
    for xml_path in sorted(ann_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        image = img_dir / f"{xml_path.stem}.jpg"
        if not image.is_file():
            continue
        boxes = []
        for obj in tree.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in ("hand", "body"):
                continue
            if name == "body" and not two_class:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
                x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append(([x1, y1, x2, y2], 0 if name == "hand" else 1))
        if boxes:
            # split on the dataset's own train/test prefix
            group = "bodyhands-test" if xml_path.stem.startswith("test") else f"bh-{xml_path.stem}"
            out.append((image, boxes, group))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="yolo_hands")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--hagrid-limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["egohands", "coco", "hagrid", "bodyhands"])
    parser.add_argument("--gestures", action="store_true",
                        help="export HaGRID's 18 gesture classes plus no_gesture (19 "
                             "classes) instead of a single `hand` class. COCO-Hand and "
                             "EgoHands hands become no_gesture, which keeps their "
                             "unlabelled faces and bodies as negative signal")
    parser.add_argument("--bodyhands", action="store_true",
                        help="include BodyHands (20,490 images labelling hands AND bodies)")
    parser.add_argument("--two-class", action="store_true",
                        help="with --bodyhands, export bodies as class 1 so the model must "
                             "actively discriminate hand from body rather than just suppress")
    args = parser.parse_args()

    import cv2

    sources = []
    if "egohands" not in args.skip:
        ego = egohands_samples()
        if args.gestures:
            # EgoHands hands are manipulating objects, not making any of the 18 gestures
            idx = GESTURE_INDEX[NO_GESTURE]
            ego = [(p, [(b, idx) for b in bx], g) for p, bx, g in ego]
        sources += [(p, b, g, False) for p, b, g in ego]
    if "coco" not in args.skip:
        sources += [(p, b, g, False) for p, b, g in coco_hand_samples(gestures=args.gestures)]
    if "hagrid" not in args.skip:
        hag = hagrid_samples(gestures=args.gestures)
        if args.hagrid_limit and len(hag) > args.hagrid_limit:
            hag = random.Random(args.seed).sample(hag, args.hagrid_limit)
        sources += hag
    if args.bodyhands and "bodyhands" not in args.skip:
        bh = bodyhands_samples(two_class=args.two_class)
        sources += [(p, b, g, False) for p, b, g in bh]
        print(f"bodyhands          {len(bh)} images")
    if not sources:
        raise SystemExit("no sources found -- check datasets/ and _LABELLED_SAMPLES/")

    # split by GROUP so near-duplicate frames never straddle the split
    groups = sorted({s[2] for s in sources})
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_frac))
    val_groups = set(groups[:n_val])

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    boxes_written = 0
    skipped = 0
    for index, (path, boxes, group, normalised) in enumerate(sources):
        split = "val" if group in val_groups else "train"
        if normalised:
            width = height = 1.0          # already fractions of the image
        else:
            image = cv2.imread(str(path))
            if image is None:
                skipped += 1
                continue
            height, width = image.shape[:2]

        lines = []
        for entry in boxes:
            # A source yields either a bare box (everything is a hand) or a
            # (box, class) pair -- only BodyHands with --two-class does the latter.
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
                box, cls = entry
            else:
                box, cls = entry, 0
            yolo = xyxy_to_yolo(box, width, height)
            if yolo is None:
                continue
            cx, cy, w, h = (float(np.clip(v, 0.0, 1.0)) for v in yolo)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines:
            skipped += 1
            continue

        stem = f"{group}_{index:06d}".replace("/", "_")
        link = out / "images" / split / f"{stem}{path.suffix}"
        if not link.exists():
            link.symlink_to(path.resolve())
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1
        boxes_written += len(lines)

    (out / "data.yaml").write_text(
        f"# Hand detection, one class. Generated by export_yolo.py\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        + (f"nc: {len(GESTURE_CLASSES)}\nnames: [{', '.join(GESTURE_CLASSES)}]\n"
           if args.gestures else
           f"nc: 2\nnames: [hand, body]\n" if args.two_class else
           f"nc: 1\nnames: [hand]\n")
    )

    if args.gestures:
        import collections
        hist = collections.Counter()
        for _, bx, _, _ in sources:
            for entry in bx:
                if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], int):
                    hist[GESTURE_CLASSES[entry[1]]] += 1
        print("class histogram (boxes):")
        for name, n in hist.most_common():
            print(f"  {name:<18} {n:>6}")
        print()

    print(f"sources            {len(sources)}")
    print(f"groups             {len(groups)}  ({len(val_groups)} held out for val)")
    print(f"images written     {counts['train']} train / {counts['val']} val")
    print(f"boxes written      {boxes_written}")
    print(f"skipped            {skipped} (unreadable or no usable box)")
    print(f"\nwrote {out}/data.yaml")


if __name__ == "__main__":
    main()
