"""
train_detector.py fine-tunes torchvision's ssdlite320_mobilenet_v3_large on the
EgoHands bounding boxes, producing the checkpoint that webcam_detect.py loads.

SSDlite is picked over a heavier detector (Faster R-CNN) on purpose: it is the one
torchvision detector that comfortably hits real-time on a laptop CPU, which is the
entire point of the live demo. Its backbone starts from COCO-pretrained weights, so
fine-tuning converges in a handful of epochs instead of from scratch.

    python train_detector.py                     # all 48 videos, 10 epochs
    python train_detector.py --epochs 3          # shorter run
    python train_detector.py --limit-videos 4    # smoke test on 4 videos

See also detection_dataset.py, webcam_detect.py
"""

import argparse
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.ops import box_iou
import torch.nn.functional as F
from tqdm import tqdm

from detection_dataset import (
    IMAGE_SIZE,
    NUM_CLASSES,
    EgoHandsDetection,
    NegativeFrames,
    collate_fn,
    split_videos,
)
from get_meta_by import get_meta_by

DEFAULT_CHECKPOINT = "checkpoints/hand_detector.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="confidence cutoff used when scoring the validation set")
    parser.add_argument("--external", action="store_true",
                        help="mix in COCO-Hand-S and HaGRID. EgoHands alone teaches "
                             "'skin blob = hand' because it contains no faces or bodies; "
                             "these supply images where other body parts are visible and "
                             "NOT labelled as hands")
    parser.add_argument("--hagrid-limit", type=int, default=8000,
                        help="cap HaGRID images. It has 31,833, which would swamp "
                             "EgoHands 9:1 and drag training time with it")
    parser.add_argument("--generic-negatives", type=int, default=6000,
                        help="hand-free crops cut from the external images (0 disables). "
                             "These come from thousands of different faces, torsos and "
                             "rooms, so the lesson generalises. A negative set recorded "
                             "from one webcam teaches only that one face and room")
    parser.add_argument("--negatives", default=None,
                        help="folder of hand-free images to train as hard negatives. "
                             "Prefer --generic-negatives: a personal capture overfits to "
                             "one person and one setting")
    parser.add_argument("--negative-repeat", type=int, default=3,
                        help="repeat the negatives this many times. 146 frames against "
                             "3600 positives is only 4%% of the data; repeating lifts "
                             "their share to roughly 11%% so they actually shift the model")
    parser.add_argument("--neg-ratio", type=int, default=None,
                        help="hard negatives mined PER POSITIVE anchor (torchvision "
                             "default 3). The face signal is already in HaGRID images, "
                             "but with only 3 slots per positive the face anchor rarely "
                             "wins a place among thousands of background anchors")
    parser.add_argument("--zoom-min", type=float, default=None,
                        help="most aggressive zoom-in crop, as a fraction of the frame. "
                             "0.35 puts training hands near 0.25 of frame; 0.15 pushes "
                             "them to ~0.45, which is where webcam hands actually sit")
    parser.add_argument("--zoom-prob", type=float, default=None,
                        help="how often the zoom-in crop fires (default 0.5)")
    parser.add_argument("--min-negatives", type=int, default=16,
                        help="hard negatives mined per image that has no positives")
    parser.add_argument("--resume", default=None,
                        help="checkpoint to fine-tune from instead of COCO weights")
    parser.add_argument("--eval-only", action="store_true",
                        help="score an existing checkpoint on the validation set and exit")
    parser.add_argument("--limit-videos", type=int, default=None,
                        help="train on only the first N videos, for a quick smoke test")
    args = parser.parse_args()

    # Validate here rather than letting a nonsense value surface as a confusing
    # failure thousands of frames later.
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error(f"--score-threshold must be in [0, 1], got {args.score_threshold}")
    if args.epochs < 1 and not args.eval_only:
        parser.error(f"--epochs must be at least 1, got {args.epochs}")
    if args.limit_videos is not None and args.limit_videos < 1:
        parser.error(f"--limit-videos must be at least 1, got {args.limit_videos}")
    if args.batch_size < 1:
        parser.error(f"--batch-size must be at least 1, got {args.batch_size}")
    return args


def pick_device(requested):
    """Resolve 'auto' to the best backend actually present on this machine.

    An explicitly requested backend is checked rather than trusted: asking for cuda
    on a Mac should say so immediately, not fail deep inside the first forward pass.
    """
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is not available on this machine")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available on this machine")
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """Load SSDlite and swap in a fresh background/hand classifier.

    pretrained=False skips the COCO download entirely. That matters when rebuilding
    the architecture purely to load a fine-tuned checkpoint over it: every COCO
    weight is about to be overwritten, so fetching 13 MB to discard it just breaks
    offline use for no gain.

    Only the classification head is replaced. The box regression head and the whole
    MobileNetV3 backbone keep their COCO weights -- "where is the object" transfers
    across datasets, "which of 91 COCO classes is it" does not.
    """
    # weights_backbone=None is load-bearing, not tidiness. torchvision derives its
    # `reduce_tail` setting from whether backbone weights were requested, so plain
    # weights=None quietly builds a WIDER network (5,198,540 params instead of
    # 3,440,060) and every checkpoint fails to load with a size mismatch. Pinning
    # weights_backbone=None keeps the architecture identical either way.
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
    )
    in_channels = det_utils.retrieve_out_channels(model.backbone, (IMAGE_SIZE, IMAGE_SIZE))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels,
        num_anchors,
        num_classes,
        partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03),
    )
    return model


def enable_negative_mining(model, min_negatives=16):
    """Make images with zero ground-truth boxes actually contribute to the loss.

    torchvision's SSD samples hard negatives in proportion to positives:

        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)

    For an image with no boxes that product is zero, so no negative anchors are
    sampled and the image yields loss 0.0 and gradient 0.0 -- verified directly.
    Hard-negative frames would train nothing at all.

    torchvision ships the fix immediately below that line, commented out:

        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio

    This restores it with a configurable floor, and also enables the second commented
    line (the isfinite guard) so a positive anchor cannot creep into the negative
    sample. Everything else matches torchvision 0.28's implementation.
    """

    def compute_loss(targets, head_outputs, anchors, matched_idxs):
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for (targets_per_image, bbox_regression_per_image, cls_logits_per_image,
             anchors_per_image, matched_idxs_per_image) in zip(
                targets, bbox_regression, cls_logits, anchors, matched_idxs):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = model.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss.append(torch.nn.functional.smooth_l1_loss(
                bbox_regression_per_image, target_regression, reduction="sum"))

            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none"
        ).view(cls_targets.size())

        foreground_idxs = cls_targets > 0
        num_negative = model.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # THE FIX: mine a floor of hard negatives even when an image has no positives
        num_negative = torch.clamp(num_negative, min=min_negatives)

        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")
        values, idx = negative_loss.sort(1, descending=True)
        background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))

        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    model.compute_loss = compute_loss
    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    """One pass over the training set. Detection models in train mode return a dict
    of losses rather than predictions, so the loss is the sum of those terms."""
    model.train()
    loop = tqdm(loader, desc=f"epoch {epoch}")
    total = 0.0
    for images, targets in loop:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        loop.set_postfix(loss=f"{loss.item():.3f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, score_threshold=0.5, iou_threshold=0.5):
    """Greedy IoU matching -> precision / recall / F1 at a fixed confidence cutoff.

    This is deliberately not COCO mAP, which would mean adding pycocotools. It
    answers the question the webcam demo actually cares about: at the threshold we
    draw boxes at, how many real hands do we find, and how many boxes are junk?
    """
    model.eval()
    true_pos = false_pos = false_neg = 0

    for images, targets in tqdm(loader, desc="  validating", leave=False):
        outputs = model([image.to(device) for image in images])
        for output, target in zip(outputs, targets):
            keep = output["scores"] >= score_threshold
            boxes = output["boxes"][keep].cpu()
            scores = output["scores"][keep].cpu()
            # highest-confidence predictions get first claim on a ground-truth box
            boxes = boxes[scores.argsort(descending=True)]

            truth = target["boxes"]
            claimed = set()
            for box in boxes:
                if len(truth) == 0:
                    false_pos += 1
                    continue
                ious = box_iou(box.unsqueeze(0), truth)[0]
                best = int(ious.argmax())
                if ious[best] >= iou_threshold and best not in claimed:
                    claimed.add(best)
                    true_pos += 1
                else:
                    false_pos += 1
            false_neg += len(truth) - len(claimed)

    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": true_pos, "fp": false_pos, "fn": false_neg}


def main():
    args = parse_args()
    device = pick_device(args.device)

    if args.zoom_min or args.zoom_prob:
        import detection_dataset
        import external_datasets as _ext
        if args.zoom_min:
            detection_dataset.ZOOM_MIN_SCALE = args.zoom_min
        if args.zoom_prob:
            detection_dataset.ZOOM_PROB = args.zoom_prob
            _ext.ZOOM_PROB = args.zoom_prob   # the external loaders read their own copy

    videos = get_meta_by()
    if args.limit_videos:
        videos = videos.iloc[: args.limit_videos]
    train_idx, val_idx = split_videos(videos)

    train_ds = EgoHandsDetection(videos, train_idx, train=True)
    val_ds = EgoHandsDetection(videos, val_idx, train=False)

    external_counts = {}
    if args.external:
        import random as _random
        from external_datasets import load_coco_hand, load_hagrid
        from torch.utils.data import Subset

        extra = []
        coco = load_coco_hand(train=True)
        extra.append(coco)
        external_counts["COCO-Hand-S"] = len(coco)

        hagrid = load_hagrid(train=True)
        if args.hagrid_limit and len(hagrid) > args.hagrid_limit:
            # deterministic subsample, so reruns compare like with like
            picks = _random.Random(0).sample(range(len(hagrid)), args.hagrid_limit)
            hagrid = Subset(hagrid, sorted(picks))
        extra.append(hagrid)
        external_counts["HaGRID"] = len(hagrid)

        if args.generic_negatives:
            from external_datasets import GenericNegatives

            negatives = GenericNegatives([coco, hagrid], per_image=1, train=True)
            if len(negatives) > args.generic_negatives:
                picks = _random.Random(1).sample(range(len(negatives)), args.generic_negatives)
                negatives = Subset(negatives, sorted(picks))
            extra.append(negatives)
            external_counts["generic negatives"] = len(negatives)

        train_ds = ConcatDataset([train_ds] + extra)

    n_negatives = 0
    if args.negatives:
        negatives = NegativeFrames(args.negatives, train=True)
        n_negatives = len(negatives)
        train_ds = ConcatDataset([train_ds] + [negatives] * max(1, args.negative_repeat))

    print(f"device            {device}")
    print(f"videos            {len(videos)}  ->  {len(train_idx)} train / {len(val_idx)} val")
    print(f"frames            {len(train_ds)} train / {len(val_ds)} val")
    for name, count in external_counts.items():
        print(f"external          {count} from {name}")
    if external_counts:
        # validation stays EgoHands-only on purpose: every number in this project so far
        # was measured on it, and changing the yardstick mid-project would make the
        # before/after comparison meaningless
        print(f"                  validation remains EgoHands-only for comparability")
    if n_negatives:
        print(f"hard negatives    {n_negatives} hand-free frames from {args.negatives}")
        print(f"                  repeated x{args.negative_repeat}, "
              f"{100 * n_negatives * args.negative_repeat / len(train_ds):.0f}% of the training set")
        print(f"                  mining {args.min_negatives} negatives per box-free image")
    if not args.eval_only:
        print(f"epochs            {args.epochs} @ batch {args.batch_size}, lr {args.lr}")

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       collate_fn=collate_fn, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    if args.eval_only:
        try:
            state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except FileNotFoundError:
            raise SystemExit(
                f"no checkpoint at {args.checkpoint}. Train one first: python train_detector.py"
            ) from None
        model = build_model(state.get("num_classes", NUM_CLASSES)).to(device)
        model.load_state_dict(state["model"])
        print(f"checkpoint        {args.checkpoint} (epoch {state.get('epoch', '?')})")
        print(f"score threshold   {args.score_threshold}\n")

        metrics = evaluate(model, val_loader, device, args.score_threshold)
        # leading newline: the validation progress bar leaves a partial line behind
        print(f"\ntrue positives    {metrics['tp']}")
        print(f"false positives   {metrics['fp']}")
        print(f"false negatives   {metrics['fn']}")
        print(f"\nprecision         {metrics['precision']:.3f}")
        print(f"recall            {metrics['recall']:.3f}")
        print(f"F1                {metrics['f1']:.3f}")
        return

    model = build_model(pretrained=args.resume is None).to(device)
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        print(f"resumed           {args.resume} "
              f"(epoch {resumed.get('epoch', '?')}, F1 {resumed.get('metrics', {}).get('f1', float('nan')):.3f})")
    # Mining must be enabled for EITHER negative source. torchvision samples hard
    # negatives in proportion to positives, so a box-free image mines none and yields
    # loss 0.0 / grad 0.0 -- the negatives would train nothing at all.
    if args.neg_ratio:
        # Applies to EVERY image, not just box-free ones: it widens how many background
        # anchors compete for the hard-negative slots, which is what gives a
        # high-scoring face anchor a chance to be sampled and penalised.
        model.neg_to_pos_ratio = args.neg_ratio
    if args.negatives or (args.external and args.generic_negatives) or args.neg_ratio:
        enable_negative_mining(model, args.min_negatives)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        metrics = evaluate(model, val_loader, device, args.score_threshold)

        print(
            f"epoch {epoch:>2}/{args.epochs}  loss {loss:6.3f}  "
            f"P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
            f"F1 {metrics['f1']:.3f}  ({time.time() - started:.0f}s)"
        )

        # keep the epoch that generalizes best, not merely the last one
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "metrics": metrics, "num_classes": NUM_CLASSES},
                checkpoint_path,
            )
            print(f"           saved {checkpoint_path} (best F1 so far)")

    print(f"\ndone. best F1 {best_f1:.3f} -> {checkpoint_path}")
    print(f"now run:  python webcam_detect.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
