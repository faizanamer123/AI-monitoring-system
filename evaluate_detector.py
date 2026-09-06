"""
evaluate_detector.py -- rigorous accuracy evaluation of the trained hand detector
against the EgoHands ground truth.

train_detector.py --eval-only reports precision/recall/F1 at ONE score threshold, which
is enough to pick a checkpoint but not enough to know whether the detector is actually
accurate. This does the real thing:

  * Average Precision (COCO-style), which is threshold-independent -- the single number
    that says how good the detector is without you having to pick a cutoff first
  * AP at IoU 0.50 / 0.75 and averaged over 0.50:0.05:0.95, so you can separate
    "finds the hand" from "draws a tight box around it"
  * a sweep of score thresholds, so you can choose the operating point deliberately
  * breakdowns by location, activity and hand size -- an aggregate number hides the
    fact that a detector can be excellent indoors and poor in sunlight
  * failure analysis: what it misses, and what its false positives actually are

Inference runs ONCE and every metric is computed from the cached predictions.

    python evaluate_detector.py                     # val split, full report
    python evaluate_detector.py --split train       # check the generalisation gap
    python evaluate_detector.py --render 12         # also write a visual comparison

See also train_detector.py, detection_dataset.py
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from detection_dataset import IMAGE_SIZE, NUM_CLASSES, EgoHandsDetection, split_videos, video_group
from get_meta_by import get_meta_by
from train_detector import DEFAULT_CHECKPOINT, build_model, pick_device

IOU_SWEEP = np.round(np.arange(0.50, 0.96, 0.05), 2)
SCORE_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# COCO area bands, scaled to the 320x320 space the detector works in
SIZE_BANDS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, 1e9)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N frames")
    parser.add_argument("--render", type=int, default=0,
                        help="write a PNG comparing predictions to ground truth for N frames")
    return parser.parse_args()


@torch.no_grad()
def collect(model, dataset, device, limit=None):
    """Run the detector once over the split and cache every prediction.

    Returns a list of per-frame records. Predictions keep ALL scores -- thresholding
    happens later during metric computation, so one inference pass serves every metric.
    """
    model.eval()
    indices = range(len(dataset)) if limit is None else range(min(limit, len(dataset)))
    records = []
    for i in tqdm(indices, desc="inference", leave=False):
        image, target = dataset[i]
        out = model([image.to(device)])[0]
        records.append({
            "index": i,
            "video": dataset.samples[i][0],
            "gt": target["boxes"].numpy(),
            "boxes": out["boxes"].cpu().numpy(),
            "scores": out["scores"].cpu().numpy(),
        })
    return records


def match(pred_boxes, gt_boxes, iou_threshold):
    """Greedy highest-score-first matching. Returns (is_true_positive per prediction,
    matched flags per ground-truth box).

    Predictions must arrive already sorted by descending score, which is how a
    confident detection gets first claim on a ground-truth box.
    """
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return tp, matched

    ious = box_iou(torch.from_numpy(pred_boxes).float(),
                   torch.from_numpy(gt_boxes).float()).numpy()
    for p in range(len(pred_boxes)):
        candidates = np.where(~matched)[0]
        if len(candidates) == 0:
            break
        best = candidates[np.argmax(ious[p, candidates])]
        if ious[p, best] >= iou_threshold:
            matched[best] = True
            tp[p] = True
    return tp, matched


def average_precision(records, iou_threshold):
    """All-point-interpolated AP over the whole split.

    Every prediction from every frame is pooled and ranked by score, exactly as COCO
    does it. This is what makes AP threshold-free: it integrates precision over the
    entire recall range instead of reporting one point on the curve.
    """
    rows = []          # (score, is_tp)
    total_gt = 0
    for rec in records:
        total_gt += len(rec["gt"])
        order = np.argsort(-rec["scores"])
        boxes, scores = rec["boxes"][order], rec["scores"][order]
        tp, _ = match(boxes, rec["gt"], iou_threshold)
        rows.extend(zip(scores, tp))

    if total_gt == 0:
        return 0.0, np.array([]), np.array([])
    if not rows:
        return 0.0, np.array([0.0]), np.array([0.0])

    rows.sort(key=lambda r: -r[0])
    tp = np.array([r[1] for r in rows], dtype=float)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)

    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # make precision monotonically decreasing, then integrate under the step curve
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    change = np.where(np.diff(np.concatenate(([0.0], recall))) > 0)[0]
    ap = float(np.sum(np.diff(np.concatenate(([0.0], recall)))[change] * envelope[change]))
    return ap, precision, recall


def prf(records, score_threshold, iou_threshold=0.5):
    """Precision / recall / F1 at one operating point."""
    tp = fp = fn = 0
    for rec in records:
        keep = rec["scores"] >= score_threshold
        order = np.argsort(-rec["scores"][keep])
        boxes = rec["boxes"][keep][order]
        is_tp, matched = match(boxes, rec["gt"], iou_threshold)
        tp += int(is_tp.sum())
        fp += int((~is_tp).sum())
        fn += int((~matched).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), tp, fp, fn


def subset_ap(records, keyfn, iou_threshold=0.5):
    """AP computed independently within each slice of the data."""
    groups = defaultdict(list)
    for rec in records:
        groups[keyfn(rec)].append(rec)
    return {k: (average_precision(v, iou_threshold)[0], sum(len(r["gt"]) for r in v))
            for k, v in sorted(groups.items())}


def size_recall(records, score_threshold=0.5, iou_threshold=0.5):
    """Recall broken down by ground-truth box area -- small hands are the hard case."""
    found = defaultdict(int)
    total = defaultdict(int)
    for rec in records:
        keep = rec["scores"] >= score_threshold
        order = np.argsort(-rec["scores"][keep])
        _, matched = match(rec["boxes"][keep][order], rec["gt"], iou_threshold)
        for box, hit in zip(rec["gt"], matched):
            area = (box[2] - box[0]) * (box[3] - box[1])
            for name, lo, hi in SIZE_BANDS:
                if lo <= area < hi:
                    total[name] += 1
                    found[name] += int(hit)
                    break
    return {n: (found[n], total[n]) for n, _, _ in SIZE_BANDS if total[n]}


def render_comparison(records, dataset, path, count):
    """Write a grid of frames with ground truth in green and predictions in red.

    Numbers tell you how accurate the detector is; this tells you HOW it is wrong --
    whether it misses hands, splits one into two, or boxes the wrong thing entirely.
    """
    import cv2

    picks = np.linspace(0, len(records) - 1, min(count, len(records)), dtype=int)
    tiles = []
    for i in picks:
        rec = records[i]
        image, _ = dataset[rec["index"]]
        frame = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
        for x1, y1, x2, y2 in rec["gt"]:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        keep = rec["scores"] >= 0.5
        for (x1, y1, x2, y2), s in zip(rec["boxes"][keep], rec["scores"][keep]):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{s:.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        tiles.append(frame)

    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.putText(grid, "green = ground truth   red = prediction (score >= 0.5)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, grid)
    return path


def main():
    args = parse_args()
    device = pick_device(args.device)

    videos = get_meta_by()
    train_idx, val_idx = split_videos(videos)
    chosen = val_idx if args.split == "val" else train_idx
    dataset = EgoHandsDetection(videos, chosen, train=False)

    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise SystemExit(f"no checkpoint at {args.checkpoint}. Train one: python train_detector.py")
    model = build_model(state.get("num_classes", NUM_CLASSES)).to(device)
    model.load_state_dict(state["model"])

    print(f"checkpoint   {args.checkpoint}  (epoch {state.get('epoch', '?')})")
    print(f"split        {args.split}: {len(chosen)} videos, {len(dataset)} frames")
    print(f"device       {device}\n")

    records = collect(model, dataset, device, args.limit)
    n_gt = sum(len(r["gt"]) for r in records)
    n_pred = sum(len(r["boxes"]) for r in records)
    print(f"frames evaluated {len(records)}   ground-truth hands {n_gt}   raw detections {n_pred}\n")

    # ---- headline: threshold-independent accuracy ----
    aps = {float(t): average_precision(records, float(t))[0] for t in IOU_SWEEP}
    print("=" * 62)
    print("AVERAGE PRECISION  (threshold-free -- the headline accuracy number)")
    print("=" * 62)
    print(f"  AP@0.50           {aps[0.5]:.4f}")
    print(f"  AP@0.75           {aps[0.75]:.4f}")
    print(f"  AP@[0.50:0.95]    {np.mean(list(aps.values())):.4f}   (COCO primary metric)")
    print("\n  AP by IoU threshold:")
    for t in IOU_SWEEP:
        bar = "#" * int(aps[float(t)] * 50)
        print(f"    IoU {t:.2f}  {aps[float(t)]:.4f}  {bar}")

    # ---- operating points ----
    print("\n" + "=" * 62)
    print("SCORE THRESHOLD SWEEP  (IoU 0.50) -- pick your operating point")
    print("=" * 62)
    print(f"  {'score':>6} {'precision':>10} {'recall':>8} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    best = (0.0, None)
    for s in SCORE_SWEEP:
        p, r, f1, tp, fp, fn = prf(records, s)
        star = ""
        if f1 > best[0]:
            best = (f1, s)
        print(f"  {s:>6.2f} {p:>10.3f} {r:>8.3f} {f1:>7.3f} {tp:>6} {fp:>6} {fn:>6}{star}")
    print(f"\n  best F1 {best[0]:.3f} at score threshold {best[1]}")

    # ---- where it works and where it does not ----
    print("\n" + "=" * 62)
    print("BREAKDOWN  (AP@0.50 within each slice)")
    print("=" * 62)
    by_loc = subset_ap(records, lambda r: video_group(videos.iloc[r["video"]])[1])
    by_act = subset_ap(records, lambda r: video_group(videos.iloc[r["video"]])[0])
    for title, table in (("location", by_loc), ("activity", by_act)):
        print(f"\n  by {title}:")
        for k, (ap, gt) in sorted(table.items(), key=lambda kv: -kv[1][0]):
            print(f"    {k:<12} AP {ap:.4f}   ({gt} hands)")

    print("\n  recall by hand size (score 0.50, IoU 0.50):")
    for name, (found, total) in size_recall(records).items():
        print(f"    {name:<8} {found:>5}/{total:<5} = {found/total:.3f}")

    # ---- what it gets wrong ----
    print("\n" + "=" * 62)
    print("FAILURE ANALYSIS  (at score 0.50, IoU 0.50)")
    print("=" * 62)
    p, r, f1, tp, fp, fn = prf(records, 0.5)
    perfect = 0
    for rec in records:
        keep = rec["scores"] >= 0.5
        boxes = rec["boxes"][keep][np.argsort(-rec["scores"][keep])]
        is_tp, matched = match(boxes, rec["gt"], 0.5)
        # a clean frame: every ground-truth hand matched, and nothing extra predicted
        if matched.all() and is_tp.all() and len(boxes) == len(rec["gt"]):
            perfect += 1

    print(f"  frames with every hand found and no extras : {perfect}/{len(records)} "
          f"({100*perfect/max(len(records),1):.1f}%)")
    print(f"  hands missed entirely (false negatives)    : {fn}")
    print(f"  spurious boxes (false positives)           : {fp}")
    print(f"  precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")

    if args.render:
        path = render_comparison(records, dataset, "evaluation_examples.png", args.render)
        print(f"\n  wrote {path}")

    print("\n" + "=" * 62)
    verdict = ("ACCURATE" if aps[0.5] >= 0.75 else
               "USABLE"   if aps[0.5] >= 0.55 else "WEAK")
    print(f"VERDICT: AP@0.50 = {aps[0.5]:.4f} -> {verdict}")
    print("=" * 62)


if __name__ == "__main__":
    main()
