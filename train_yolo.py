"""
Train a YOLO hand detector on the exported dataset.

    python export_yolo.py          # build yolo_hands/ first
    python train_yolo.py           # then train
    python train_yolo.py --model yolo11s.pt --epochs 30

WHY YOLO RATHER THAN THE SSDlite IN train_detector.py.

The SSDlite reached AP@0.50 = 0.9335 on EgoHands and still boxed faces, elbows and feet
on a webcam, because EgoHands contains no faces to learn from and its hands sit at
0.207 of the frame while a webcam user's sit at 0.439. Two things had to change: the
DATA (hence COCO-Hand and HaGRID, where faces and bodies appear unlabelled) and the
SCALE COVERAGE.

Ultralytics addresses the second for free. Mosaic augmentation tiles four images into
one, so hands appear at a wide spread of sizes in every batch, and `scale` jitters the
zoom on top of that. Getting the equivalent out of the SSDlite pipeline took a custom
log-uniform zoom crop; here it is a default.

`scale` is raised from its 0.5 default because even mosaic centres on the training
distribution, and ours still sits below webcam scale.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="yolo_hands/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="yolo11n.pt is fastest; yolo11s.pt is more accurate and ~3x slower")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="hands")
    parser.add_argument("--scale", type=float, default=0.9,
                        help="zoom jitter (ultralytics default 0.5). Raised because our "
                             "training hands sit at ~0.16 of frame and a webcam user's "
                             "at ~0.44")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not Path(args.data).is_file():
        raise SystemExit(f"{args.data} not found -- run: python export_yolo.py")
    return args


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        # augmentation aimed squarely at the scale gap
        scale=args.scale,
        mosaic=1.0,          # four images per tile -> many hand sizes per batch
        # close_mosaic defaults to 10, meaning "turn mosaic off for the last 10
        # epochs". With epochs<=10 that fires at epoch 0 and mosaic NEVER RUNS -- while
        # args.yaml still records mosaic: 1.0, so the run's own config says it was on.
        # Keep it on for all but the final epoch or two.
        close_mosaic=max(1, min(2, args.epochs - 1)),
        fliplr=0.5,
        flipud=0.0,          # hands are not seen upside down in any of our sources
        degrees=10.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # lighting variety, since rooms differ
        patience=20,
        plots=True,
        val=True,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print("\n=== validation ===")
    print(f"  mAP50     {metrics.box.map50:.4f}")
    print(f"  mAP50-95  {metrics.box.map:.4f}")
    print(f"  precision {metrics.box.mp:.4f}")
    print(f"  recall    {metrics.box.mr:.4f}")
    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    print(f"run it with:  python run_pipeline.py --detector yolo --yolo-weights {best}")


if __name__ == "__main__":
    main()
