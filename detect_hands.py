"""
Simple hand detection: a box around each hand. Nothing else.

    python detect_hands.py                     # webcam
    python detect_hands.py --image photo.jpg   # a single image
    python detect_hands.py --video clip.mp4    # a video file
    python detect_hands.py --conf 0.4          # stricter (fewer, surer boxes)

Press q or Esc to quit.

This is deliberately the whole program. No segmentation mask, no 21-point skeleton, no
tracking, no gestures -- run_pipeline.py does all of that if you want it. Here a frame
goes to YOLO, boxes come back, OpenCV draws them. That is the entire loop, and it runs
at ~80 fps because nothing else is competing for the GPU.

The detector is a YOLO11n trained on EgoHands + COCO-Hand + HaGRID. The last two matter:
they contain faces, torsos and feet that are NOT labelled as hands, which is what stops
the model calling any skin-coloured blob a hand.
"""

import argparse
import time
from collections import deque

import cv2

DEFAULT_WEIGHTS = "checkpoints/hand_yolo.pt"
BOX_COLOR = (0, 220, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", help="run on one image instead of the camera")
    parser.add_argument("--video", help="run on a video file instead of the camera")
    parser.add_argument("--folder", help="run on every image under this folder (recursive)")
    parser.add_argument("--out", default="detect_hands_results",
                        help="with --folder: where annotated images and detections.csv go")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--save", help="write the annotated output here")
    parser.add_argument("--headless", type=float, default=None, metavar="SECONDS",
                        help="run without a window for N seconds and print stats")
    args = parser.parse_args()
    if not 0.0 <= args.conf <= 1.0:
        parser.error(f"--conf must be in [0, 1], got {args.conf}")
    return args


def yolo_device(requested="auto"):
    """Resolve 'auto' to what this machine has, spelled the way ultralytics wants it.

    Ultralytics takes "0" for the first GPU, not "cuda" -- it would treat "cuda" as a
    device index and fail.
    """
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "0" if requested == "cuda" else requested


def load(weights, device):
    from pathlib import Path

    if not Path(weights).is_file():
        raise SystemExit(
            f"no weights at {weights}\n"
            "Train some first:  python export_yolo.py && python train_yolo.py"
        )
    from ultralytics import YOLO

    model = YOLO(weights)
    model.to(device)
    return model


def detect(model, frame, conf, max_hands, device):
    """Frame in, list of (x1, y1, x2, y2, score) out. That is the whole detector."""
    # verbose=False, or ultralytics prints a line for every frame
    # agnostic_nms matters once the model has more than one class: ultralytics puts
    # each class in its own NMS coordinate band, so without it the same hand returns
    # once per plausible class instead of once.
    result = model.predict(frame, conf=conf, verbose=False, agnostic_nms=True,
                           device=device)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = getattr(model, "names", {}) or {}
    out = [
        (int(x1), int(y1), int(x2), int(y2), float(s), str(names.get(int(c), "hand")))
        for (x1, y1, x2, y2), s, c in zip(boxes.xyxy.cpu().numpy(),
                                          boxes.conf.cpu().numpy(),
                                          boxes.cls.cpu().numpy())
    ]
    out.sort(key=lambda b: -b[4])
    return out[:max_hands]


def draw(frame, hands, fps=None):
    for x1, y1, x2, y2, score, name in hands:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # filled strip behind the text, so it stays readable over a busy background
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if fps is not None:
        hud = f"{fps:4.1f} fps   hands: {len(hands)}   q to quit"
        for thickness, color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, thickness, cv2.LINE_AA)
    return frame


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_folder(model, args):
    """Every image under --folder -> annotated copy + one CSV row per detected hand."""
    import csv
    from pathlib import Path

    root, out = Path(args.folder), Path(args.out)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found under {root}")

    total_hands = unreadable = 0
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "detections.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["image", "x1", "y1", "x2", "y2", "confidence"])
        for n, path in enumerate(images, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                unreadable += 1
                continue
            hands = detect(model, frame, args.conf, args.max_hands, args.device)
            relative = path.relative_to(root)
            for hand in hands:
                rows.writerow([relative.as_posix(), *hand[:4], f"{hand[4]:.4f}"])
            total_hands += len(hands)
            # keep the folder structure, so same-named files in subfolders don't collide
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), draw(frame, hands))
            if n % 50 == 0 or n == len(images):
                print(f"  {n}/{len(images)} images, {total_hands} hands so far")

    elapsed = time.perf_counter() - started
    print(f"\n{len(images) - unreadable} images, {total_hands} hands, {elapsed:.1f}s")
    if unreadable:
        print(f"skipped {unreadable} unreadable image(s)")
    print(f"wrote {out}/ (annotated images + detections.csv)")


def main():
    args = parse_args()
    args.device = yolo_device(args.device)
    model = load(args.weights, args.device)

    if args.folder:
        run_folder(model, args)
        return

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
        hands = detect(model, frame, args.conf, args.max_hands, args.device)
        print(f"{len(hands)} hand(s) in {args.image}")
        for x1, y1, x2, y2, score, name in hands:
            print(f"  {name:<16} {score:.2f}  box ({x1}, {y1}, {x2}, {y2})")
        out = args.save or "detect_hands_output.jpg"
        cv2.imwrite(out, draw(frame, hands))
        print(f"wrote {out}")
        return

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open {'video ' + args.video if args.video else f'camera {args.camera}'}.\n"
            "On macOS grant camera access in System Settings -> Privacy & Security -> Camera."
        )
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"camera {args.camera} at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
              f"x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    writer = None
    recent = deque(maxlen=30)
    started = time.perf_counter()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # mirror before detecting, so the boxes match what is on screen
            if not args.video and not args.no_mirror:
                frame = cv2.flip(frame, 1)

            tick = time.perf_counter()
            hands = detect(model, frame, args.conf, args.max_hands, args.device)
            recent.append(time.perf_counter() - tick)
            frame = draw(frame, hands, len(recent) / max(sum(recent), 1e-6))
            frames += 1

            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             20.0, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            if args.headless is not None:
                if time.perf_counter() - started > args.headless:
                    break
                continue

            cv2.imshow("hand detection  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.save}")
        if args.headless is None:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"\n{frames} frames in {elapsed:.1f}s -> {frames/max(elapsed,1e-6):.1f} fps end-to-end")
    if recent:
        print(f"detection only: {len(recent)/sum(recent):.1f} fps")


if __name__ == "__main__":
    main()
