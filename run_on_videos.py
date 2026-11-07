"""
Run stage 1 (hand detection) over a folder of videos and report what it found.

    python run_on_videos.py --videos ../mm-office/MM_Office_Dataset/video/test
    python run_on_videos.py --videos DIR --detector yolo --weights best.pt
    python run_on_videos.py --videos DIR --detector ssdlite --upscale 3 --every 5

For each video it writes an annotated .mp4 and appends to one detections.csv
(video, frame, time, box, score). summary.csv has one row per video: how many sampled
frames had at least one hand, and the mean confidence.

This is for footage WITHOUT hand annotations, so it reports detection rates, not
precision or recall -- a frame with a box is not necessarily a frame with a hand.
Look at the annotated videos before believing any number here.

--upscale exists because surveillance-style footage (e.g. MM-Office, 320x240 from the
corners of a room) puts hands at 5-15 px, far below the ~32 px the detectors resolve.
Enlarging the frame does not add detail, but it does move the hands into the size
range the models were trained on. Boxes are mapped back to original coordinates.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

from pipeline.detector import HandDetector

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder of videos (searched recursively)")
    parser.add_argument("--detector", default="mediapipe", choices=["mediapipe", "yolo", "ssdlite"])
    parser.add_argument("--weights", default=None,
                        help="YOLO .pt or SSDlite .pth; defaults to each backend's usual path")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--max-hands", type=int, default=8)
    parser.add_argument("--every", type=int, default=3, help="process every Nth frame")
    parser.add_argument("--upscale", type=float, default=1.0,
                        help="enlarge frames by this factor before detecting")
    parser.add_argument("--limit", type=int, default=None, help="only the first N videos")
    parser.add_argument("--out", default="video_results")
    args = parser.parse_args()
    if args.every < 1 or args.upscale <= 0:
        parser.error("--every must be >= 1 and --upscale > 0")
    return args


def draw(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, f"{d.score:.2f}", (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    videos = sorted(p for p in Path(args.videos).rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"no videos under {args.videos}")
    videos = videos[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det_file = open(out / "detections.csv", "w", newline="")
    detections_csv = csv.writer(det_file)
    detections_csv.writerow(["video", "frame", "time_s", "x1", "y1", "x2", "y2", "score"])
    summary = []
    started = time.perf_counter()

    for index, path in enumerate(videos, 1):
        # a fresh detector per video: MediaPipe's video mode needs timestamps that only
        # ever increase, and they restart at zero with each file
        detector = HandDetector(backend=args.detector, checkpoint=args.weights,
                                threshold=args.conf, max_hands=args.max_hands)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out / f"{path.stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / args.every, (width, height))

        frame_no = sampled = with_hand = boxes = 0
        score_sum = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % args.every == 0:
                big = frame if args.upscale == 1.0 else cv2.resize(
                    frame, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC)
                found = detector.detect(big, int(frame_no * 1000 / fps))
                for d in found:
                    d.box = tuple(int(round(v / args.upscale)) for v in d.box)
                    detections_csv.writerow([path.name, frame_no, f"{frame_no / fps:.2f}",
                                             *d.box, f"{d.score:.4f}"])
                    score_sum += d.score
                sampled += 1
                boxes += len(found)
                with_hand += bool(found)
                writer.write(draw(frame, found))
            frame_no += 1
        cap.release()
        writer.release()
        detector.close()

        summary.append([path.name, sampled, with_hand,
                        f"{with_hand / max(sampled, 1):.3f}", boxes,
                        f"{score_sum / boxes:.3f}" if boxes else ""])
        print(f"[{index}/{len(videos)}] {path.name}: hands in {with_hand}/{sampled} "
              f"sampled frames, {boxes} boxes")

    det_file.close()
    with open(out / "summary.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["video", "frames_sampled", "frames_with_hand", "hand_frame_rate",
                       "boxes", "mean_score"])
        rows.writerows(summary)

    sampled = sum(r[1] for r in summary)
    with_hand = sum(r[2] for r in summary)
    print(f"\n{len(videos)} videos, {sampled} frames sampled, a hand in {with_hand} "
          f"({with_hand / max(sampled, 1):.1%}), {time.perf_counter() - started:.0f}s")
    print(f"wrote {out}/ (annotated videos, detections.csv, summary.csv)")


if __name__ == "__main__":
    main()
