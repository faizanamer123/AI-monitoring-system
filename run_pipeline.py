"""
The full hand pipeline, live.

    WEBCAM -> OpenCV captures each frame
        -> 1. DETECT      where is the hand?        box + score
        -> 2. SEGMENT     which pixels are hand?    mask
        -> 3. POSE        21 landmarks              fingertip coordinates
        -> 4. TRACK       frame to frame            id, trajectory, velocity
        -> gesture / movement

    python run_pipeline.py                        # live webcam
    python run_pipeline.py --record session.mp4   # live, and save the video
    python run_pipeline.py --video session.mp4    # replay a recording
    python run_pipeline.py --detector ssdlite     # use our EgoHands model instead
    python run_pipeline.py --show mask,skeleton   # choose overlays

Press q or Esc to quit.

WHY RECORD AND REPLAY. Stage 4 is about motion, and motion cannot be evaluated on still
images. EgoHands cannot help here either: its 100 "labelled frames" per video are sampled
roughly 0.7 seconds apart (median gap 21 frames at 30 fps), so consecutive annotations are
not consecutive in time and frame-to-frame association is meaningless on them. Recording
your own sequences gives real temporal data, and replaying a recording makes tracking
results reproducible instead of depending on whatever you did in front of the camera.

macOS: grant camera access to your terminal under
System Settings -> Privacy & Security -> Camera, then restart it.
"""

import argparse
import time
from collections import deque

import cv2

from pipeline.detector import HandDetector
from pipeline.mediapipe_hands import DEFAULT_MODEL, MediaPipeHands
from pipeline.pose import PoseEstimator

ALL_OVERLAYS = ("mask", "skeleton", "box", "trail", "label")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", help="replay a video file instead of the camera")
    parser.add_argument("--record", help="write the annotated session to this .mp4")
    parser.add_argument("--detector", default="mediapipe", choices=["mediapipe", "ssdlite"])
    parser.add_argument("--checkpoint", default="checkpoints/hand_detector.pth")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--segmentation", default="hybrid",
                        choices=["hybrid", "landmark", "skin", "none"])
    parser.add_argument("--show", default=",".join(ALL_OVERLAYS),
                        help="comma-separated: " + ",".join(ALL_OVERLAYS))
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--headless", type=float, default=None, metavar="SECONDS",
                        help="run without a window for N seconds, printing stats (for testing)")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error(f"--confidence must be in [0, 1], got {args.confidence}")
    if args.max_hands < 1:
        parser.error(f"--max-hands must be at least 1, got {args.max_hands}")
    args.show = tuple(s.strip() for s in args.show.split(",") if s.strip())
    return args


def build_stages(args):
    """Assemble the four stages. Stage 1 and 3 share one MediaPipe runner (halves cost)."""
    from pipeline.segmenter import HandSegmenter
    from pipeline.tracker import HandTracker

    video_mode = True
    shared = MediaPipeHands(args.model, args.max_hands, args.confidence, video_mode)

    detector = HandDetector(
        backend=args.detector, model_path=args.model, checkpoint=args.checkpoint,
        threshold=args.confidence, max_hands=args.max_hands, shared=shared,
        video_mode=video_mode,
    )
    pose = PoseEstimator(shared=shared)
    segmenter = None if args.segmentation == "none" else HandSegmenter(method=args.segmentation)
    tracker = HandTracker()
    return detector, pose, segmenter, tracker, shared


def open_source(args):
    """Camera or video file, plus the frame size we end up with."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"could not open video {args.video}")
        print(f"replaying {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(
                f"could not open camera {args.camera}. On macOS grant camera access in "
                "System Settings -> Privacy & Security -> Camera, then restart your terminal."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"camera {args.camera} open at "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def main():
    args = parse_args()
    from pipeline.gestures import classify
    from pipeline.visualize import render

    detector, pose_est, segmenter, tracker, shared = build_stages(args)
    cap = open_source(args)
    writer = None

    recent = deque(maxlen=30)
    started = time.perf_counter()
    frames = 0
    window = "hand pipeline  (q to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.video and not args.no_mirror:
                # mirror before everything, so drawn results match what is on screen
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            stamp_ms = int((now - started) * 1000)
            tick = now

            detections = detector.detect(frame, stamp_ms)                    # 1
            poses = pose_est.estimate(frame, detections, stamp_ms)           # 3
            masks = ([segmenter.segment(frame, d, p) for d, p in zip(detections, poses)]
                     if segmenter else [None] * len(detections))             # 2
            tracks = tracker.update(detections, poses, masks, timestamp=now) # 4

            for track in tracks:
                if track.pose is not None:
                    track.gesture, track.gesture_confidence = classify(track.pose, track.detection)

            recent.append(time.perf_counter() - tick)
            fps = len(recent) / max(sum(recent), 1e-6)
            frame = render(frame, tracks, fps, show=args.show)
            frames += 1

            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             20.0, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            if args.headless is not None:
                if time.perf_counter() - started > args.headless:
                    break
                continue

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"wrote {args.record}")
        if args.headless is None:
            cv2.destroyAllWindows()
        shared.close()

    elapsed = time.perf_counter() - started
    print(f"\n{frames} frames in {elapsed:.1f}s -> {frames/max(elapsed,1e-6):.1f} fps end-to-end")
    if recent:
        print(f"pipeline stages only: {len(recent)/sum(recent):.1f} fps")


if __name__ == "__main__":
    main()
