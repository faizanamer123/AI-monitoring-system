"""
webcam_detect_mp.py -- live hand box detection with MediaPipe.

Division of labour, exactly as intended:

    OpenCV                          MediaPipe
    ------                          ---------
    opening the webcam              finding the hand
    reading frames                  predicting the hand box
    colour conversion               detection confidence + handedness
    drawing bounding boxes
    displaying the live result

Usage:
    python webcam_detect_mp.py                   # live camera
    python webcam_detect_mp.py --image x.png     # test on a still
    python webcam_detect_mp.py --max-hands 2
    python webcam_detect_mp.py --landmarks       # also draw the 21 joints

Press q or Esc to quit.

NOTE ON THE API: mediapipe 1.x REMOVED the old `mp.solutions.hands` interface that
almost every tutorial online still uses. This file uses the current Tasks API
(mp.tasks.vision.HandLandmarker), which needs the model file in models/.

macOS: grant camera access to your terminal in
System Settings -> Privacy & Security -> Camera, then restart the terminal.

See also webcam_detect.py (EgoHands-trained SSDlite), webcam_detect_cv.py (classical)
"""

import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "models/hand_landmarker.task"

# MediaPipe returns 21 joint centres, not a silhouette. The true hand extends past
# the outermost joints -- knuckles, fingertip pads, the heel of the palm -- so the
# raw landmark extent is padded outward to land on something box-like.
BOX_MARGIN = 0.12

LEFT_COLOR = (255, 160, 0)
RIGHT_COLOR = (0, 220, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", help="run on a still image instead of the camera")
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="minimum detection confidence")
    parser.add_argument("--landmarks", action="store_true", help="also draw the 21 joints")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        parser.error(f"--confidence must be in [0, 1], got {args.confidence}")
    if args.max_hands < 1:
        parser.error(f"--max-hands must be at least 1, got {args.max_hands}")
    return args


def build_detector(model_path, max_hands, confidence, video_mode):
    """Create the MediaPipe HandLandmarker.

    VIDEO mode carries tracking state between frames, so a hand that is briefly
    blurred or half-occluded keeps its box instead of flickering out. IMAGE mode is
    stateless and is what a single still needs.
    """
    vision = mp.tasks.vision
    try:
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        return vision.HandLandmarker.create_from_options(options)
    except Exception as error:
        raise SystemExit(
            f"could not load the MediaPipe model from {model_path}: {error}\n"
            "Download it with:\n"
            "  curl -L -o models/hand_landmarker.task https://storage.googleapis.com/"
            "mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        ) from None


def landmarks_to_box(landmarks, width, height):
    """Collapse 21 normalised landmarks into one padded xyxy box in pixel coords."""
    xs = np.array([point.x for point in landmarks])
    ys = np.array([point.y for point in landmarks])

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad_x = (x2 - x1) * BOX_MARGIN
    pad_y = (y2 - y1) * BOX_MARGIN

    # A hand running off the edge produces landmarks outside [0, 1], so every corner
    # needs clamping at BOTH ends. Guarding only x1's low end and x2's high end lets
    # x1 exceed the frame and x2 go negative, yielding inverted boxes like
    # (652, 180, 639, 299) that draw as garbage.
    def clamp(value, limit):
        return int(min(max(value, 0), limit))

    return (
        clamp((x1 - pad_x) * width, width - 1),
        clamp((y1 - pad_y) * height, height - 1),
        clamp((x2 + pad_x) * width, width - 1),
        clamp((y2 + pad_y) * height, height - 1),
    )


def monotonic_timestamp_ms(started, tick, last_stamp):
    """Millisecond stamp for VIDEO mode, guaranteed greater than the previous one.

    detect_for_video rejects a repeated timestamp with
    ValueError("Input timestamp must be monotonically increasing"). A millisecond is
    coarse enough that two frames genuinely can land on the same value, so when that
    happens we step forward by one rather than letting the loop die.
    """
    stamp = int((tick - started) * 1000)
    return stamp if stamp > last_stamp else last_stamp + 1


def detect(detector, frame_bgr, timestamp_ms=None):
    """MediaPipe's half of the job: frame in, hand boxes out.

    Returns a list of (box, label, score, landmarks).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    if timestamp_ms is None:
        result = detector.detect(mp_image)
    else:
        result = detector.detect_for_video(mp_image, timestamp_ms)

    height, width = frame_bgr.shape[:2]
    detections = []
    for index, landmarks in enumerate(result.hand_landmarks):
        label, score = "hand", 1.0
        if index < len(result.handedness) and result.handedness[index]:
            category = result.handedness[index][0]
            label, score = category.category_name, category.score
        detections.append((landmarks_to_box(landmarks, width, height), label, score, landmarks))
    return detections


def draw(frame, detections, show_landmarks=False, fps=None):
    """OpenCV's half of the job: everything the user actually sees."""
    height, width = frame.shape[:2]

    for (x1, y1, x2, y2), label, score, landmarks in detections:
        color = LEFT_COLOR if label.lower().startswith("l") else RIGHT_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if show_landmarks:
            for point in landmarks:
                cv2.circle(frame, (int(point.x * width), int(point.y * height)),
                           2, color, -1)

    if fps is not None:
        hud = f"{fps:4.1f} fps   hands: {len(detections)}   q to quit"
        cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run_on_image(args):
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"could not read {args.image}")

    detector = build_detector(args.model, args.max_hands, args.confidence, video_mode=False)
    detections = detect(detector, frame)

    print(f"{len(detections)} hand(s) in {args.image}:")
    for (x1, y1, x2, y2), label, score, _ in detections:
        print(f"  {label:<5} {score:.2f}  box ({x1}, {y1}, {x2}, {y2})")

    cv2.imwrite("mp_detect_result.png", draw(frame.copy(), detections, args.landmarks))
    print("wrote mp_detect_result.png")


def run_on_camera(args):
    detector = build_detector(args.model, args.max_hands, args.confidence, video_mode=True)

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

    recent = deque(maxlen=30)
    started = time.perf_counter()
    last_stamp = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # mirror before detection so the drawn boxes match what is on screen
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            tick = time.perf_counter()
            # detect_for_video demands a STRICTLY increasing millisecond clock, and
            # a millisecond is coarse enough that two frames can land on the same
            # value. Step forward by one whenever that happens rather than letting
            # MediaPipe raise.
            stamp = monotonic_timestamp_ms(started, tick, last_stamp)
            last_stamp = stamp
            detections = detect(detector, frame, stamp)
            recent.append(time.perf_counter() - tick)

            frame = draw(frame, detections, args.landmarks,
                         len(recent) / max(sum(recent), 1e-6))
            cv2.imshow("MediaPipe hand detection  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    run_on_image(args) if args.image else run_on_camera(args)


if __name__ == "__main__":
    main()
