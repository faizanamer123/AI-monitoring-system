"""
webcam_detect_cv.py draws boxes around hands using CLASSICAL OpenCV only -- no neural
network, no training, and no EgoHands download required. Run it and it works.

    python webcam_detect_cv.py                  # live camera
    python webcam_detect_cv.py --image x.png    # test on a still image
    python webcam_detect_cv.py --show-mask      # see the skin mask it is working from

READ THIS BEFORE TRUSTING IT
This does not detect hands. It detects skin-coloured blobs and then guesses which of
them are hands. That distinction is the whole story:
  - your face and bare arms are skin-coloured, so they get boxed too
  - wooden desks, cardboard and beige walls fall inside skin range
  - it degrades with unusual lighting and across skin tones
  - a closed fist scores almost exactly like a face

The finger heuristic below claws back some of that, but if you want something that
actually knows what a hand is, use webcam_detect.py (trained SSDlite) or MediaPipe.

See also webcam_detect.py, train_detector.py
"""

import argparse

import cv2
import numpy as np

# YCrCb skin envelope. Preferred over HSV because Cr/Cb separate chroma from luma,
# so it holds up better when brightness shifts across the frame.
YCRCB_LOW = np.array([0, 133, 77], dtype=np.uint8)
YCRCB_HIGH = np.array([255, 173, 127], dtype=np.uint8)

# HSV envelope, used as a second opinion -- a pixel must satisfy BOTH to count as skin,
# which throws out a lot of wood and cardboard that passes YCrCb alone.
HSV_LOW = np.array([0, 30, 60], dtype=np.uint8)
HSV_HIGH = np.array([25, 170, 255], dtype=np.uint8)

BOX_COLOR = (0, 220, 0)
WEAK_COLOR = (0, 165, 255)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", help="run on a still image instead of the camera")
    parser.add_argument("--min-area", type=float, default=0.004,
                        help="smallest blob to consider, as a fraction of frame area")
    parser.add_argument("--max-area", type=float, default=0.25,
                        help="largest blob to consider, as a fraction of frame area")
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--show-mask", action="store_true",
                        help="show the skin mask side by side with the result")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    if args.max_hands < 1:
        parser.error(f"--max-hands must be at least 1, got {args.max_hands}")
    if not 0.0 <= args.min_area < args.max_area <= 1.0:
        parser.error("--min-area and --max-area must satisfy 0 <= min < max <= 1")
    return args


def skin_mask(frame_bgr):
    """Two-space skin threshold, then morphology to close finger gaps and drop speckle."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_and(
        cv2.inRange(ycrcb, YCRCB_LOW, YCRCB_HIGH),
        cv2.inRange(hsv, HSV_LOW, HSV_HIGH),
    )
    # blur first so JPEG noise does not survive as isolated skin pixels
    mask = cv2.medianBlur(mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def finger_defects(contour, hull_indices):
    """Count deep convexity defects -- the valleys between extended fingers.

    This is the one signal that separates a hand from a face: a spread hand has
    several deep notches around its hull, a face is close to convex. It buys nothing
    on a closed fist, which is exactly why this approach has a floor.
    """
    if hull_indices is None or len(hull_indices) < 4:
        return 0
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0
    if defects is None or len(defects) == 0:
        return 0
    # OpenCV <5 returns (N, 1, 4); OpenCV 5 returns (N, 4). Normalise to (N, 4).
    defects = np.asarray(defects).reshape(-1, 4)
    # depth is in 1/256 px; 12px+ is a real valley rather than contour jitter
    return int((defects[:, 3] / 256.0 > 12).sum())


def score_contour(contour, frame_area, args):
    """Return (score, box) for a candidate blob, or None if it cannot be a hand.

    Score is a crude 0-1 'handness': mostly the finger count, nudged by how far the
    blob's solidity sits from face-like (very solid) territory.
    """
    area = cv2.contourArea(contour)
    if not (args.min_area * frame_area <= area <= args.max_area * frame_area):
        return None

    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / float(h)
    if not 0.3 <= aspect <= 3.0:          # discard long thin streaks (arms, edges)
        return None

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    solidity = area / hull_area
    if solidity < 0.30:                    # too ragged to be a body part at all
        return None

    fingers = finger_defects(contour, cv2.convexHull(contour, returnPoints=False))

    # a face is near-convex (solidity ~0.9+) with no deep valleys; a spread hand
    # sits lower and notched. Reward notches, penalise blobs that are too solid.
    score = min(fingers / 4.0, 1.0) * 0.7 + max(0.0, (0.92 - solidity)) / 0.5 * 0.3
    return min(score, 1.0), (x, y, w, h), fingers, solidity


def detect(frame_bgr, args):
    """Find candidate hand boxes. Returns a list of (box, score, fingers, solidity)."""
    mask = skin_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    scored = []
    for contour in contours:
        result = score_contour(contour, frame_area, args)
        if result is not None:
            scored.append(result)

    scored.sort(key=lambda r: r[0], reverse=True)
    # max_hands is guarded here as well as in parse_args, because detect() is also
    # called directly. Negative slicing would silently drop the weakest detections
    # rather than returning none, which is not what a caller passing -1 means.
    limit = max(0, args.max_hands)
    return scored[:limit], mask


def draw(frame, detections):
    for score, (x, y, w, h), fingers, solidity in detections:
        color = BOX_COLOR if score >= 0.4 else WEAK_COLOR
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{score:.2f} f={fingers} s={solidity:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
        detections, mask = detect(frame, args)
        out = draw(frame.copy(), detections)
        if args.show_mask:
            out = np.hstack([out, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        print(f"{len(detections)} candidate(s) in {args.image}:")
        for score, box, fingers, solidity in detections:
            print(f"  score {score:.2f}  box {box}  fingers {fingers}  solidity {solidity:.2f}")
        # cv2.imwrite reports failure through its return value, not an exception,
        # so an unwritable path would otherwise print "wrote ..." and exit 0.
        written = []
        for path, image in (("cv_detect_result.png", out), ("cv_detect_mask.png", mask)):
            if cv2.imwrite(path, image):
                written.append(path)
            else:
                raise SystemExit(f"could not write {path}")
        print("wrote " + " and ".join(written))
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open camera {args.camera}. On macOS grant camera access in "
            "System Settings -> Privacy & Security -> Camera, then restart your terminal."
        )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            detections, mask = detect(frame, args)
            out = draw(frame, detections)
            if args.show_mask:
                out = np.hstack([out, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            cv2.imshow("classical OpenCV hand boxes  (q to quit)", out)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
