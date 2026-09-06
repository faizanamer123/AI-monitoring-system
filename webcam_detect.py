"""
webcam_detect.py turns on your camera and draws a box around every hand it finds,
using the detector produced by train_detector.py.

    python webcam_detect.py                    # default camera and checkpoint
    python webcam_detect.py --camera 1         # a different camera
    python webcam_detect.py --threshold 0.4    # looser (more boxes, more noise)
    python webcam_detect.py --device cpu       # force CPU

Press q or Esc to quit.

macOS note: the first run asks for camera access on behalf of whichever app is
running Python (Terminal, iTerm, VS Code). If no prompt appears and the frames come
back empty, grant it by hand in
System Settings -> Privacy & Security -> Camera, then restart that app.

See also train_detector.py, detection_dataset.py
"""

import argparse
import pickle
import time
from collections import deque

import cv2
import numpy as np
import torch

from detection_dataset import FRAME_HEIGHT, FRAME_WIDTH, IMAGE_SIZE, NUM_CLASSES
from train_detector import DEFAULT_CHECKPOINT, build_model, pick_device

BOX_COLOR = (0, 220, 0)
HUD_COLOR = (255, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="minimum confidence before a box is drawn. 0.2 is the "
                             "best-F1 operating point measured by evaluate_detector.py "
                             "(F1 0.897 vs 0.848 at 0.5)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--no-mirror", action="store_true",
                        help="show the raw camera feed instead of a mirrored selfie view")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error(f"--threshold must be in [0, 1], got {args.threshold}")
    return args


def load_model(checkpoint_path, device):
    """Rebuild the SSDlite architecture and load the fine-tuned weights into it."""
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise SystemExit(
            f"no checkpoint at {checkpoint_path}.\n"
            "Train one first:  python train_detector.py"
        ) from None
    except (IsADirectoryError, PermissionError, RuntimeError, EOFError,
            pickle.UnpicklingError, ValueError) as error:
        # a directory, an unreadable file, or a truncated/corrupt .pth all land here
        raise SystemExit(f"could not read the checkpoint at {checkpoint_path}: {error}") from None
    if not isinstance(state, dict) or "model" not in state:
        raise SystemExit(
            f"{checkpoint_path} is not a hand-detector checkpoint "
            "(expected a dict with a 'model' key)"
        )

    # pretrained=False: the COCO weights would be overwritten by load_state_dict below
    model = build_model(state.get("num_classes", NUM_CLASSES), pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    metrics = state.get("metrics", {})
    if metrics:
        print(f"loaded {checkpoint_path} (epoch {state.get('epoch', '?')}, "
              f"val F1 {metrics.get('f1', float('nan')):.3f})")
    return model


def suppress_contained(boxes, scores, containment=0.6):
    """Drop boxes that sit mostly inside a higher-scoring box.

    SSD's own NMS suppresses by IoU, which cannot remove a nested duplicate: a small
    box inside a large one has IoU = small/large, a low value, so it survives at any
    nms_thresh. That is what produces several boxes on one hand -- one covering the
    whole hand and another covering just the palm or a couple of fingers.

    Containment (intersection / area-of-smaller) is the right measure here, because
    it goes to 1.0 exactly when one box is swallowed by another.
    """
    if len(boxes) < 2:
        return np.arange(len(boxes))

    order = np.argsort(-scores)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 1) * np.maximum(boxes[:, 3] - boxes[:, 1], 1)
    keep = []
    for i in order:
        swallowed = False
        for j in keep:
            x1 = max(boxes[i, 0], boxes[j, 0]); y1 = max(boxes[i, 1], boxes[j, 1])
            x2 = min(boxes[i, 2], boxes[j, 2]); y2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if inter / min(areas[i], areas[j]) >= containment:
                swallowed = True
                break
        if not swallowed:
            keep.append(i)
    return np.array(sorted(keep), dtype=int)


@torch.no_grad()
def detect(model, frame_bgr, device, threshold):
    """Run the detector on one BGR frame; boxes come back in that frame's pixel coords.

    The frame is resized here rather than handed over at full resolution. Both paths
    end at 320x320, but by different roads: detection_dataset uses cv2.INTER_AREA,
    while SSD's internal transform would use bilinear. That is a different picture,
    and on the same frame it returned a different number of boxes (4 vs 3). Matching
    the training resize means the live demo behaves like the measured accuracy says
    it does.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    square = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(square.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)

    output = model([tensor])[0]
    keep = output["scores"] >= threshold
    boxes = output["boxes"][keep].cpu().numpy()
    # boxes come back in 320x320 space; scale them onto the caller's frame
    boxes[:, [0, 2]] *= width / IMAGE_SIZE
    boxes[:, [1, 3]] *= height / IMAGE_SIZE
    scores = output["scores"][keep].cpu().numpy()

    survivors = suppress_contained(boxes, scores)
    return boxes[survivors], scores[survivors]


def draw(frame, boxes, scores, fps):
    """Draw each box with its confidence, plus a small HUD."""
    for (x1, y1, x2, y2), score in zip(boxes.astype(int), scores):
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"hand {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # filled strip behind the text so it stays readable over a busy background
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    hud = f"{fps:4.1f} fps   hands: {len(boxes)}   q to quit"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                HUD_COLOR, 1, cv2.LINE_AA)
    return frame


def open_camera(index):
    """Open the camera and ask it for 1280x720.

    The request matters: the model was trained on 16:9 EgoHands frames squashed to a
    square, so feeding it a 4:3 webcam frame would distort hands differently than
    anything it saw in training. Cameras are free to ignore the request, so we print
    whatever we actually got.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open camera {index}.\n"
            "On macOS, check System Settings -> Privacy & Security -> Camera, "
            "or try a different --camera index."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {index} open at {width}x{height}")
    return cap


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"device {device}")

    model = load_model(args.checkpoint, device)
    cap = open_camera(args.camera)

    recent = deque(maxlen=30)   # rolling window so the fps readout does not flicker
    window = "EgoHands - hand detection"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped returning frames")
                break

            # mirror before inference, so the boxes line up with what is on screen
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            started = time.perf_counter()
            boxes, scores = detect(model, frame, device, args.threshold)
            recent.append(time.perf_counter() - started)

            frame = draw(frame, boxes, scores, len(recent) / max(sum(recent), 1e-6))
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
