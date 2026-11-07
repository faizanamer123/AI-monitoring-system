"""
STAGE 1 -- where is the hand?

    frame -> [HandDetection(box=(120, 80, 420, 500), score=0.95), ...]

Two backends, and the choice matters:

  "mediapipe" (default) -- Google's palm detector. Purpose-built for hands, trained on
      large diverse data, and it does NOT fire on faces, elbows or feet.

  "yolo" -- a YOLO detector trained on EgoHands + COCO-Hand + HaGRID. Those last two
      contain faces, torsos and feet that are NOT labelled as hands, which is the signal
      EgoHands alone cannot provide.

  "ssdlite" -- our own SSDlite fine-tuned on EgoHands. It scores AP@0.50 = 0.93 on
      EgoHands validation, which is genuine, but EgoHands is head-mounted footage in
      which the only skin-coloured objects are hands. The model therefore learned
      "skin-coloured blob = hand" and on a webcam it boxes faces, elbows and feet with
      high confidence. Use it for first-person footage, not for a webcam.

The default is mediapipe because everything downstream -- segmentation, pose, tracking,
gestures -- inherits stage 1's mistakes. A false box on a face becomes a tracked face
with a trajectory and a gesture.
"""

import numpy as np

from .mediapipe_hands import DEFAULT_MODEL, MediaPipeHands
from .types import HandDetection


class HandDetector:
    """Stage 1. Call detect(frame) to get boxes."""

    def __init__(self, backend="mediapipe", model_path=DEFAULT_MODEL, checkpoint=None,
                 threshold=0.4, max_hands=4, device=None, shared=None, video_mode=True):
        self.backend = backend
        self.threshold = threshold
        self.max_hands = max_hands

        if backend == "mediapipe":
            # `shared` lets the pose stage hand us its runner so MediaPipe executes once
            self.mp_hands = shared or MediaPipeHands(model_path, max_hands, threshold, video_mode)
        elif backend == "yolo":
            from ultralytics import YOLO

            from detect_hands import yolo_device

            self.yolo = YOLO(checkpoint or "checkpoints/hand_yolo.pt")
            self.yolo_device = yolo_device(device or "auto")
        elif backend == "ssdlite":
            import torch
            from train_detector import build_model, pick_device
            from webcam_detect import detect as ssd_detect

            self._torch = torch
            self._ssd_detect = ssd_detect
            self.device = device or pick_device("auto")
            state = torch.load(checkpoint or "checkpoints/hand_detector.pth",
                               map_location=self.device, weights_only=False)
            self.model = build_model(state.get("num_classes", 2), pretrained=False)
            self.model.load_state_dict(state["model"])
            self.model.to(self.device).eval()
        else:
            raise ValueError(
                f"unknown backend {backend!r}; use 'mediapipe', 'yolo' or 'ssdlite'"
            )

    def detect(self, frame_bgr, timestamp_ms=None):
        if self.backend == "mediapipe":
            detections, _ = self.mp_hands.process(frame_bgr, timestamp_ms)
            return detections[: self.max_hands]

        if self.backend == "yolo":
            # verbose=False or ultralytics prints a line per frame, which at 30 fps
            # buries every other message the pipeline emits
            # agnostic_nms is essential for a multi-class model. Ultralytics offsets
            # each class into its own coordinate band before NMS, so without it two
            # boxes of DIFFERENT classes can never suppress each other however much
            # they overlap -- one hand comes back as palm 0.62 + stop 0.58 +
            # no_gesture 0.45, three boxes stacked on the same hand.
            result = self.yolo.predict(frame_bgr, conf=self.threshold, verbose=False,
                                       agnostic_nms=True, device=self.yolo_device)[0]
            names = getattr(self.yolo, "names", {}) or {}
            out = []
            for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy(),
                                       result.boxes.cls.cpu().numpy()):
                # The class index is the gesture. Dropping it here -- as this backend
                # used to -- makes a 19-class model behave exactly like a 1-class one:
                # it detects hands and silently discards what they are doing.
                label = str(names.get(int(cls), "hand"))
                out.append(HandDetection(box=tuple(int(v) for v in box),
                                         score=float(score), handedness="unknown",
                                         gesture=None if label == "hand" else label))
            out.sort(key=lambda d: -d.score)
            return out[: self.max_hands]

        boxes, scores = self._ssd_detect(self.model, frame_bgr, self.device, self.threshold)
        order = np.argsort(-scores)[: self.max_hands]
        return [
            HandDetection(box=tuple(int(v) for v in boxes[i]), score=float(scores[i]),
                          handedness="unknown")
            for i in order
        ]

    def close(self):
        if self.backend == "mediapipe":
            self.mp_hands.close()
