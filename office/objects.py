"""
Which office objects are where: phones, laptops, keyboards, monitors, cups.

    frame + people -> [Thing(name="cell phone", box=(x1, y1, x2, y2), score=0.71), ...]

A stock COCO YOLO model already knows every object we need, so nothing is trained here.

TWO MODES.
  frame   one pass over the whole frame. Right for a 720p/1080p camera, where a phone
          is 40+ px across.
  person  one pass per person over a crop around them, enlarged to the model's input
          size. On 320x240 surveillance footage a phone is 5-10 px and invisible to a
          whole-frame pass; cropped and enlarged it is 30-60 px. Costs one inference
          per person instead of one per frame.

OBJECTS ARE REMEMBERED. Detection runs a few times a second, not every frame -- a phone
does not vanish in 100 ms -- and each sighting is kept for `hold_s` seconds. That keeps
the frame rate up and bridges the frames where a hand covers the phone.
"""

from dataclasses import dataclass

import numpy as np

WANTED = ("cell phone", "laptop", "keyboard", "tv", "mouse", "cup")
COMPUTER = ("laptop", "keyboard", "tv", "mouse")
DEFAULT_WEIGHTS = "yolo11n.pt"


@dataclass
class Thing:
    name: str
    box: tuple      # (x1, y1, x2, y2) ints, full-frame pixels
    score: float
    t: float = 0.0  # when it was last seen


class ObjectDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, mode="frame", conf=0.25, imgsz=640,
                 device="auto", every_s=0.33, hold_s=2.0, crop_margin=0.35):
        from ultralytics import YOLO

        from detect_hands import yolo_device

        if mode not in ("frame", "person"):
            raise ValueError(f"mode must be 'frame' or 'person', got {mode!r}")
        self.model = YOLO(weights)
        self.classes = [i for i, n in self.model.names.items() if n in WANTED]
        self.mode = mode
        self.conf = conf
        self.imgsz = imgsz
        self.device = yolo_device(device)
        self.every_s = every_s
        self.hold_s = hold_s
        self.crop_margin = crop_margin
        self.last_run = None
        self.memory = []

    def update(self, frame, people, t):
        """Objects believed present at time t: fresh detections when it is time to
        look again, otherwise what was seen within the last hold_s seconds."""
        if self.last_run is None or t - self.last_run >= self.every_s:
            self.last_run = t
            fresh = self._detect(frame, people, t)
            # a fresh sighting replaces an older one of the same object -- same name,
            # overlapping box -- rather than piling up duplicates
            kept = [m for m in self.memory
                    if not any(f.name == m.name and _iou(f.box, m.box) > 0.3 for f in fresh)]
            self.memory = kept + fresh
        self.memory = [m for m in self.memory if t - m.t <= self.hold_s]
        return list(self.memory)

    def _detect(self, frame, people, t):
        if self.mode == "frame":
            return self._run(frame, (0, 0), t)
        h, w = frame.shape[:2]
        found = []
        for person in people:
            x1, y1, x2, y2 = person.box
            mx, my = (x2 - x1) * self.crop_margin, (y2 - y1) * self.crop_margin
            cx1, cy1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
            cx2, cy2 = min(int(x2 + mx), w), min(int(y2 + my), h)
            if cx2 - cx1 > 8 and cy2 - cy1 > 8:
                found += self._run(frame[cy1:cy2, cx1:cx2], (cx1, cy1), t)
        return _dedupe(found)

    def _run(self, image, origin, t):
        # imgsz enlarges a small crop up to the model's input size: that is the point
        # of person mode
        result = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, classes=self.classes,
                                    verbose=False, device=self.device)[0]
        ox, oy = origin
        return [Thing(self.model.names[int(c)],
                      (int(b[0]) + ox, int(b[1]) + oy, int(b[2]) + ox, int(b[3]) + oy),
                      float(s), t)
                for b, s, c in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy())]


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _dedupe(things, iou=0.5):
    """Overlapping person crops see the same phone twice; keep the surer sighting."""
    kept = []
    for thing in sorted(things, key=lambda x: -x.score):
        if not any(k.name == thing.name and _iou(k.box, thing.box) > iou for k in kept):
            kept.append(thing)
    return kept


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
