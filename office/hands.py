"""
Hands, found by the project's own hand detector, only where a phone is near someone.

    frame + people + phones -> [Thing(name="hand", box=..., score=...), ...]

Pose wrists are a guess about where the hand is, and are often missing when an arm is
partly hidden. The hand detector looks at the hand itself, so "a hand is on the phone"
is direct evidence where the wrist rule only infers it. It runs on a crop around each
person with a phone nearby, so rooms without phones cost nothing extra.
"""

from .activity import _near_person
from .objects import Thing, _dedupe, _iou

DEFAULT_WEIGHTS = "checkpoints/hand_yolo.pt"


class HandFinder:
    # conf 0.2: on MM-Office, 0.4 dropped most real hands at small sizes, while 0.2
    # still gave mostly true hands
    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.2, imgsz=448, device="auto",
                 every_s=0.33, hold_s=1.0, crop_margin=0.35):
        from ultralytics import YOLO

        from detect_hands import yolo_device

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = yolo_device(device)
        self.every_s = every_s
        self.hold_s = hold_s
        self.crop_margin = crop_margin
        self.last_run = None
        self.memory = []

    def update(self, frame, people, things, t):
        if self.last_run is None or t - self.last_run >= self.every_s:
            self.last_run = t
            phones = [th for th in things if th.name == "cell phone"]
            fresh = []
            for person in people:
                if any(_near_person(p.box, person, 0.25) for p in phones):
                    fresh += self._crop(frame, person, t)
            fresh = _dedupe(fresh)
            kept = [m for m in self.memory if not any(_iou(f.box, m.box) > 0.3 for f in fresh)]
            self.memory = kept + fresh
        self.memory = [m for m in self.memory if t - m.t <= self.hold_s]
        return list(self.memory)

    def _crop(self, frame, person, t):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person.box
        mx, my = (x2 - x1) * self.crop_margin, (y2 - y1) * self.crop_margin
        cx1, cy1 = max(int(x1 - mx), 0), max(int(y1 - my), 0)
        cx2, cy2 = min(int(x2 + mx), w), min(int(y2 + my), h)
        if cx2 - cx1 <= 8 or cy2 - cy1 <= 8:
            return []
        result = self.model.predict(frame[cy1:cy2, cx1:cx2], conf=self.conf, imgsz=self.imgsz,
                                    verbose=False, device=self.device)[0]
        return [Thing("hand", (int(b[0]) + cx1, int(b[1]) + cy1, int(b[2]) + cx1, int(b[3]) + cy1),
                      float(s), t)
                for b, s in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
