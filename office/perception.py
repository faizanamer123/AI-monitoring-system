"""
Who is in the frame, and where is their body?

    frame -> [Person(track_id=3, box=(x1, y1, x2, y2), keypoints=(17, 3) array), ...]

One YOLO-pose model finds people AND their 17 COCO body keypoints in a single pass, and
ultralytics' built-in ByteTrack carries an id from frame to frame. Running a separate
person detector and pose model would cost two inferences for the same boxes.

Keypoints are full-frame pixels with a per-point confidence. A point the model could
not see (legs behind a desk) still comes back with coordinates, just with low
confidence -- always check the confidence before trusting a position.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO keypoint order, as every YOLO-pose model emits it
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

DEFAULT_WEIGHTS = "models/yolo11n-pose.pt"
TRACKER_CONFIG = str(Path(__file__).with_name("tracker.yaml"))


@dataclass
class Person:
    track_id: int
    box: tuple              # (x1, y1, x2, y2) ints, full-frame pixels
    score: float
    keypoints: np.ndarray   # (17, 3): x, y, confidence


class PersonTracker:
    """Call update(frame) once per frame, in order. One instance per video or camera:
    the tracker's state is what keeps ids stable, so sharing it across sources would
    hand one video's ids to another."""

    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.3, imgsz=640, device="auto"):
        from ultralytics import YOLO

        from detect_hands import yolo_device

        if not Path(weights).is_file():
            # a fresh clone has no models/ folder: given a bare name, Ultralytics
            # downloads its own pretrained model
            weights = Path(weights).name
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = yolo_device(device)

    def update(self, frame_bgr, dedupe=True):
        """dedupe=False returns the tracker's raw output, duplicates included -- what
        gets cached, so the duplicate rule can still be tuned from the cache."""
        result = self.model.track(frame_bgr, persist=True, tracker=TRACKER_CONFIG,
                                  conf=self.conf, imgsz=self.imgsz, verbose=False,
                                  device=self.device)[0]
        boxes = result.boxes
        # ids are None until ByteTrack has confirmed a track, typically the first frame
        if boxes is None or boxes.id is None or result.keypoints is None:
            return []
        keypoints = result.keypoints.data.cpu().numpy()
        people = [
            Person(track_id=int(tid), box=tuple(int(v) for v in box), score=float(score),
                   keypoints=kp)
            for box, score, tid, kp in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                           boxes.id.cpu().numpy(), keypoints)
        ]
        return drop_nested(people) if dedupe else people


def drop_nested(people, containment=0.8):
    """Remove a box that lies mostly INSIDE a higher-scoring one.

    A seated person often draws two detections: the whole body, and the upper body
    above the desk. Their IoU is low (one box is much larger), so NMS keeps both, and
    the timeline gains a phantom second person sitting in the same chair. Containment
    -- intersection over the SMALLER box's area -- catches what IoU misses. Two people
    side by side overlap a little; one inside the other is the same person.
    """
    keep = []
    for person in sorted(people, key=lambda p: -p.score):
        x1, y1, x2, y2 = person.box
        area = max((x2 - x1) * (y2 - y1), 1)
        nested = False
        for kept in keep:
            kx1, ky1, kx2, ky2 = kept.box
            iw, ih = min(x2, kx2) - max(x1, kx1), min(y2, ky2) - max(y1, ky1)
            if iw > 0 and ih > 0:
                kept_area = max((kx2 - kx1) * (ky2 - ky1), 1)
                if iw * ih / min(area, kept_area) >= containment:
                    nested = True
                    break
        if not nested:
            keep.append(person)
    return keep
