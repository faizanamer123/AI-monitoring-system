"""
Shared data types for the hand pipeline.

    WEBCAM -> OpenCV frame
        -> 1. DETECT    boxes            HandDetection
        -> 2. SEGMENT   hand mask        HandMask
        -> 3. POSE      21 keypoints     HandPose
        -> 4. TRACK     ids, motion      TrackedHand
        -> gesture / trajectory

Every stage takes and returns these, so any stage can be swapped without touching
the others. Pixel coordinates are always in FULL FRAME space unless a field says
otherwise -- mixing frame space with ROI space is the easiest way to break this
kind of pipeline, so there is exactly one convention.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# MediaPipe's 21-landmark topology, used by pose, segmentation and gestures alike
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = {"thumb": THUMB, "index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}
FINGERTIPS = (4, 8, 12, 16, 20)

# pairs of landmark indices that form the skeleton, for drawing
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class HandDetection:
    """Stage 1 output: where a hand is."""
    box: tuple                      # (x1, y1, x2, y2) ints, full-frame pixels
    score: float
    handedness: str = "unknown"     # "Left" | "Right" | "unknown"

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class HandPose:
    """Stage 3 output: 21 keypoints."""
    points: np.ndarray              # (21, 2) float32, full-frame pixels
    depth: np.ndarray = None        # (21,) relative z, smaller is nearer the camera
    score: float = 1.0

    def point(self, index):
        return self.points[index]


@dataclass
class HandMask:
    """Stage 2 output: which pixels are hand.

    Stored as an ROI mask plus its origin rather than a full-frame array, because a
    full 720x1280 uint8 per hand per frame is 900 KB of allocation at 30 fps for no
    reason. Use to_frame() when a full-frame mask is genuinely needed.
    """
    mask: np.ndarray                # (h, w) uint8, 0 or 255, ROI-local
    origin: tuple                   # (x, y) of the ROI's top-left in frame space

    def to_frame(self, frame_shape):
        full = np.zeros(frame_shape[:2], dtype=np.uint8)
        x, y = self.origin
        h, w = self.mask.shape[:2]
        y2, x2 = min(y + h, full.shape[0]), min(x + w, full.shape[1])
        if y2 > y and x2 > x:
            full[y:y2, x:x2] = self.mask[: y2 - y, : x2 - x]
        return full

    @property
    def pixel_count(self):
        return int((self.mask > 0).sum())


@dataclass
class TrackedHand:
    """Stage 4 output: one hand, followed through time."""
    track_id: int
    detection: HandDetection
    pose: HandPose = None
    mask: HandMask = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=64))
    gesture: str = "none"
    gesture_confidence: float = 0.0
    velocity: tuple = (0.0, 0.0)    # pixels per second
    age: int = 0                    # frames seen
    missing: int = 0                # consecutive frames not matched

    @property
    def speed(self):
        return float(np.hypot(*self.velocity))
