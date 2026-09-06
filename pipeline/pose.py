"""
STAGE 3 -- 21 landmarks per hand.

    hand box -> HandPose(points=(21, 2) full-frame pixels)

    wrist 0; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20
    fingertips are 4, 8, 12, 16, 20

Landmarks come from MediaPipe. When the detector is also MediaPipe they share one
runner, so the model executes once per frame and this stage costs nothing extra --
see mediapipe_hands.py. With the SSDlite detector, poses are matched to the detector's
boxes by overlap, so the two stages stay independently swappable.
"""

import numpy as np

from .mediapipe_hands import DEFAULT_MODEL, MediaPipeHands


class PoseEstimator:
    """Stage 3. Call estimate(frame, detections) to get one HandPose per detection."""

    def __init__(self, model_path=DEFAULT_MODEL, max_hands=4, confidence=0.4,
                 shared=None, video_mode=True):
        self.mp_hands = shared or MediaPipeHands(model_path, max_hands, confidence, video_mode)

    def estimate(self, frame_bgr, detections, timestamp_ms=None):
        """Return a HandPose per detection, or None where no pose could be matched.

        The returned list is index-aligned with `detections`, so callers can zip the two
        without re-deriving the correspondence.
        """
        _, poses = self.mp_hands.process(frame_bgr, timestamp_ms)
        if not poses:
            return [None] * len(detections)

        matched = []
        claimed = set()
        for detection in detections:
            best, best_score = None, 0.0
            for index, pose in enumerate(poses):
                if index in claimed:
                    continue
                score = _fraction_inside(pose.points, detection.box)
                if score > best_score:
                    best, best_score = index, score
            # a pose whose landmarks barely fall in the box belongs to another hand
            if best is not None and best_score >= 0.5:
                claimed.add(best)
                matched.append(poses[best])
            else:
                matched.append(None)
        return matched

    def close(self):
        self.mp_hands.close()


def _fraction_inside(points, box):
    """What share of the 21 landmarks fall inside this box."""
    x1, y1, x2, y2 = box
    inside = ((points[:, 0] >= x1) & (points[:, 0] <= x2) &
              (points[:, 1] >= y1) & (points[:, 1] <= y2))
    return float(inside.mean())
