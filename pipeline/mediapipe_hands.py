"""
Shared MediaPipe runner for stages 1 and 3.

MediaPipe's HandLandmarker produces the bounding region AND the 21 landmarks in a single
inference. The pipeline keeps DETECT and POSE as separate stages because they are
conceptually separate and either can be swapped -- but running MediaPipe twice per frame
to honour that separation would halve the frame rate for nothing.

So this module runs it once per frame and caches the result. The detector and the pose
estimator both ask this object; whichever asks first pays for the inference and the other
gets it free. The stages stay swappable, the cost stays paid once.
"""

import cv2
import mediapipe as mp
import numpy as np

from .types import HandDetection, HandPose

DEFAULT_MODEL = "models/hand_landmarker.task"
BOX_MARGIN = 0.12          # landmarks are joint centres; the hand extends past them


class MediaPipeHands:
    """Runs HandLandmarker once per frame and serves both stages from the cache."""

    def __init__(self, model_path=DEFAULT_MODEL, max_hands=4, confidence=0.4, video_mode=True):
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
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as error:
            raise SystemExit(
                f"could not load the MediaPipe model from {model_path}: {error}\n"
                "Fetch it with:\n  mkdir -p models && curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            ) from None

        self.video_mode = video_mode
        self._cache_key = None
        self._cache = ([], [])
        self._last_stamp = -1

    def process(self, frame_bgr, timestamp_ms=None):
        """Return (detections, poses) for this frame, running inference at most once.

        The cache key is the frame buffer's identity plus its checksum, so a repeated
        call on the same array is free while a genuinely new frame is not mistaken for
        a cached one.
        """
        key = (id(frame_bgr), frame_bgr.shape, int(frame_bgr[::37, ::37].sum()))
        if key == self._cache_key:
            return self._cache

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video_mode:
            stamp = int(timestamp_ms) if timestamp_ms is not None else self._last_stamp + 1
            # detect_for_video rejects a repeated stamp; a millisecond is coarse enough
            # that two frames can share one, so step forward rather than crash
            if stamp <= self._last_stamp:
                stamp = self._last_stamp + 1
            self._last_stamp = stamp
            result = self._landmarker.detect_for_video(image, stamp)
        else:
            result = self._landmarker.detect(image)

        height, width = frame_bgr.shape[:2]
        detections, poses = [], []
        for index, landmarks in enumerate(result.hand_landmarks):
            points = np.array([[lm.x * width, lm.y * height] for lm in landmarks],
                              dtype=np.float32)
            depth = np.array([lm.z for lm in landmarks], dtype=np.float32)

            label, score = "unknown", 1.0
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                label, score = category.category_name, float(category.score)

            detections.append(HandDetection(box=_box_from_points(points, width, height),
                                            score=score, handedness=label))
            poses.append(HandPose(points=points, depth=depth, score=score))

        self._cache_key = key
        self._cache = (detections, poses)
        return self._cache

    def close(self):
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None


def _box_from_points(points, width, height):
    """Padded, clamped bounding box around the 21 landmarks.

    Every corner is clamped at BOTH ends. Guarding only x1's low end and x2's high end
    lets a hand leaving the frame produce an inverted box like (652, 180, 639, 299).
    """
    x1, y1 = points[:, 0].min(), points[:, 1].min()
    x2, y2 = points[:, 0].max(), points[:, 1].max()
    pad_x, pad_y = (x2 - x1) * BOX_MARGIN, (y2 - y1) * BOX_MARGIN

    def clamp(value, limit):
        return int(min(max(value, 0), limit))

    return (clamp(x1 - pad_x, width - 1), clamp(y1 - pad_y, height - 1),
            clamp(x2 + pad_x, width - 1), clamp(y2 + pad_y, height - 1))
