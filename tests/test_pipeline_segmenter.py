"""
Tests for pipeline/segmenter.py -- stage 2, pixel-level hand segmentation.

Three groups, in order of how much they are worth:

1. COORDINATE SPACES. HandMask.mask is the only ROI-local array in the pipeline and
   everything else is full-frame. Every plausible way to confuse the two is asserted
   against here, because the failure is silent: a mask built in the wrong space is
   still a valid uint8 array of plausible shape, it just marks the wrong pixels, and
   nothing downstream can tell. Several of these tests would pass on a segmenter that
   returned a full-frame array, so they check the *origin* and the *placement* too,
   not only the shape.

2. BEHAVIOUR ON SYNTHETIC ARRAYS. Painted rectangles with known answers, so a failure
   points at one line rather than at "the mask got worse".

3. GROUND TRUTH. EgoHands ships a per-hand segmentation polygon for every labelled
   frame, so the quality claims in the module docstring are measured, not asserted.
   These run MediaPipe to get the landmarks that the landmark and hybrid methods need,
   which is the same stage-3 model the real pipeline uses.

Run:  .venv/bin/python -m pytest tests/test_pipeline_segmenter.py -v
"""

import time

import cv2
import numpy as np
import pytest

from conftest import needs_dataset, needs_mp_model
from pipeline.segmenter import (HandSegmenter, METHODS, adaptive_skin_mask,
                                clean_mask, fixed_skin_mask, hand_scale,
                                landmark_hull_mask, largest_component, mask_iou)
from pipeline.types import HandDetection, HandMask, HandPose

# A mid-tone skin BGR that clears both fixed gates, and a background that clears
# neither. Chosen by construction rather than by eye: see test_synthetic_colours.
SKIN_BGR = (110, 140, 195)
BACKGROUND_BGR = (60, 130, 60)          # green, unambiguously not skin

FRAME_H, FRAME_W = 480, 640


# --------------------------------------------------------------------- helpers

def synthetic_frame(box, skin=SKIN_BGR, background=BACKGROUND_BGR, noise=0):
    """A frame that is background everywhere except a solid skin rectangle at `box`."""
    frame = np.full((FRAME_H, FRAME_W, 3), background, dtype=np.uint8)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = skin
    if noise:
        rng = np.random.RandomState(0)
        frame = np.clip(frame.astype(np.int16)
                        + rng.randint(-noise, noise + 1, frame.shape), 0, 255)
        frame = frame.astype(np.uint8)
    return frame


def fake_pose(box, margin=0.15):
    """21 plausible landmarks laid out inside `box`, in FULL-FRAME pixels.

    Not anatomically real -- a wrist, five MCPs across the palm and four joints up
    each finger -- but it has the right topology, so hand_scale() and the hull come
    out sane and any coordinate-space bug still shows.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px1, py1 = x1 + margin * w, y1 + margin * h
    px2, py2 = x2 - margin * w, y2 - margin * h
    points = np.zeros((21, 2), dtype=np.float32)
    points[0] = (0.5 * (px1 + px2), py2)                       # wrist at the bottom
    for finger in range(5):
        base_x = px1 + finger * (px2 - px1) / 4.0
        for joint in range(4):
            idx = 1 + finger * 4 + joint
            t = (joint + 1) / 4.0
            points[idx] = (base_x, py2 - t * (py2 - py1))
    return HandPose(points=points)


def centroid(binary):
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


# ------------------------------------------------------- 1. coordinate spaces

class TestCoordinateSpaces:
    """The one convention: everything is full-frame except HandMask.mask."""

    def test_mask_is_roi_sized_not_frame_sized(self):
        box = (200, 150, 320, 290)
        frame = synthetic_frame(box)
        mask = HandSegmenter(method="skin").segment(frame, HandDetection(box, 0.9))
        assert mask.mask.shape == (box[3] - box[1], box[2] - box[0])
        assert mask.mask.shape != frame.shape[:2], "mask must be ROI-local, not full-frame"

    def test_origin_is_the_roi_top_left_in_frame_space(self):
        box = (200, 150, 320, 290)
        frame = synthetic_frame(box)
        mask = HandSegmenter(method="skin", pad=0.0).segment(frame, HandDetection(box, 0.9))
        assert mask.origin == (200, 150)

    @pytest.mark.parametrize("method", ["skin", "landmark", "hybrid"])
    def test_to_frame_puts_the_hand_where_the_box_is(self, method):
        """The load-bearing test. An ROI-local mask read as full-frame lands at the
        top-left corner of the image; a correctly placed one lands inside the box."""
        box = (400, 300, 520, 440)
        frame = synthetic_frame(box)
        pose = fake_pose(box)
        mask = HandSegmenter(method=method).segment(frame, HandDetection(box, 0.9), pose)
        full = mask.to_frame(frame.shape)
        cx, cy = centroid(full)
        assert box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
        assert full[:box[1], :].sum() == 0, "hand pixels above the box: ROI/frame mixup"
        assert full[:, :box[0]].sum() == 0, "hand pixels left of the box: ROI/frame mixup"

    def test_two_identical_hands_at_different_places_give_identical_roi_masks(self):
        """Same crop content, different frame position => byte-identical ROI mask and
        different origins. If any frame coordinate leaked into the ROI computation the
        two masks would differ."""
        seg = HandSegmenter(method="hybrid")
        box_a, box_b = (100, 100, 220, 240), (420, 260, 540, 400)
        mask_a = seg.segment(synthetic_frame(box_a), HandDetection(box_a, 1.0),
                             fake_pose(box_a))
        mask_b = seg.segment(synthetic_frame(box_b), HandDetection(box_b, 1.0),
                             fake_pose(box_b))
        assert mask_a.origin != mask_b.origin
        assert np.array_equal(mask_a.mask, mask_b.mask)

    def test_pose_is_read_as_full_frame_not_roi_local(self):
        """Landmarks arrive in frame space. Feeding the same numbers as if they were
        ROI-local would place the hull 400 px away and produce a near-empty mask."""
        box = (400, 300, 520, 440)
        frame = synthetic_frame(box)
        pose = fake_pose(box)
        mask = HandSegmenter(method="landmark").segment(frame, HandDetection(box, 1.0), pose)
        assert mask.pixel_count > 0.2 * mask.mask.size, (
            "landmark mask is nearly empty -- pose points were probably treated as "
            "ROI-local when they are full-frame"
        )

    def test_landmarks_outside_the_roi_are_clipped_not_crashed(self):
        """Stage 3 can put landmarks outside stage 1's box; that must not raise."""
        box = (300, 200, 380, 280)
        frame = synthetic_frame(box)
        pose = fake_pose((260, 160, 420, 320))          # deliberately wider than the box
        mask = HandSegmenter(method="landmark").segment(frame, HandDetection(box, 1.0), pose)
        assert mask.mask.shape == (80, 80)


# --------------------------------------------------- 2. synthetic-array behaviour

class TestContract:

    def test_returns_a_handmask_with_the_documented_dtype_and_values(self):
        box = (100, 100, 250, 260)
        frame = synthetic_frame(box)
        for method in METHODS:
            pose = fake_pose(box)
            mask = HandSegmenter(method=method).segment(frame, HandDetection(box, 1.0), pose)
            assert isinstance(mask, HandMask)
            assert mask.mask.dtype == np.uint8
            assert set(np.unique(mask.mask)).issubset({0, 255}), method

    def test_unknown_method_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="unknown method"):
            HandSegmenter(method="magic")

    def test_landmark_without_a_pose_raises(self):
        frame = synthetic_frame((100, 100, 200, 200))
        with pytest.raises(ValueError, match="needs a pose"):
            HandSegmenter(method="landmark").segment(frame, HandDetection((100, 100, 200, 200), 1.0))

    def test_hybrid_without_a_pose_falls_back_to_skin(self):
        box = (100, 100, 250, 260)
        frame = synthetic_frame(box)
        detection = HandDetection(box, 1.0)
        hybrid = HandSegmenter(method="hybrid").segment(frame, detection, None)
        skin = HandSegmenter(method="skin").segment(frame, detection, None)
        assert np.array_equal(hybrid.mask, skin.mask)
        assert hybrid.origin == skin.origin

    def test_degenerate_and_offscreen_boxes_return_empty_masks(self):
        frame = synthetic_frame((100, 100, 200, 200))
        seg = HandSegmenter(method="hybrid")
        for box in [(50, 50, 50, 50),                    # zero area
                    (100, 100, 90, 90),                  # inverted
                    (900, 900, 1000, 1000),              # entirely off-frame
                    (-200, -200, -100, -100)]:           # entirely off-frame, negative
            mask = seg.segment(frame, HandDetection(box, 1.0))
            assert mask.pixel_count == 0
            assert mask.to_frame(frame.shape).sum() == 0

    def test_box_partly_outside_the_frame_is_clamped(self):
        frame = synthetic_frame((0, 0, 120, 120))
        mask = HandSegmenter(method="skin").segment(frame, HandDetection((-40, -40, 120, 120), 1.0))
        assert mask.origin == (0, 0)
        assert mask.mask.shape == (120, 120)

    def test_the_input_frame_is_never_modified(self):
        box = (100, 100, 250, 260)
        frame = synthetic_frame(box)
        before = frame.copy()
        for method in METHODS:
            HandSegmenter(method=method).segment(frame, HandDetection(box, 1.0), fake_pose(box))
        assert np.array_equal(frame, before)

    def test_segment_many_matches_segment(self):
        boxes = [(60, 60, 180, 200), (300, 220, 430, 380)]
        frame = synthetic_frame(boxes[0])
        frame[220:380, 300:430] = SKIN_BGR
        detections = [HandDetection(b, 1.0) for b in boxes]
        poses = [fake_pose(b) for b in boxes]
        seg = HandSegmenter()
        many = seg.segment_many(frame, detections, poses)
        assert len(many) == 2
        for got, det, pose in zip(many, detections, poses):
            one = seg.segment(frame, det, pose)
            assert np.array_equal(got.mask, one.mask)
            assert got.origin == one.origin

    @pytest.mark.parametrize("work_size", [0, 48, 96, 192, 512])
    def test_working_resolution_never_changes_the_output_shape(self, work_size):
        """The internal downscale is an optimisation; the mask must still come back at
        full ROI resolution whatever it does."""
        box = (100, 80, 460, 400)
        frame = synthetic_frame(box)
        mask = HandSegmenter(method="hybrid", work_size=work_size).segment(
            frame, HandDetection(box, 1.0), fake_pose(box))
        assert mask.mask.shape == (box[3] - box[1], box[2] - box[0])
        assert set(np.unique(mask.mask)).issubset({0, 255})


class TestHybridGuards:
    """The two ways hybrid is meant to survive colour going wrong."""

    def test_collapse_guard_falls_back_to_the_hull(self):
        """Fixed gates + a hand that is not skin-coloured => no colour evidence at all.
        The result must be the landmark hull, not an empty mask."""
        box = (150, 120, 310, 300)
        frame = synthetic_frame(box, skin=(200, 60, 40))     # a blue glove
        pose = fake_pose(box)
        detection = HandDetection(box, 1.0)
        hybrid = HandSegmenter(method="hybrid", adaptive=False).segment(frame, detection, pose)
        hull = HandSegmenter(method="landmark").segment(frame, detection, pose)
        assert hybrid.pixel_count > 0, "guard did not fire; hybrid returned nothing"
        assert np.array_equal(hybrid.mask, hull.mask)

    def test_adaptive_colour_segments_a_hand_the_fixed_gates_reject(self):
        """With the adaptive model on, the same blue glove is segmented properly --
        the colour window is fitted to the glove, not to a literature skin box. This
        is the guard NOT firing, and it is the better outcome."""
        box = (150, 120, 310, 300)
        frame = synthetic_frame(box, skin=(200, 60, 40))
        pose = fake_pose(box)
        detection = HandDetection(box, 1.0)
        adaptive = HandSegmenter(method="hybrid", adaptive=True).segment(frame, detection, pose)
        hull = HandSegmenter(method="landmark").segment(frame, detection, pose)
        assert adaptive.pixel_count > hull.pixel_count, (
            "adaptive hybrid should have found the whole glove, not just the hull")

    def test_shadowed_interior_is_not_carved_out_of_its_own_hand(self):
        """A dark band across the palm fails the colour gate. Unioning the eroded hull
        back in is what stops the hand being cut in two."""
        box = (150, 120, 310, 300)
        frame = synthetic_frame(box)
        frame[190:230, 160:300] = (20, 25, 35)               # deep shadow across the palm
        pose = fake_pose(box)
        mask = HandSegmenter(method="hybrid").segment(frame, HandDetection(box, 1.0), pose)
        full = mask.to_frame(frame.shape)
        assert full[205, 230] == 255, "shadowed palm pixels were dropped"
        count, _ = cv2.connectedComponents((full > 0).astype(np.uint8))
        assert count == 2, "the hand was split into more than one blob"


class TestSkinStage:

    def test_synthetic_colours(self):
        """Guard the constants the rest of the synthetic tests rest on."""
        patch = np.full((4, 4, 3), SKIN_BGR, dtype=np.uint8)
        assert fixed_skin_mask(patch).all(), "SKIN_BGR must pass the fixed gates"
        patch = np.full((4, 4, 3), BACKGROUND_BGR, dtype=np.uint8)
        assert not fixed_skin_mask(patch).any(), "BACKGROUND_BGR must fail the gates"

    def test_skin_recovers_a_painted_rectangle(self):
        """Hand fills the left half of the ROI; the answer is known exactly."""
        frame = np.full((FRAME_H, FRAME_W, 3), BACKGROUND_BGR, dtype=np.uint8)
        frame[100:300, 100:200] = SKIN_BGR
        box = (100, 100, 300, 300)                       # ROI is twice as wide as the hand
        mask = HandSegmenter(method="skin").segment(frame, HandDetection(box, 1.0))
        recovered = mask.mask > 0
        assert recovered[:, :95].mean() > 0.95, "missed most of the painted hand"
        assert recovered[:, 105:].mean() < 0.05, "claimed the background as hand"

    def test_largest_component_drops_specks(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, 20:80] = 255                         # the hand
        mask[5, 5] = 255                                 # a speck
        mask[95, 95] = 255                               # another speck
        kept = largest_component(mask)
        assert kept[5, 5] == 0 and kept[95, 95] == 0
        assert kept[50, 50] == 255

    def test_largest_component_prefers_the_seeded_blob_over_the_bigger_one(self):
        """A forearm fragment can be larger than the hand. The seed must win."""
        mask = np.zeros((100, 100), np.uint8)
        mask[0:90, 0:30] = 255                           # big blob (the "forearm")
        mask[40:70, 60:90] = 255                         # smaller blob (the "hand")
        seed = np.zeros((100, 100), np.uint8)
        seed[50:60, 70:80] = 255
        kept = largest_component(mask, seed)
        assert kept[55, 75] == 255
        assert kept[45, 15] == 0

    def test_specks_are_removed_before_gaps_are_filled(self):
        """clean_mask opens then closes; reversing it would merge the specks in."""
        mask = np.zeros((60, 60), np.uint8)
        mask[20:40, 20:40] = 255
        mask[28:32, 28:32] = 0                           # a pinhole to fill
        mask[5, 50] = mask[6, 51] = 255                  # specks to delete
        cleaned = clean_mask(mask, 5, 9)
        assert cleaned[30, 30] == 255, "pinhole was not closed"
        assert cleaned[5, 50] == 0, "speck survived the opening"

    def test_adaptive_model_declines_a_seed_that_is_too_small(self):
        roi = np.full((50, 50, 3), SKIN_BGR, dtype=np.uint8)
        seed = np.zeros((50, 50), np.uint8)
        seed[0:3, 0:3] = 255                             # 9 pixels
        assert adaptive_skin_mask(roi, seed) is None

    def test_adaptive_model_beats_the_fixed_gates_on_an_off_palette_hand(self):
        """A hand under a colour cast that the literature box does not cover: the
        fixed gates lose it, a window fitted to its own pixels keeps it. This is the
        entire argument for doing colour per-hand instead of globally."""
        roi = np.full((80, 80, 3), (150, 120, 140), dtype=np.uint8)   # violet cast
        assert fixed_skin_mask(roi).mean() < 0.5
        seed = np.zeros((80, 80), np.uint8)
        seed[30:50, 30:50] = 255
        adapted = adaptive_skin_mask(roi, seed)
        assert adapted is not None and adapted.mean() > 250


class TestLandmarkStage:

    def test_hull_contains_every_landmark(self):
        pose = fake_pose((0, 0, 200, 200))
        mask = landmark_hull_mask(pose.points, (200, 200), radius=0)
        for x, y in pose.points:
            assert mask[int(round(y)), int(round(x))] == 255

    def test_dilation_only_grows_the_hull(self):
        pose = fake_pose((0, 0, 200, 200))
        small = landmark_hull_mask(pose.points, (200, 200), radius=0)
        big = landmark_hull_mask(pose.points, (200, 200), radius=8)
        assert big.sum() > small.sum()
        assert np.all(big[small > 0] == 255), "dilation must not erase anything"

    def test_skeleton_prior_is_not_convex(self):
        """The point of the skeleton style: it does not flood the finger gaps."""
        pose = fake_pose((0, 0, 200, 200))
        hull = landmark_hull_mask(pose.points, (200, 200), 6, style="hull")
        skeleton = landmark_hull_mask(pose.points, (200, 200), 6, style="skeleton")
        assert skeleton.sum() < hull.sum()

    def test_hand_scale_is_linear_in_hand_size(self):
        small = fake_pose((0, 0, 100, 100)).points
        big = fake_pose((0, 0, 200, 200)).points
        assert hand_scale(big) == pytest.approx(2 * hand_scale(small), rel=0.02)

    def test_degenerate_landmarks_do_not_crash(self):
        for points in (np.zeros((21, 2), np.float32),
                       np.full((21, 2), np.nan, np.float32)):
            mask = landmark_hull_mask(points, (50, 50), radius=4)
            assert mask.shape == (50, 50) and mask.dtype == np.uint8


class TestMaskIou:

    def _handmask(self, origin, shape):
        return HandMask(np.full(shape, 255, np.uint8), origin)

    def test_identical_regions_score_one(self):
        mask = self._handmask((10, 20), (40, 30))
        truth = mask.to_frame((FRAME_H, FRAME_W))
        assert mask_iou(mask, truth, (FRAME_H, FRAME_W)) == pytest.approx(1.0)

    def test_disjoint_regions_score_zero(self):
        mask = self._handmask((10, 20), (40, 30))
        truth = self._handmask((300, 300), (40, 30)).to_frame((FRAME_H, FRAME_W))
        assert mask_iou(mask, truth, (FRAME_H, FRAME_W)) == 0.0

    def test_half_overlap_scores_one_third(self):
        """Two 40x40 squares overlapping in half: 800 / (1600+1600-800) = 1/3."""
        mask = HandMask(np.full((40, 40), 255, np.uint8), (100, 100))
        truth = np.zeros((FRAME_H, FRAME_W), np.uint8)
        truth[100:140, 120:160] = 255
        assert mask_iou(mask, truth, (FRAME_H, FRAME_W)) == pytest.approx(1 / 3)

    def test_accepts_the_three_channel_ground_truth_egohands_returns(self):
        mask = self._handmask((10, 20), (40, 30))
        truth = np.repeat(mask.to_frame((FRAME_H, FRAME_W))[:, :, None], 3, axis=2)
        assert mask_iou(mask, truth, (FRAME_H, FRAME_W)) == pytest.approx(1.0)

    def test_empty_union_scores_zero(self):
        mask = HandMask(np.zeros((10, 10), np.uint8), (0, 0))
        truth = np.zeros((FRAME_H, FRAME_W), np.uint8)
        assert mask_iou(mask, truth, (FRAME_H, FRAME_W)) == 0.0

    def test_a_wrongly_sized_ground_truth_is_an_error_not_a_score(self):
        mask = self._handmask((10, 20), (40, 30))
        with pytest.raises(ValueError, match="full-frame"):
            mask_iou(mask, np.zeros((100, 100), np.uint8), (FRAME_H, FRAME_W))

    def test_origin_is_actually_used(self):
        """Same ROI mask, two origins, one ground truth: the scores must differ. A
        mask_iou that ignored origin would return the same number twice."""
        truth = np.zeros((FRAME_H, FRAME_W), np.uint8)
        truth[100:140, 100:140] = 255
        here = mask_iou(HandMask(np.full((40, 40), 255, np.uint8), (100, 100)),
                        truth, (FRAME_H, FRAME_W))
        there = mask_iou(HandMask(np.full((40, 40), 255, np.uint8), (300, 300)),
                         truth, (FRAME_H, FRAME_W))
        assert here == pytest.approx(1.0)
        assert there == 0.0


# ------------------------------------------------------------ 3. ground truth

EVAL_FRAMES = 44        # the quality bar asks for at least 40 real frames


@pytest.fixture(scope="module")
def gt_cases(videos):
    """Real EgoHands hands: (frame, box, GT mask, MediaPipe pose or None).

    One case per annotated hand, drawn from EVAL_FRAMES frames spread across all 48
    videos so no single location, activity or actor dominates the score. Poses come
    from the same MediaPipe HandLandmarker that stage 3 uses, run on a padded crop of
    the ground-truth box -- that is what stage 3 would hand us at run time.
    """
    pytest.importorskip("mediapipe")
    import mediapipe as mp
    from get_bounding_boxes import get_bounding_boxes
    from get_frame_path import get_frame_path
    from get_segmentation_mask import get_segmentation_mask

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1, min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2)
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    hand_types = ("my_left", "my_right", "your_left", "your_right")
    rng = np.random.RandomState(0)
    cases = []
    for n in range(EVAL_FRAMES):
        video = videos.iloc[n % len(videos)]
        index = int(rng.randint(0, 100))
        frame = cv2.imread(get_frame_path(video, index))
        if frame is None:
            continue
        boxes = get_bounding_boxes(video, index)
        for which, hand_type in enumerate(hand_types):
            x, y, w, h = boxes[which]
            if w <= 1 or h <= 1:
                continue
            box = (int(x), int(y), int(x + w), int(y + h))
            truth = get_segmentation_mask(video, index, hand_type)
            # pose from a padded crop, converted straight back to full-frame pixels
            pad_x, pad_y = 0.35 * (box[2] - box[0]), 0.35 * (box[3] - box[1])
            cx1 = int(max(0, box[0] - pad_x)); cy1 = int(max(0, box[1] - pad_y))
            cx2 = int(min(frame.shape[1], box[2] + pad_x))
            cy2 = int(min(frame.shape[0], box[3] + pad_y))
            pose = None
            if cx2 - cx1 > 10 and cy2 - cy1 > 10:
                crop = frame[cy1:cy2, cx1:cx2]
                result = landmarker.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if result.hand_landmarks:
                    ch, cw = crop.shape[:2]
                    pose = HandPose(points=np.array(
                        [[p.x * cw + cx1, p.y * ch + cy1] for p in result.hand_landmarks[0]],
                        dtype=np.float32))
            cases.append(dict(frame=frame, box=box, truth=truth, pose=pose,
                              video=str(video.loc['video_id'][0])))
    landmarker.close()
    assert len(cases) >= 40, f"only gathered {len(cases)} hands"
    return cases


def _score(cases, method, use_pose=True, **kwargs):
    seg = HandSegmenter(method=method, **kwargs)
    scores, milliseconds = [], []
    for case in cases:
        if method == "landmark" and case["pose"] is None:
            continue
        start = time.perf_counter()
        mask = seg.segment(case["frame"], HandDetection(case["box"], 1.0),
                           case["pose"] if use_pose else None)
        milliseconds.append((time.perf_counter() - start) * 1000)
        scores.append(mask_iou(mask, case["truth"], case["frame"].shape))
    return np.array(scores), np.array(milliseconds)


@needs_dataset
@needs_mp_model
class TestAgainstGroundTruth:
    """The quality bar. Numbers here are measured on real EgoHands polygons."""

    def test_ground_truth_round_trips_through_mask_iou(self, gt_cases):
        """Sanity check on the scoring itself before it is used to judge anything.
        A HandMask cut out of the ground truth at the ROI must score ~1 against that
        same ground truth -- if this fails, every IoU below is measuring a coordinate
        bug rather than a segmenter."""
        for case in gt_cases[:10]:
            x1, y1, x2, y2 = case["box"]
            roi_truth = (case["truth"][y1:y2, x1:x2, 0] > 0).astype(np.uint8) * 255
            perfect = HandMask(roi_truth, (x1, y1))
            assert mask_iou(perfect, case["truth"], case["frame"].shape) > 0.99

    def test_every_method_beats_the_naive_whole_box_baseline(self, gt_cases):
        """Returning the entire ROI is what a segmenter that does nothing achieves.
        Anything that does not clear it is not earning its 1-2 ms."""
        baseline = []
        for case in gt_cases:
            x1, y1, x2, y2 = case["box"]
            box_mask = HandMask(np.full((y2 - y1, x2 - x1), 255, np.uint8), (x1, y1))
            baseline.append(mask_iou(box_mask, case["truth"], case["frame"].shape))
        baseline = float(np.mean(baseline))
        for method in METHODS:
            scores, _ = _score(gt_cases, method)
            assert scores.mean() > baseline, (
                f"{method} scored {scores.mean():.3f}, worse than the "
                f"whole-box baseline {baseline:.3f}")

    def test_hybrid_is_the_best_method(self, gt_cases):
        """The claim the module docstring makes, measured. Compared on the hands where
        a pose exists, because that is the only population all three can segment."""
        posed = [c for c in gt_cases if c["pose"] is not None]
        assert len(posed) >= 20
        results = {m: _score(posed, m)[0].mean() for m in METHODS}
        print("\n  mean IoU on %d posed hands: %s" %
              (len(posed), {k: round(v, 4) for k, v in results.items()}))
        assert results["hybrid"] > results["landmark"]
        assert results["hybrid"] > results["skin"]

    def test_measured_iou_matches_the_reported_numbers(self, gt_cases):
        """Regression floors, set ~0.05 below what was measured when this was written
        (hybrid 0.75, landmark 0.72, skin 0.72 on posed hands). Loose enough that a
        different frame sample will not trip them, tight enough that a real
        regression will."""
        posed = [c for c in gt_cases if c["pose"] is not None]
        assert _score(posed, "hybrid")[0].mean() > 0.68
        assert _score(posed, "landmark")[0].mean() > 0.65
        assert _score(posed, "skin")[0].mean() > 0.65

    def test_pose_free_hybrid_still_works(self, gt_cases):
        """When stage 3 misses a hand, hybrid must degrade to skin, not to nothing."""
        scores, _ = _score(gt_cases, "hybrid", use_pose=False)
        assert scores.mean() > 0.60

    def test_masks_stay_inside_their_roi(self, gt_cases):
        seg = HandSegmenter()
        for case in gt_cases[:25]:
            x1, y1, x2, y2 = case["box"]
            mask = seg.segment(case["frame"], HandDetection(case["box"], 1.0), case["pose"])
            assert mask.mask.shape == (y2 - y1, x2 - x1)
            full = mask.to_frame(case["frame"].shape)
            outside = full.copy()
            outside[y1:y2, x1:x2] = 0
            assert outside.sum() == 0, "mask leaked outside its own ROI"

    def test_real_time_budget(self, gt_cases):
        """Stage 2 gets ~5 ms per hand per frame. Median and p95 both have to fit,
        because a p95 that busts the budget is a visible stutter, not an average."""
        for method in METHODS:
            _, milliseconds = _score(gt_cases, method)
            median = float(np.median(milliseconds))
            p95 = float(np.percentile(milliseconds, 95))
            print("\n  %-9s median %.2f ms, p95 %.2f ms (n=%d)"
                  % (method, median, p95, len(milliseconds)))
            assert median < 5.0, f"{method} median {median:.2f} ms exceeds the budget"
            assert p95 < 10.0, f"{method} p95 {p95:.2f} ms exceeds twice the budget"

    def test_the_known_weak_case_is_still_the_known_weak_case(self, gt_cases):
        """Documented limit, asserted so it cannot quietly change: EgoHands' polygons
        for the observer's OWN hands run down the forearm to the frame edge, and the
        landmarks stop at the wrist, so those hands score materially worse than the
        partner's. If this ever inverts, the module's docstring is out of date."""
        posed = [c for c in gt_cases if c["pose"] is not None]
        mine = [c for c in posed if c["video"] and _is_own_hand(c)]
        yours = [c for c in posed if not _is_own_hand(c)]
        if len(mine) < 5 or len(yours) < 5:
            pytest.skip("not enough of each hand type in this sample")
        own = _score(mine, "hybrid")[0].mean()
        partner = _score(yours, "hybrid")[0].mean()
        print("\n  hybrid IoU: own hands %.3f, partner hands %.3f" % (own, partner))
        assert own < partner


def _is_own_hand(case):
    """The observer's own hands enter from the bottom of the frame."""
    return case["box"][3] >= 719
