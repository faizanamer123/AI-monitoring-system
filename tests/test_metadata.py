"""Behavioural tests for the EgoHands metadata / annotation layer.

Covers get_meta_by.py, get_frame_path.py, get_bounding_boxes.py and
get_segmentation_mask.py against the REAL metadata.mat and the REAL
_LABELLED_SAMPLES frames.  Nothing here is mocked: the invariants worth
protecting are the ones that tie the .mat structure to what is on disk.

Run:  .venv/bin/python -m pytest tests/test_metadata.py -v
"""

import collections
import os

import numpy as np
import pytest

from conftest import needs_dataset

from get_bounding_boxes import get_bounding_boxes
from get_frame_path import get_frame_path
from get_meta_by import get_meta_by
from get_segmentation_mask import get_segmentation_mask

FRAME_W, FRAME_H = 1280, 720

ACTIVITIES = {"CARDS", "CHESS", "JENGA", "PUZZLE"}
LOCATIONS = {"COURTYARD", "LIVINGROOM", "OFFICE"}
ACTORS = {"B", "S", "T", "H"}

# labelled_frames record layout: 0=frame_num, 1=myleft, 2=myright, 3=yourleft, 4=yourright
HAND_SLOTS = ("my_left", "my_right", "your_left", "your_right")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _Meta:
    """Holds the 48-row metadata frame behind a cheap __repr__.

    pytest prints every fixture argument at the top of a failure traceback, and
    the repr of this DataFrame -- 48 rows whose `labelled_frames` cells are the
    raw nested MATLAB structs -- takes ~40 seconds per failure to format. That
    turns a 45 s module into a 3 minute one as soon as anything goes red, so
    tests take this wrapper and reach through `.df`.
    """

    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return f"<EgoHands metadata: {len(self.df)} videos>"


@pytest.fixture(scope="session")
def meta(videos):
    """conftest's session-scoped `videos` frame, wrapped for cheap tracebacks."""
    return _Meta(videos)


def scalar(value):
    """get_meta_by returns some cells as bare strings and some as 1-element
    arrays (see test_categorical_columns_have_a_uniform_scalar_type).  Callers
    that only want the value should not have to care which."""
    if isinstance(value, np.ndarray):
        return str(value[0])
    return str(value)


def video_ids(df):
    return [scalar(v) for v in df["video_id"]]


def parts(video_id):
    """PUZZLE_COURTYARD_B_S -> ('PUZZLE', 'COURTYARD', 'B', 'S')"""
    activity, location, viewer, partner = video_id.split("_")
    return activity, location, viewer, partner


def hand_present(video, frame_idx, slot):
    """slot is 1..4; a hand absent from the frame is stored as an empty array."""
    return video.loc["labelled_frames"][0][frame_idx][slot].size > 0


def find_frame(video, wanted):
    """First frame index whose presence pattern equals `wanted` (4 bools)."""
    for i in range(100):
        if tuple(hand_present(video, i, s) for s in range(1, 5)) == tuple(wanted):
            return i
    return None


# --------------------------------------------------------------------------
# get_meta_by: the 4 x 3 x 4 design
# --------------------------------------------------------------------------

def test_no_args_returns_all_48_distinct_videos(meta):
    assert len(meta.df) == 48
    ids = video_ids(meta.df)
    assert len(set(ids)) == 48
    assert list(meta.df.columns) == [
        "video_id", "partner_video_id", "ego_viewer_id", "partner_id",
        "location_id", "activity_id", "labelled_frames",
    ]


def test_design_is_a_balanced_4x3x4(meta):
    """4 activities x 3 locations x 4 egocentric viewers, fully crossed."""
    decomposed = [parts(v) for v in video_ids(meta.df)]

    activities = collections.Counter(p[0] for p in decomposed)
    locations = collections.Counter(p[1] for p in decomposed)
    viewers = collections.Counter(p[2] for p in decomposed)

    assert set(activities) == ACTIVITIES and set(activities.values()) == {12}
    assert set(locations) == LOCATIONS and set(locations.values()) == {16}
    assert set(viewers) == ACTORS and set(viewers.values()) == {12}

    # every (activity, location) cell holds exactly the four viewers, once each
    cells = collections.defaultdict(list)
    for activity, location, viewer, _partner in decomposed:
        cells[(activity, location)].append(viewer)
    assert len(cells) == 12
    for cell, cell_viewers in cells.items():
        assert sorted(cell_viewers) == sorted(ACTORS), cell


def test_video_id_string_agrees_with_the_metadata_columns(meta):
    """The id string is the source of truth used downstream (detection_dataset
    parses it); it must not drift from the structured columns."""
    for _, row in meta.df.iterrows():
        vid = scalar(row.loc["video_id"])
        activity, location, viewer, partner = parts(vid)
        assert scalar(row.loc["activity_id"]) == activity, vid
        assert scalar(row.loc["location_id"]) == location, vid
        assert scalar(row.loc["ego_viewer_id"]) == viewer, vid
        assert scalar(row.loc["partner_id"]) == partner, vid
        assert viewer != partner, vid
        # the partner's own video is the same session filmed from the other head
        assert scalar(row.loc["partner_video_id"]) == "_".join(
            [activity, location, partner, viewer]
        ), vid


# --------------------------------------------------------------------------
# get_meta_by: filtering
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,value,expected_n,part_index",
    [
        ("Location", "OFFICE", 16, 1),
        ("Activity", "CHESS", 12, 0),
        ("Viewer", "B", 12, 2),
        ("Partner", "B", 12, 3),
    ],
)
def test_each_filter_selects_the_right_videos(name, value, expected_n, part_index):
    result = get_meta_by(name, value)
    ids = video_ids(result)
    assert len(ids) == expected_n, f"{name}={value} -> {ids}"
    assert {parts(v)[part_index] for v in ids} == {value}


@pytest.mark.parametrize(
    "args",
    [
        ("Location", "OFFICE", "Activity", "CHESS", "Viewer", "B", "Partner", "S"),
        ("Activity", "CHESS", "Location", "OFFICE", "Partner", "S", "Viewer", "B"),
        ("Viewer", "B", "Partner", "S", "Activity", "CHESS", "Location", "OFFICE"),
        ("Partner", "S", "Viewer", "B", "Location", "OFFICE", "Activity", "CHESS"),
        ("Viewer", "B", "Location", "OFFICE", "Activity", "CHESS", "Partner", "S"),
        ("Partner", "S", "Activity", "CHESS", "Location", "OFFICE", "Viewer", "B"),
    ],
    ids=["loc-first", "act-first", "viewer-first", "partner-first", "viewer-loc", "partner-act"],
)
def test_filter_order_does_not_matter(args):
    """Regression: a call whose FIRST filter was not 'Location' used to raise
    UnboundLocalError on the seeded parameter_check.  All four leading
    positions must work and must agree."""
    assert video_ids(get_meta_by(*args)) == ["CHESS_OFFICE_B_S"]


def test_single_filter_in_each_leading_position_does_not_raise():
    """The narrow form of the same regression: one filter, on its own."""
    for name, value in [
        ("Activity", "PUZZLE"),
        ("Viewer", "T"),
        ("Partner", "H"),
        ("Location", "LIVINGROOM"),
    ]:
        result = get_meta_by(name, value)
        assert len(result) > 0, f"{name}={value} returned nothing"


def test_multi_value_filter_unions_the_values():
    result = get_meta_by("Viewer", "B, S")
    ids = video_ids(result)
    assert len(ids) == 24
    assert {parts(v)[2] for v in ids} == {"B", "S"}


def test_multi_value_filter_combined_with_another_filter():
    result = get_meta_by("Location", "OFFICE, COURTYARD", "Activity", "CHESS")
    ids = video_ids(result)
    assert len(ids) == 8
    assert {parts(v)[1] for v in ids} == {"OFFICE", "COURTYARD"}
    assert {parts(v)[0] for v in ids} == {"CHESS"}


def test_multi_value_filter_example_from_the_module_docstring():
    """get_meta_by.py line 25 documents this exact call:

        get_meta_by('Location','OFFICE, COURTYARD', 'Activity','CHESS',
                    'Viewer', 'B,S,T')

    and promises "all videos of Chess played with B, S or T as the egocentric
    observer filmed at the Office or Courtyard locations" -- 6 videos.
    """
    result = get_meta_by(
        "Location", "OFFICE, COURTYARD", "Activity", "CHESS", "Viewer", "B,S,T"
    )
    ids = video_ids(result)
    assert len(ids) == 6, f"documented call returned {len(ids)} videos: {ids}"
    assert {parts(v)[2] for v in ids} == {"B", "S", "T"}


def test_filter_matching_nothing_returns_an_empty_frame_not_an_error():
    """B is never his own partner."""
    result = get_meta_by("Viewer", "B", "Partner", "B")
    assert len(result) == 0
    assert list(result.columns) == [
        "video_id", "partner_video_id", "ego_viewer_id", "partner_id",
        "location_id", "activity_id", "labelled_frames",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ("Location", "NOTAPLACE"),
        ("Activity", "FLYING"),
        ("Viewer", "Z"),
        ("Location", "office"),  # wrong case
    ],
    ids=["bad-location", "bad-activity", "bad-viewer", "wrong-case"],
)
def test_nonsense_filter_value_never_silently_widens_the_query(args):
    """A value outside the vocabulary must NOT come back as "everything".  The
    library chooses to return an empty frame rather than raise; that is
    defensible, quietly returning all 48 would not be."""
    result = get_meta_by(*args)
    assert len(result) < 48, f"{args} returned the unfiltered set"


def test_unrecognised_filter_name_is_not_silently_ignored():
    """An unknown filter must raise, not quietly return the whole table.

    The old loop tracked state across arguments and dropped any name it did not
    recognise, so a typo -- or the main_split filter the docstring used to promise --
    returned all 48 videos with no hint the filter had been ignored.
    """
    with pytest.raises(ValueError, match="unknown filter"):
        get_meta_by("MainSplit", "TEST")
    with pytest.raises(ValueError, match="unknown filter"):
        get_meta_by("Locaton", "OFFICE")          # typo
    with pytest.raises(ValueError, match="pairs"):
        get_meta_by("Location")                    # value missing


def test_categorical_columns_have_a_uniform_scalar_type(meta):
    """get_meta_by flattens the 1-element MATLAB cells into scalars.  All four
    categorical columns should come back the same way; today only the first one
    processed does, so callers need row['ego_viewer_id'] but
    row['location_id'][0] (see the workaround comment in detection_dataset.py)."""
    row = meta.df.iloc[0]
    kinds = {
        col: type(row.loc[col]).__name__
        for col in ("ego_viewer_id", "partner_id", "location_id", "activity_id")
    }
    assert len(set(kinds.values())) == 1, f"mixed cell types: {kinds}"


def test_repeated_calls_are_independent(meta):
    """No leakage of filter state between calls (module-level defaults)."""
    narrow = video_ids(get_meta_by("Viewer", "B"))
    wide = video_ids(get_meta_by())
    assert len(narrow) == 12
    assert len(wide) == 48
    assert set(narrow) < set(wide)


# --------------------------------------------------------------------------
# get_frame_path
# --------------------------------------------------------------------------

@needs_dataset
def test_get_frame_path_resolves_to_a_real_file_across_all_videos(meta, repo):
    """Every video, first / middle / last labelled frame."""
    missing = []
    for idx in range(len(meta.df)):
        video = meta.df.iloc[idx]
        vid = scalar(video.loc["video_id"])
        for i in (0, 49, 99):
            path = get_frame_path(video, i)
            if not os.path.isfile(path):
                missing.append(path)
                continue
            assert os.path.isabs(path)
            assert os.path.dirname(path) == str(repo / "_LABELLED_SAMPLES" / vid)
    assert missing == [], f"{len(missing)} frame paths do not exist, e.g. {missing[:3]}"


@needs_dataset
def test_get_frame_path_encodes_the_recorded_frame_number(meta):
    """The filename must be frame_%04d of the frame_num stored in the .mat,
    not of the annotation index."""
    for idx in (0, 11, 24, 47):
        video = meta.df.iloc[idx]
        frames = video.loc["labelled_frames"][0]
        for i in (0, 33, 99):
            recorded = int(frames[i][0][0][0])
            assert os.path.basename(get_frame_path(video, i)) == "frame_%04d.jpg" % recorded


@needs_dataset
def test_annotation_indices_are_strictly_increasing_video_frames(meta):
    """100 labelled frames per video, sampled forward through the recording."""
    for idx in range(len(meta.df)):
        video = meta.df.iloc[idx]
        frames = video.loc["labelled_frames"][0]
        assert len(frames) == 100, scalar(video.loc["video_id"])
        numbers = [int(frames[i][0][0][0]) for i in range(100)]
        assert numbers == sorted(numbers)
        assert len(set(numbers)) == 100


# --------------------------------------------------------------------------
# get_bounding_boxes
# --------------------------------------------------------------------------

def test_bounding_boxes_are_always_four_rows(meta):
    for idx in (0, 9, 18, 27, 36, 45):
        video = meta.df.iloc[idx]
        for i in (0, 25, 50, 75, 99):
            boxes = get_bounding_boxes(video, i)
            assert boxes.shape == (4, 4)


def test_zero_rows_correspond_exactly_to_absent_hands(meta):
    """A row is all-zero if and only if that hand is not annotated in the
    frame.  Sampled across all 48 videos so the invariant is not a property of
    video 0 alone."""
    checked = 0
    for idx in range(len(meta.df)):
        video = meta.df.iloc[idx]
        vid = scalar(video.loc["video_id"])
        for i in range(0, 100, 10):
            boxes = get_bounding_boxes(video, i)
            for slot in range(1, 5):
                row = boxes[slot - 1]
                present = hand_present(video, i, slot)
                assert bool(row.any()) == present, (vid, i, HAND_SLOTS[slot - 1], row)
                checked += 1
    assert checked == 48 * 10 * 4


def test_present_hand_boxes_lie_inside_the_1280x720_frame(meta):
    """x, y are 1-based top-left; width/height must be positive and the box
    must not run off the frame."""
    seen_present = 0
    for idx in range(len(meta.df)):
        video = meta.df.iloc[idx]
        vid = scalar(video.loc["video_id"])
        for i in range(0, 100, 10):
            boxes = get_bounding_boxes(video, i)
            for slot in range(1, 5):
                if not hand_present(video, i, slot):
                    continue
                x, y, w, h = boxes[slot - 1]
                where = (vid, i, HAND_SLOTS[slot - 1], boxes[slot - 1].tolist())
                assert w > 0 and h > 0, where
                assert 0 <= x <= FRAME_W - 1, where
                assert 0 <= y <= FRAME_H - 1, where
                assert x + w - 1 <= FRAME_W, where
                assert y + h - 1 <= FRAME_H, where
                seen_present += 1
    assert seen_present > 1000, "sample did not contain enough annotated hands"


def test_boxes_are_tight_around_the_segmentation_polygon(meta):
    """Cross-layer invariant: the box for a hand away from the frame border is
    the exact extent of that hand's filled mask.  Only frames with all four
    hands annotated are used, so the per-hand mask calls stay on the code path
    that works."""
    mismatches = []
    checked = 0
    for idx in (2, 14, 29, 41):
        video = meta.df.iloc[idx]
        vid = scalar(video.loc["video_id"])
        i = find_frame(video, (True, True, True, True))
        assert i is not None, vid
        boxes = get_bounding_boxes(video, i)
        for slot, hand_type in enumerate(HAND_SLOTS, start=1):
            mask = get_segmentation_mask(video, i, hand_type)[:, :, 0]
            ys, xs = np.nonzero(mask)
            x, y, w, h = boxes[slot - 1]
            if xs.min() == 0 or ys.min() == 0:
                continue  # border case, covered by the next test
            got = (x, y, w, h)
            want = (
                float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1),
            )
            if got != want:
                mismatches.append(f"{vid} frame {i} {hand_type}: box {got} vs mask {want}")
            checked += 1
    assert mismatches == [], "; ".join(mismatches)
    assert checked >= 12, "sample did not contain enough interior hands"


def test_boxes_cover_hands_that_touch_the_top_or_left_frame_edge(meta):
    """segmentation2box clamps the top-left corner with max(1, ...) and defines
    width as x2 - x1 + 1 -- MATLAB's 1-based, inclusive pixel convention, taken
    straight from getBoundingBoxes.m.  The Python pipeline around it is
    0-indexed: cv2.fillPoly paints row 0 and column 0, and detection_dataset
    feeds these boxes to torchvision as 0-indexed pixel corners.

    A hand whose polygon reaches row 0 or column 0 therefore gets a box that
    starts at 1 and misses the outermost row/column of the hand.
    """
    offenders = []
    for idx in range(len(meta.df)):
        video = meta.df.iloc[idx]
        vid = scalar(video.loc["video_id"])
        for i in range(0, 100, 10):
            boxes = get_bounding_boxes(video, i)
            for slot, hand_type in enumerate(HAND_SLOTS, start=1):
                if not hand_present(video, i, slot):
                    continue
                polygon = video.loc["labelled_frames"][0][i][slot]
                x, y, w, h = boxes[slot - 1]
                # np.int32 truncates toward zero, so this is what the box saw
                px, py = np.int32(polygon)[:, 0].min(), np.int32(polygon)[:, 1].min()
                if px == 0 and x != 0:
                    offenders.append(f"{vid} frame {i} {hand_type}: polygon x0=0 but box x={x}")
                if py == 0 and y != 0:
                    offenders.append(f"{vid} frame {i} {hand_type}: polygon y0=0 but box y={y}")
        if len(offenders) >= 3:
            break
    shown = "; ".join(offenders[:3])
    assert not offenders, f"1-based clamp drops edge pixels ({len(offenders)} cases), e.g. {shown}"


def test_box_matches_the_pixels_the_mask_actually_paints(videos):
    """The box must cover exactly the pixel extent get_segmentation_mask rasterises.

    This replaces a weaker, and in fact unsatisfiable, check that the box contains
    the raw float polygon. It cannot:

      * get_segmentation_mask rasterises through np.int32, which truncates, so the
        painted hand already ends at floor(max). A box that ceils would be one pixel
        wider than the hand that exists.
      * 6.8% of the dataset's polygons run up to 0.99 px outside the 1280x720 frame,
        and a box clamped to the frame can never contain those.

    Agreement between the box and the mask is the property that actually matters --
    they are two views of one annotation and must not drift apart.
    """
    mismatches = []
    for vi in (0, 11, 23, 47):
        video = videos.iloc[vi]
        name = str(video.loc["video_id"][0])
        frames = video.loc["labelled_frames"][0]
        for fi in (0, 37, 88):
            boxes = get_bounding_boxes(video, fi)
            for slot, box in zip((1, 2, 3, 4), boxes):
                polygon = frames[fi][slot]
                if polygon is None or np.asarray(polygon).size == 0:
                    continue
                painted = np.int32(np.asarray(polygon, dtype=float))
                expected_x = max(0, min(1279, int(painted[:, 0].min())))
                expected_y = max(0, min(719, int(painted[:, 1].min())))
                if (box[0], box[1]) != (expected_x, expected_y):
                    mismatches.append(
                        f"{name} frame {fi} slot {slot}: box starts "
                        f"({box[0]}, {box[1]}), mask starts ({expected_x}, {expected_y})"
                    )
    assert mismatches == [], "; ".join(mismatches[:6])

def test_mask_shape_and_dtype(meta):
    mask = get_segmentation_mask(meta.df.iloc[0], 0, "all")
    assert mask.shape == (FRAME_H, FRAME_W, 3)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}


def test_all_is_the_union_of_the_individual_hands(meta):
    """On a frame where every hand is annotated, 'all' == my_left | my_right |
    your_left | your_right, and 'mine' / 'yours' are the two halves."""
    video = meta.df.iloc[5]
    i = find_frame(video, (True, True, True, True))
    assert i is not None

    singles = [get_segmentation_mask(video, i, h)[:, :, 0] > 0 for h in HAND_SLOTS]
    union = np.logical_or.reduce(singles)

    assert np.array_equal(get_segmentation_mask(video, i, "all")[:, :, 0] > 0, union)
    assert np.array_equal(
        get_segmentation_mask(video, i, "mine")[:, :, 0] > 0, singles[0] | singles[1]
    )
    assert np.array_equal(
        get_segmentation_mask(video, i, "yours")[:, :, 0] > 0, singles[2] | singles[3]
    )
    assert union.sum() > 0


def test_all_returns_a_blank_mask_when_no_hand_is_annotated(meta):
    """The np.any() presence guard on the 'all' branch works: a frame with no
    hands at all yields zeros rather than an error."""
    for idx in range(len(meta.df)):
        video = meta.df.iloc[idx]
        i = find_frame(video, (False, False, False, False))
        if i is None:
            continue
        assert not get_segmentation_mask(video, i, "all").any()
        return
    pytest.skip("no fully unannotated frame in the dataset")


def test_specific_hand_type_is_blank_when_that_hand_is_absent(meta):
    """get_segmentation_mask.py line 31 reads

        if (hand_type == 'my_left' or hand_type == 'mine' or hand_type == 'all'
                and np.any(...)):

    which Python parses as `a or b or (c and d)`.  The np.any() presence guard
    therefore only protects the 'all' branch: asking for a hand that is not in
    the frame skips the guard and hands an empty polygon to cv2.fillPoly.

    Expected: an all-zero mask, exactly as 'all' gives for a blank frame.
    """
    failures = []
    for idx in range(len(meta.df)):
        video = meta.df.iloc[idx]
        vid = scalar(video.loc["video_id"])
        for slot, hand_type in enumerate(HAND_SLOTS, start=1):
            pattern = [True, True, True, True]
            pattern[slot - 1] = False
            i = find_frame(video, pattern)
            if i is None:
                continue
            try:
                mask = get_segmentation_mask(video, i, hand_type)
            except Exception as exc:  # noqa: BLE001 - characterising the defect
                name = f"{type(exc).__module__}.{type(exc).__name__}"
                failures.append(f"{vid} frame {i} {hand_type}: {name}")
                continue
            if mask.any():
                failures.append(f"{vid} frame {i} {hand_type}: non-blank mask")
        if len(failures) >= 4:
            break
    assert failures == [], "absent hand + specific hand_type: " + "; ".join(failures)


def test_mine_and_yours_survive_a_half_empty_frame(meta):
    """'mine' with only one of the two egocentric hands annotated is the
    common real case -- 2780 of the 4800 labelled frames are missing at least
    one hand -- and must return the mask of whichever hand is there."""
    video = meta.df.iloc[0]
    i = find_frame(video, (True, False, True, True))
    if i is None:
        i = find_frame(video, (False, True, True, True))
    assert i is not None, "expected a frame with exactly one of my hands"
    try:
        populated = bool(get_segmentation_mask(video, i, "mine").any())
    except Exception as exc:  # noqa: BLE001 - characterising the defect
        populated = f"{type(exc).__module__}.{type(exc).__name__}"
    assert populated is True, f"'mine' on frame {i} of video 0 gave: {populated}"


def test_unknown_hand_type_yields_a_blank_mask(meta):
    assert not get_segmentation_mask(meta.df.iloc[0], 0, "not_a_hand").any()


# --------------------------------------------------------------------------
# metadata vs. what is actually on disk
# --------------------------------------------------------------------------

@needs_dataset
def test_every_video_id_has_a_labelled_samples_folder(meta, repo):
    root = repo / "_LABELLED_SAMPLES"
    on_disk = {p.name for p in root.iterdir() if p.is_dir()}
    in_metadata = set(video_ids(meta.df))
    assert in_metadata - on_disk == set(), "metadata references missing folders"
    assert on_disk - in_metadata == set(), "folders with no metadata entry"


@needs_dataset
def test_every_folder_holds_exactly_100_labelled_frames(meta, repo):
    root = repo / "_LABELLED_SAMPLES"
    counts = {}
    for vid in video_ids(meta.df):
        counts[vid] = len(list((root / vid).glob("frame_*.jpg")))
    wrong = {k: v for k, v in counts.items() if v != 100}
    assert wrong == {}, wrong
