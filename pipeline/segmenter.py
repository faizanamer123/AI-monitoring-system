"""
pipeline/segmenter.py -- STAGE 2: which pixels inside the hand box are hand?

    WEBCAM -> frame
        -> 1. DETECT    HandDetection (box)
        -> 2. SEGMENT   HandMask          <-- this file
        -> 3. POSE      HandPose (21 pts)
        -> 4. TRACK     TrackedHand

WHY COLOUR IS ALLOWED TO WORK HERE
----------------------------------
Skin thresholding has a terrible reputation, and it deserves it -- on a whole webcam
frame it lights up the operator's face, a wooden desk, a cardboard box and half of a
beige wall, and there is no principled way to pick the hand out of that. But this
stage never sees a whole frame. It is handed a box that stage 1 has already decided
contains a hand, so the face is not in the picture, the hand usually occupies 30-70%
of the ROI, and the background is a small, local, mostly-uniform patch. The prior
"most of this crop is hand" is exactly what a colour threshold needs and never gets
when it is run globally. That is the whole idea behind this module.

The second thing worth exploiting: when stage 3 has already run, the 21 landmarks
tell us *where inside the ROI* the hand certainly is. Two uses follow from that, and
the hybrid method uses both:

  * as a spatial prior -- background skin (the other player's arm creeping into the
    corner of the box, a wooden table) is cut away because it is outside the hull;
  * as a colour sampler -- pixels well inside the hull are hand by construction, so
    we can fit the Cr/Cb window to *this* hand under *this* light instead of relying
    on a literature-average skin box. EgoHands alone spans four actors, three
    locations and full sun to indoor tungsten; one fixed threshold cannot cover that,
    and a per-hand one does not have to.

THE METHODS
-----------
  "landmark"  convex hull of the 21 landmarks, dilated to cover finger thickness.
              Needs a pose. Immune to lighting and to skin tone; coarse at the
              silhouette, and convex by definition, so the gaps between spread
              fingers get filled in.
  "skin"      fixed YCrCb + HSV gates inside the ROI, morphological cleanup, then
              largest-connected-component so specks and background skin drop out.
              No pose needed. Follows the true silhouette when the light is kind and
              collapses when it is not.
  "hybrid"    DEFAULT. A generously dilated landmark prior, per-hand adaptive colour
              to refine its boundary, the eroded hull unioned back in so a shadowed
              palm is never carved out, and a coverage check that falls back to the
              plain hull when colour collapses entirely. Falls back to "skin" when no
              pose is available.

MEASURED, NOT CLAIMED
---------------------
Scored against the EgoHands per-hand segmentation polygons (get_segmentation_mask),
using get_bounding_boxes for the ROI. 449 annotated hands from 150 frames sampled
across all 48 videos, held out from the sample the parameters were tuned on. Poses
come from MediaPipe HandLandmarker, i.e. the same stage 3 the pipeline uses; it finds
a hand on 67% of these, and "posed" below is that subset -- the only population all
three methods can segment.

    method     mean IoU (301 posed hands)    mean IoU (all 449)    ms/hand (median/p95)
    landmark             0.724                    n/a *                0.07 / 0.12
    skin                 0.722                   0.704                 0.95 / 2.0
    hybrid               0.750                   0.723                 1.50 / 3.4
                                            * needs a pose; cannot score the rest

Hybrid wins by +0.026 IoU over landmark and +0.028 over skin. That gap is small but it
is consistent: it survives a change of frame sample (+0.026 on the tuning set too) and
it widens under a simulated detector box error of 8% translation and 12% scale, where
the three go 0.730 / 0.707 / 0.697 -- colour alone is what degrades fastest when the
box is wrong, and the landmark prior is what stops hybrid degrading with it.

Keep the size of these numbers in perspective. The do-nothing baseline -- return the
entire ROI and call all of it hand -- already scores 0.643 on the posed hands, because
a tight box around a hand IS mostly hand. Hybrid's 0.750 is therefore +0.107 over
doing nothing, not +0.75 from nothing, and any claim about this stage should be read
against that 0.643 rather than against zero. Note also that the ROI here is the
ground-truth box; a real stage-1 box is looser, which lowers the baseline and widens
every method's margin over it.

WHERE IT FAILS -- honestly
--------------------------
* The observer's OWN hands score 0.675 against the partner's 0.773. EgoHands' polygons
  for them run down the forearm to the bottom of the frame, and there is no landmark
  past the wrist, so the prior stops where the arm starts and colour has to carry the
  whole forearm on its own. A wrist-anchored arm extension would fix this and is not
  implemented.
* 5% of hands score below 0.5, and the worst score is 0.0 -- a hand that is mostly a
  dark silhouette against a bright window, where both the fixed and the adapted colour
  window are fitted to noise.
* The hybrid answer is bounded above by the dilated landmark prior. If stage 3 returns
  a badly wrong pose, hybrid cannot recover; it will confidently segment the wrong
  region. `prior_scale` sets how loose that ceiling is.
* Hands smaller than ~100x100 px score 0.62. Below that the morphology kernels are
  1-2 px and there is not enough colour evidence to beat the hull.
* Everything here assumes an uncalibrated colour camera and skin that is visible.
  Gloves, body paint, an IR camera or a hand behind glass all break the colour half;
  the min_cover guard then falls back to the landmark hull rather than returning
  nothing, which is degradation rather than failure, but it is degradation.

COORDINATES
-----------
Everything coming in (detection.box, pose.points) is FULL-FRAME pixels. HandMask.mask
is the one and only ROI-local array in the pipeline and it carries its origin with it,
so the conversion happens exactly twice in this file: once on the way in
(`_to_roi`) and once on the way out (HandMask(mask, origin)). Nothing in between
touches frame coordinates.

See also pipeline/types.py, pipeline/detector.py, pipeline/pose.py
"""

import time

import cv2
import numpy as np

# pipeline/ has no __init__.py and is imported as a namespace package, so the
# relative form works for `import pipeline.segmenter` from the repo root. The absolute
# fallback is for the case where this module has been loaded outside a package context
# (an ad-hoc importlib load, a notebook) but the repo root is on sys.path. Running the
# file as a bare script is not supported by either path, and does not need to be --
# nothing here is a command-line entry point.
try:
    from .types import CONNECTIONS, HandDetection, HandMask, HandPose
except ImportError:
    from pipeline.types import CONNECTIONS, HandDetection, HandMask, HandPose

METHODS = ("landmark", "skin", "hybrid")

# Fixed skin gates.
#
# YCrCb: the Chai & Ngan (1999) chrominance box, 133<=Cr<=173 and 77<=Cb<=127. It is
# the workhorse rule for skin because Cr/Cb separate skin from most of the world
# almost independently of how bright the pixel is, which is what you want when part
# of a hand is in shadow. The Y floor is ours: below ~35 the chrominance of a JPEG
# pixel is mostly quantisation noise and everything looks like skin.
YCRCB_LOWER = np.array([35, 133, 77], dtype=np.uint8)
YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

# HSV: skin hue wraps around red, hence two ranges. The saturation floor rejects grey
# and white surfaces (paper, walls) that survive the Cr/Cb box; the ceiling rejects
# fully saturated reds (a red card, a jumper) that skin never reaches.
HSV_LOWER_A = np.array([0, 25, 50], dtype=np.uint8)
HSV_UPPER_A = np.array([25, 190, 255], dtype=np.uint8)
HSV_LOWER_B = np.array([160, 25, 50], dtype=np.uint8)
HSV_UPPER_B = np.array([180, 190, 255], dtype=np.uint8)


def _odd(value, minimum=3):
    """Nearest odd integer >= minimum. OpenCV structuring elements want odd sizes."""
    value = int(round(value))
    if value < minimum:
        value = minimum
    return value if value % 2 else value + 1


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _clamp_roi(box, frame_shape, pad=0.0):
    """Turn a full-frame detection box into a clamped integer ROI rectangle.

    `pad` is a fraction of the box's own size, applied on every side. A detector box
    is rarely pixel-tight -- fingertips get clipped when the hand is moving -- so a
    little slack costs a few background pixels and buys back real hand. It is a
    fraction rather than a constant because a hand ROI in this pipeline ranges from
    ~60 px (a partner's hand across the table) to ~600 px (your own hand at the
    lens), and a fixed 10 px pad means opposite things at those two scales.

    Returns (x1, y1, x2, y2) with x2/y2 exclusive, already clipped to the frame.
    """
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    if pad:
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    x1 = int(max(0, min(width, np.floor(x1))))
    y1 = int(max(0, min(height, np.floor(y1))))
    x2 = int(max(0, min(width, np.ceil(x2))))
    y2 = int(max(0, min(height, np.ceil(y2))))
    return x1, y1, max(x1, x2), max(y1, y2)


def _to_roi(points, origin):
    """Full-frame points -> ROI-local points. The only inbound coordinate change."""
    return np.asarray(points, dtype=np.float32) - np.asarray(origin, dtype=np.float32)


def _downscale(roi, points_roi, work_size):
    """Shrink the ROI so the colour work runs at a bounded resolution.

    Your own hand near the lens fills a 600x700 crop; a partner's hand across the
    table fills 70x60. Every colour and morphology operation here is area-bound, so
    without a cap the same code costs 0.4 ms on one hand and 40 ms on the next and the
    pipeline stutters exactly when a hand is closest and most interesting.

    Capping the long side at `work_size` makes the cost per hand flat. It is close to
    free in accuracy because nothing downstream is finer than the ~2 px of a
    morphological kernel anyway, and INTER_AREA averaging is a mild denoise that the
    colour threshold actually benefits from -- JPEG chroma noise is what produces the
    speckle that the opening step then has to remove.

    Returns (small_roi, scaled_points, factor); factor is 1.0 when no resize happened.
    """
    height, width = roi.shape[:2]
    longest = max(height, width)
    if work_size <= 0 or longest <= work_size:
        return roi, points_roi, 1.0
    factor = work_size / float(longest)
    small = cv2.resize(roi, (max(1, int(round(width * factor))),
                             max(1, int(round(height * factor)))),
                       interpolation=cv2.INTER_AREA)
    scaled = None if points_roi is None else points_roi * factor
    return small, scaled, factor


def _center_ellipse(shape, frac):
    """A filled ellipse covering the middle `frac` of an ROI, as a pose-free seed."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    height, width = shape[:2]
    cv2.ellipse(mask, (width // 2, height // 2),
                (max(1, int(width * frac / 2)), max(1, int(height * frac / 2))),
                0, 0, 360, 255, -1)
    return mask


def _upscale_mask(mask, shape):
    """Take a mask computed on a shrunken ROI back to full ROI resolution.

    Linear interpolation then a mid-level threshold, not nearest neighbour: nearest
    leaves the visible staircase of the working resolution on the silhouette, while
    the linear ramp cut at 128 lands the boundary on the sub-pixel position the
    downscaled evidence actually implies.
    """
    height, width = shape[:2]
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.where(resized >= 128, np.uint8(255), np.uint8(0))


def hand_scale(points):
    """A length in pixels that says how big this hand is, from its landmarks.

    Used to size every dilation and every morphological kernel, so that the same
    configuration works on a 60 px hand and a 600 px one. Palm length (wrist -> middle
    MCP) and palm width (index MCP -> pinky MCP) are the two most stable spans on a
    hand: unlike the fingers they barely foreshorten and they never fold away. Taking
    the max of the two keeps the estimate sane when the palm is edge-on to the camera
    and one of them collapses.
    """
    points = np.asarray(points, dtype=np.float32)
    # A pose shorter than the 21-landmark contract, or carrying NaN/inf, must not reach
    # the arithmetic below: points[17] would IndexError and a NaN would propagate into
    # int(round(...)) downstream and raise "cannot convert float NaN to integer",
    # killing the live loop. Return a neutral scale and let the caller fall back.
    if points.shape[0] < 18 or not np.isfinite(points[[0, 5, 9, 17]]).all():
        return 1.0
    palm_length = float(np.linalg.norm(points[9] - points[0]))
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    scale = max(palm_length, palm_width, 1.0)
    return scale if np.isfinite(scale) else 1.0


def landmark_hull_mask(points_roi, shape, radius=0, style="hull"):
    """Rasterise the 21 landmarks into an ROI-local uint8 {0,255} mask.

    The landmarks are joint *centres*. The real hand extends half a finger's thickness
    beyond every one of them, plus the fingertip pad past landmark 4/8/12/16/20 and
    the heel of the palm past the wrist, so the raw hull is systematically too small
    and is always dilated outward by `radius`.

    style="hull" is the convex hull, and it is what the "landmark" method returns. Its
    known weakness is written into its name: a hand with spread fingers is not convex,
    so the hull floods the gaps between the fingers with false positives.

    style="skeleton" instead strokes the 21 CONNECTIONS as thick lines over a filled
    palm polygon. That is non-convex, so it tracks a splayed hand much better, and it
    is the default shape for the hybrid *prior* for a reason that only applies there:
    the prior is a ceiling, colour can only ever remove pixels from it, so hull area
    spilling into the finger gaps is a problem colour then has to solve and sometimes
    cannot. Measured on EgoHands it is also 30% cheaper, because the stroke is thin
    where the hull's dilation is not. It is NOT used for the "landmark" method, which
    is specified as the dilated convex hull.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    points = np.asarray(points_roi, dtype=np.float32)
    if points.shape[0] < 3 or not np.isfinite(points).all():
        return mask

    if style == "skeleton":
        thickness = max(1, int(round(radius)))
        palm = np.int32(points[[0, 1, 5, 9, 13, 17]])
        cv2.fillConvexPoly(mask, palm, 255)
        for a, b in CONNECTIONS:
            cv2.line(mask, tuple(np.int32(points[a])), tuple(np.int32(points[b])),
                     255, thickness=2 * thickness + 1, lineType=cv2.LINE_8)
        # a fingertip is a rounded cap, not a line end
        for tip in (4, 8, 12, 16, 20, 0):
            cv2.circle(mask, tuple(np.int32(points[tip])), thickness, 255, -1)
        return mask

    hull = cv2.convexHull(np.int32(points))
    cv2.fillConvexPoly(mask, hull, 255)
    if radius >= 1:
        # Stroke the hull outline at width 2r+1 rather than cv2.dilate-ing the filled
        # polygon. The two give the same shape to within a pixel (a filled convex
        # region grown by a disc IS itself plus a band of width r along its boundary),
        # but dilation costs O(area * kernel) and the stroke costs O(perimeter * r).
        # On a 600 px own-hand ROI that is the difference between 8 ms and 0.2 ms,
        # which decides whether this stage fits in its frame budget at all.
        cv2.polylines(mask, [hull], True, 255,
                      thickness=2 * int(round(radius)) + 1, lineType=cv2.LINE_8)
    return mask


def fixed_skin_mask(roi_bgr):
    """Fixed-threshold skin mask: YCrCb box AND HSV box, ROI-local uint8 {0,255}.

    AND, not OR. Each gate alone has a characteristic failure -- YCrCb passes anything
    warm and unsaturated (bare wood, cardboard, beige walls, all of which EgoHands has
    in quantity), HSV passes anything reddish including deep shadow -- and their false
    positives are largely uncorrelated, so requiring both agree costs a little recall
    on the darkest hand pixels and removes most of the background.
    """
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(ycrcb, YCRCB_LOWER, YCRCB_UPPER)
    hue = cv2.inRange(hsv, HSV_LOWER_A, HSV_UPPER_A)
    cv2.bitwise_or(hue, cv2.inRange(hsv, HSV_LOWER_B, HSV_UPPER_B), dst=hue)
    return cv2.bitwise_and(mask, hue)


def adaptive_skin_mask(roi_bgr, seed_mask, k=3.0, min_half=5.0, max_half=14.0,
                       min_seed=60, max_samples=4000):
    """Fit the Cr/Cb window to THIS hand from pixels that are certainly hand.

    `seed_mask` marks pixels we already believe are hand -- in practice the eroded
    landmark hull. Their median Cr/Cb is this hand's skin colour under this light, and
    a robust spread around it gives the window. The point is that a fixed literature
    box has to be wide enough for every human being and every illuminant at once,
    which makes it wide enough to also admit the table; a box fitted to one hand in
    one frame can be several times narrower and still keep the whole hand.

    Spread uses the median absolute deviation, scaled by 1.4826 to read as a standard
    deviation for normal data. MAD rather than std because the seed is not pure: the
    eroded hull still catches the odd sliver of background between two fingers, and a
    handful of dark outliers would inflate std enough to reopen the window we are
    trying to narrow. The half-width is clamped: too narrow and a slight gradient
    across the palm splits the hand in two, too wide and we are back to the fixed box.

    Returns None when the seed is too small to fit anything -- the caller should fall
    back to `fixed_skin_mask` rather than trust a window fitted to 12 pixels.
    """
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    samples = ycrcb[seed_mask > 0]
    if samples.shape[0] < min_seed:
        return None
    if samples.shape[0] > max_samples:                       # fitting 2 medians does
        step = samples.shape[0] // max_samples + 1           # not need 300k pixels
        samples = samples[::step]

    lower = np.empty(3, dtype=np.uint8)
    upper = np.empty(3, dtype=np.uint8)
    lower[0], upper[0] = YCRCB_LOWER[0], 255                 # keep only the Y floor
    for channel in (1, 2):
        values = samples[:, channel].astype(np.float32)
        centre = float(np.median(values))
        half = 1.4826 * float(np.median(np.abs(values - centre))) * k
        half = float(np.clip(half, min_half, max_half))
        lower[channel] = int(max(0, round(centre - half)))
        upper[channel] = int(min(255, round(centre + half)))
    return cv2.inRange(ycrcb, lower, upper)


def clean_mask(mask, open_size, close_size):
    """Open then close. Order matters and it is this way round on purpose.

    Opening first deletes the isolated specks -- a few pixels of a wooden table that
    happened to land inside the colour window. Closing afterwards fills the pinholes
    inside the hand (specular highlights on knuckles, a ring, a shadow crease). Doing
    it the other way round would first *grow* every speck into its neighbours and then
    be unable to open the merged blob away.
    """
    if open_size >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(open_size))
    if close_size >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(close_size))
    return mask


def largest_component(mask, seed_mask=None, min_area_frac=0.0):
    """Keep one blob: the one the seed points at, else the biggest.

    A skin threshold inside a hand ROI typically returns the hand plus a couple of
    background fragments (the far edge of an arm, a patch of table). Exactly one of
    them is the hand, and connectivity is the cheapest way to say which. When a seed
    is supplied -- hybrid has one, the hull interior -- the component holding the most
    seed pixels wins, which is strictly better than area: a hand half-occluded by a
    playing card can easily be smaller than the forearm fragment beside it.

    min_area_frac drops the result entirely if the winner covers less than that
    fraction of the ROI, i.e. we found nothing worth calling a hand.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    if seed_mask is not None and np.any(seed_mask):
        seeded = labels[seed_mask > 0]
        seeded = seeded[seeded > 0]
        if seeded.size:
            best = int(np.bincount(seeded, minlength=count)[1:].argmax()) + 1
        else:
            best = int(areas.argmax()) + 1
    else:
        best = int(areas.argmax()) + 1

    if stats[best, cv2.CC_STAT_AREA] < min_area_frac * mask.size:
        return np.zeros_like(mask)
    return np.where(labels == best, np.uint8(255), np.uint8(0))


def mask_iou(a, b_binary_full_frame, frame_shape):
    """Intersection over union between a HandMask and a full-frame binary mask.

    This is the module's scoring function and it lives here rather than in the tests
    because the ROI/full-frame conversion is the part that goes wrong: comparing an
    ROI-local mask against a frame-sized ground truth by shape alone silently scores
    the wrong region. `a` is lifted to frame space through its own origin, which is
    the only correct way to do it.

    `b_binary_full_frame` may be (H, W) or (H, W, 3) -- EgoHands' own
    get_segmentation_mask returns 720x1280x3 -- and any non-zero value counts as set.
    Returns 0.0 when the union is empty, i.e. an empty prediction scores nothing even
    against an empty target; there is no hand to agree about.
    """
    predicted = a.to_frame(frame_shape) > 0
    truth = np.asarray(b_binary_full_frame)
    if truth.ndim == 3:
        truth = truth[:, :, 0]
    truth = truth > 0
    if truth.shape != predicted.shape:
        raise ValueError(
            f"ground-truth mask {truth.shape} does not match frame {predicted.shape}; "
            "both must be full-frame"
        )
    union = np.count_nonzero(predicted | truth)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(predicted & truth) / union)


class HandSegmenter:
    """Stage 2: turn (frame, HandDetection[, HandPose]) into a HandMask.

    Stateless between calls and cheap to construct, so a runner can keep one instance
    and call it once per hand per frame. Every tuning constant is a keyword here
    rather than a module constant so the runner can trade precision for recall without
    editing this file.

    Parameters worth understanding:
      method           "hybrid" (default), "landmark" or "skin".
      pad              ROI padding as a fraction of box size; slack for a box that
                       clipped a fingertip. Default 0, because on EgoHands it only
                       ever cost IoU (0.750 at 0.0, 0.691 at 0.10, 0.679 at 0.50 --
                       every padded pixel is background the segmenter then has to
                       reject). Raise it only if your stage 1 is known to cut hands
                       off rather than to over-box them.
      dilate_frac      landmark dilation radius, as a fraction of hand_scale().
      prior_scale      how much *more* the hybrid prior is dilated than "landmark"
                       would be. The prior is a ceiling on the answer, so it is
                       deliberately loose; colour tightens it back down. 2.0 is a
                       compromise, not an optimum: with the ground-truth box a much
                       looser prior scores better (0.766 at 20.0 vs 0.750 at 2.0)
                       because the box is already doing the containing, but that
                       collapses to 0.52 once the box is 50% too big, where 2.0 holds
                       at 0.68. Tune it to how much you trust stage 1.
      prior_style      "skeleton" (default) or "hull" for the shape of that prior.
      core_frac        hull erosion, as a fraction of hand_scale(), producing the
                       certain-hand region used to seed both the colour model and the
                       component choice.
      min_cover        if the colour mask alone keeps less than this fraction of the
                       plain landmark hull, colour is judged to have collapsed (a
                       glove, a hand lit blue, heavy motion blur) and the hull is
                       returned instead. It did not fire once in 827 EgoHands hands
                       at the default 0.25 -- it is insurance against conditions this
                       dataset does not contain, not a tuned parameter, and it is
                       reachable (0.55 fires) rather than dead.
    """

    def __init__(self, method="hybrid", pad=0.0, dilate_frac=0.15, prior_scale=2.0,
                 core_frac=0.28, adaptive=True, prior_style="skeleton",
                 open_frac=0.018, close_frac=0.02, min_area_frac=0.01,
                 min_cover=0.25, work_size=192, adaptive_k=3.0, adaptive_max_half=14.0,
                 center_seed=0.5):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self.method = method
        self.pad = float(pad)
        self.dilate_frac = float(dilate_frac)
        self.prior_scale = float(prior_scale)
        self.core_frac = float(core_frac)
        self.adaptive = bool(adaptive)
        self.prior_style = prior_style
        self.open_frac = float(open_frac)
        self.close_frac = float(close_frac)
        self.min_area_frac = float(min_area_frac)
        self.min_cover = float(min_cover)
        self.work_size = int(work_size)
        self.adaptive_k = float(adaptive_k)
        self.adaptive_max_half = float(adaptive_max_half)
        self.center_seed = float(center_seed)
        self.last_ms = 0.0          # wall time of the most recent segment(), for HUDs

    # ---------------------------------------------------------------- public API

    def segment(self, frame_bgr, detection, pose=None):
        """Segment one hand. Returns a HandMask whose origin is the ROI's top-left.

        `detection.box` and `pose.points` are full-frame; the returned mask is
        ROI-local, as HandMask documents. A degenerate box (zero area, or entirely
        outside the frame) yields an empty mask rather than an exception -- stage 1
        occasionally emits one at a frame edge and a live pipeline must not die on it.
        """
        # One guard for every pose-dependent branch below. MediaPipe can emit NaN
        # landmarks on a partly-occluded hand, and a pose that is short or non-finite
        # is worse than no pose at all: it produces a plausible-looking hull in the
        # wrong place. Treat it as absent and fall back to the colour-only path.
        if pose is not None:
            points = np.asarray(getattr(pose, "points", None), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 18 or not np.isfinite(points).all():
                pose = None

        start = time.perf_counter()
        try:
            if frame_bgr is None or frame_bgr.size == 0:
                raise ValueError("frame_bgr is empty")
            box = detection.box if isinstance(detection, HandDetection) else detection
            x1, y1, x2, y2 = _clamp_roi(box, frame_bgr.shape, self.pad)
            origin = (x1, y1)
            if x2 <= x1 or y2 <= y1:
                return HandMask(np.zeros((0, 0), dtype=np.uint8), origin)

            roi = frame_bgr[y1:y2, x1:x2]
            if roi.ndim == 2:                      # tolerate a greyscale frame
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            points_roi = None
            if pose is not None:
                points = pose.points if isinstance(pose, HandPose) else pose
                points_roi = _to_roi(points, origin)

            if self.method == "landmark":
                if points_roi is None:
                    raise ValueError(
                        'method="landmark" needs a pose; pass one, or use '
                        '"hybrid"/"skin" which work without landmarks'
                    )
                mask = self._landmark(roi, points_roi)
            elif self.method == "skin":
                mask = self._skin(roi)
            else:
                mask = (self._hybrid(roi, points_roi) if points_roi is not None
                        else self._skin(roi))
            return HandMask(mask, origin)
        finally:
            self.last_ms = (time.perf_counter() - start) * 1000.0

    def segment_many(self, frame_bgr, detections, poses=None):
        """Convenience for a whole frame. Poses are positional, None where missing."""
        poses = poses if poses is not None else [None] * len(detections)
        return [self.segment(frame_bgr, d, p) for d, p in zip(detections, poses)]

    # --------------------------------------------------------------- the methods

    def _landmark(self, roi, points_roi):
        """Convex hull of the landmarks, dilated by dilate_frac * hand_scale."""
        radius = self.dilate_frac * hand_scale(points_roi)
        return landmark_hull_mask(points_roi, roi.shape, radius, style="hull")

    def _skin(self, roi):
        """Fixed thresholds -> morphology -> one component. No pose required.

        The component is chosen by area unless `center_seed` is on, in which case a
        central ellipse breaks the tie instead. That ellipse is the pose-free version
        of the hybrid core: stage 1 centres its box on the hand, so the middle of the
        ROI is the best guess at "certainly hand" that exists without landmarks. It
        matters when the ROI clips a forearm along one edge -- the forearm blob can
        genuinely be the larger of the two, and area alone then picks the arm.
        """
        small, _, _ = _downscale(roi, None, self.work_size)
        scale = float(np.sqrt(small.shape[0] * small.shape[1]))
        mask = fixed_skin_mask(small)
        mask = clean_mask(mask, _odd(self.open_frac * scale),
                          _odd(self.close_frac * scale))
        seed = _center_ellipse(small.shape, self.center_seed) if self.center_seed else None
        mask = largest_component(mask, seed, self.min_area_frac)
        return _upscale_mask(mask, roi.shape)

    def _hybrid(self, roi, points_roi):
        """Hull prior + per-hand colour refinement + a collapse guard.

        Five steps, each undoing a specific failure of the two pure methods:
          1. prior   -- generously dilated hull. Caps the answer, so background skin
                        outside the hand (the partner's arm at the box edge) can never
                        be selected however skin-coloured it is.
          2. core    -- eroded hull. Certain hand: seeds the colour model and, later,
                        the component choice.
          3. colour  -- adaptive window fitted on the core, else the fixed gates.
                        Intersected with the prior; this is where the silhouette gets
                        its accuracy back, since colour follows the true finger edges
                        that the hull cuts across.
          4. guard   -- if colour on its own kept almost none of the hull, colour has
                        failed on this frame (a glove, a hand lit blue, heavy motion
                        blur) and the plain hull is returned instead. The check
                        happens HERE, before step 5, and that ordering is the whole
                        point: once the core has been unioned in, the result is
                        guaranteed to cover the core and the check can never fail, so
                        a guard placed after step 5 is dead code that looks alive.
          5. repair  -- union the core back in (a palm in shadow must not be carved
                        out of its own hand), keep the core's component, close the
                        pinholes left by highlights and rings.
        """
        small, points, _ = _downscale(roi, points_roi, self.work_size)
        scale = hand_scale(points)
        radius = self.dilate_frac * scale
        prior = landmark_hull_mask(points, small.shape, radius * self.prior_scale,
                                   style=self.prior_style)
        plain = landmark_hull_mask(points, small.shape, radius, style="hull")

        core_radius = _odd(self.core_frac * scale)
        core = cv2.erode(plain, _ellipse(core_radius))
        if not np.any(core):                      # a thin, edge-on hand can erode away
            core = plain

        skin = (adaptive_skin_mask(small, core, k=self.adaptive_k,
                                   max_half=self.adaptive_max_half)
                if self.adaptive else None)
        if skin is None:
            skin = fixed_skin_mask(small)

        refined = cv2.bitwise_and(skin, prior)
        refined = clean_mask(refined, _odd(self.open_frac * scale * 2),
                             _odd(self.close_frac * scale * 2))

        plain_area = int(np.count_nonzero(plain))
        colour_inside_hull = int(np.count_nonzero(cv2.bitwise_and(refined, plain)))
        if plain_area and colour_inside_hull < self.min_cover * plain_area:
            return _upscale_mask(plain, roi.shape)

        cv2.bitwise_or(refined, core, dst=refined)
        refined = largest_component(refined, core, 0.0)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE,
                                   _ellipse(_odd(self.close_frac * scale * 2)))
        return _upscale_mask(refined, roi.shape)
