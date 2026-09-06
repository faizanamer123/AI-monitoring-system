"""
pipeline/gestures.py -- turn 21 hand landmarks into a named gesture.

    WEBCAM -> frame
        -> 1. DETECT   HandDetection
        -> 2. SEGMENT  HandMask
        -> 3. POSE     HandPose      <-- this module reads the 21 points
        -> 4. TRACK    TrackedHand   <-- and the tracker stores what we return
        -> GESTURE  ("open_palm", 0.93)

Public API:

    classify(pose, detection=None) -> (name, confidence)
    finger_states(pose)            -> {"thumb": bool, "index": bool, ...}
    pinch_distance(pose)           -> float, thumb tip to index tip / hand size

    extension_scores(pose)         -> the same five fingers as continuous [0, 1]
    hand_scale(pose)               -> the normalising length, in pixels

Recognised: open_palm, fist, point, peace, thumbs_up, pinch, none.


WHY EVERYTHING IS A RATIO
-------------------------
A hand 30 cm from the webcam is roughly 250 px across; the same hand at arm's
length is roughly 60 px. Any rule written in pixels ("the tip is more than 80 px
from the wrist") is therefore correct at exactly one distance from the camera and
wrong everywhere else. So not one number in this file is a pixel threshold: every
geometric quantity is divided by hand_scale() first, which makes the whole module
scale invariant by construction rather than by luck. Rotation invariance comes
free from the same choice -- distances and dot products between landmarks do not
care how the hand is rolled in the image, so a fist stays a fist when the wrist
turns, which is the case a "is the fingertip above the knuckle in image y?" rule
gets wrong the moment somebody tilts their hand.

WHY THE THUMB IS NOT A FINGER
-----------------------------
The four fingers curl *into* the palm, so a curled tip ends up well behind its own
knuckle and the chain rolls up. The thumb folds *sideways across* the palm,
pivoting at the wrist end and staying nearly straight, so a tucked thumb barely
moves on either measurement -- a fifth of the reach margin a curled finger gives,
and three quarters of its straightness kept. Applied to real landmarks, the
four-finger test calls 98% of thumbs "extended" whatever they are doing, which
turns every fist into a thumbs_up. The thumb gets its own rule, built on how far
sideways it sits rather than how far out. See _thumb_score() for the derivation.

THE KNOWN LIMIT: FORESHORTENING
-------------------------------
A hand pointing away from the camera projects its fingers to almost nothing, and
in x and y alone that looks very much like a hand with its fingers curled. The
straightness half of the extension test carries an open palm well past 60 degrees
of tilt (see test_known_limit_foreshortening), but at some angle the two really
are the same picture and no 2D rule can separate them.

The fix is the depth channel: HandPose.depth exists for it. This module does not
read it, deliberately. pipeline/pose.py leaves depth as None today, so the code
would be dead; and if it starts filling it, the units are the pose stage's to
define -- MediaPipe's raw z is normalised by image width, not pixels -- and
silently mixing a z in one unit with an x in another would produce plausible
wrong answers rather than an error. When depth arrives with a documented unit,
the place to use it is _four_finger_score, which becomes a 3D distance.

See also pipeline/types.py (the shared dataclasses), pipeline/pose.py (produces
HandPose), run_pipeline.py (the runner that calls this).
"""

import numpy as np

from pipeline.types import FINGERS, WRIST

# ---------------------------------------------------------------------------
# Landmark chains. Each finger is (MCP, PIP, DIP, TIP) -- knuckle to fingertip.
# The thumb's four are (CMC, MCP, IP, TIP); it has one fewer bone than it looks,
# which is exactly why it needs its own rule below.
# ---------------------------------------------------------------------------
THUMB_TIP, INDEX_TIP = 4, 8
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
THUMB_IP = 3

# The four palm spans that hand_scale() measures. All four are bone-to-bone
# across the rigid part of the hand, so none of them changes when fingers curl.
PALM_SPANS = ((WRIST, MIDDLE_MCP), (INDEX_MCP, PINKY_MCP),
              (WRIST, INDEX_MCP), (WRIST, PINKY_MCP))

# --- thresholds, all in units of hand_scale (dimensionless) ----------------
# Each pair is the soft band of a linear ramp: at or below LO the finger scores
# 0 (curled), at or above HI it scores 1 (extended), and in between the score --
# and therefore the reported confidence -- degrades smoothly. A borderline pose
# SHOULD come back with a low number instead of a confident coin flip.

# How much further from the wrist the tip sits than the middle knuckle does.
REACH_LO, REACH_HI = -0.10, 0.35
# How straight the finger is: tip-to-knuckle distance over the length of the
# three bones. 1.0 is a ruler, ~0.3 is a finger rolled up into the palm.
STRAIGHT_LO, STRAIGHT_HI = 0.55, 0.88

# Thumb, signal 1: how far past the index knuckle the tip sits, measured along
# the knuckle line, away from the pinky.
ABDUCT_LO, ABDUCT_HI = -0.05, 0.30
# Thumb, signal 2: does the tip stick further out of the hand than its own IP
# joint, measuring from the far corner of the palm.
SPLAY_LO, SPLAY_HI = -0.02, 0.22

# Pinch: thumb tip to index tip. Below ON the tips are touching, above OFF they
# are plainly apart. Note that "touching" is never 0 -- the landmarks sit at the
# centres of the fingertips, so two pads pressed together still measure ~0.15.
PINCH_ON, PINCH_OFF = 0.30, 0.55
# ...but a closed fist ALSO parks the thumb tip next to the index tip, so the gap
# alone would call every fist a pinch. The separator is how far the index tip is
# from its OWN knuckle: rolled into a fist it is 0.30-0.44 palm widths, arched
# out into a pinch it is 0.53-0.82. Measured across curl and thumb-position
# sweeps of the articulated hand model in tests/test_pipeline_gestures.py.
PINCH_ARCH_LO, PINCH_ARCH_HI = 0.40, 0.58

# Below this, no template fits well enough to name the pose.
MIN_MATCH = 0.62

# A hand smaller than this many pixels across has landmark noise comparable to
# the distances we measure, so the answer would be a random name at high
# confidence. Refuse instead. (This is the one pixel number in the file, and it
# is a "can this be measured at all" floor, not a gesture rule.)
MIN_SCALE_PX = 12.0

# --- gesture templates ------------------------------------------------------
# Which fingers a gesture wants extended. None means "don't care": people point
# with the thumb tucked in or cocked out and mean the same thing by it, so
# scoring the thumb there would only punish half the population.
GESTURE_TEMPLATES = {
    "open_palm": {"thumb": True,  "index": True,  "middle": True,  "ring": True,  "pinky": True},
    "fist":      {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False},
    "point":     {"thumb": None,  "index": True,  "middle": False, "ring": False, "pinky": False},
    "peace":     {"thumb": None,  "index": True,  "middle": True,  "ring": False, "pinky": False},
    "thumbs_up": {"thumb": True,  "index": False, "middle": False, "ring": False, "pinky": False},
}

GESTURES = ("open_palm", "fist", "point", "peace", "thumbs_up", "pinch", "none")

_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_points(pose):
    """Accept a HandPose or a bare (21, 2) array, and return float64 points.

    The runner hands us HandPose objects, but tests, notebooks and the tracker's
    replay path all naturally hold raw arrays, and forcing every caller to wrap
    an array in a dataclass just to ask a geometric question is friction for no
    benefit. Returns None when there is nothing measurable, so every public
    entry point can bail out the same way instead of raising into the frame loop
    -- a dropped landmark should cost one frame's gesture, not the process.
    """
    if pose is None:
        return None
    points = getattr(pose, "points", pose)
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] != 21 or points.shape[1] < 2:
        raise ValueError(f"expected 21 landmarks of 2+ coordinates, got {points.shape}")
    points = points[:, :2]
    if not np.isfinite(points).all():
        return None
    return points


def _ramp(value, low, high):
    """Linear 0..1 ramp, clipped. The soft middle is the whole point.

    A hard `value > threshold` would report full confidence for a pose sitting a
    hair over the line, which is how a gesture UI ends up flickering between two
    names on a hand that is not really doing either.
    """
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(min(max((value - low) / (high - low), 0.0), 1.0))


def _dist(points, a, b):
    return float(np.hypot(*(points[a] - points[b])))


def hand_scale(pose, detection=None):
    """The length everything else is divided by: the size of the rigid palm.

    Returns pixels, or 0.0 if the pose is unusable.

    The obvious choice is the wrist-to-middle-knuckle distance, and on a hand
    held up facing the camera it is fine. It falls apart in exactly the view this
    pipeline sees most. Measured on 123 MediaPipe hands from the EgoHands frames
    (egocentric, hands reaching away from the camera): that single span
    foreshortens to as little as 8 px on a hand 130 px wide, and normalised
    fingertip distances that should sit near 1.5 blow up past 16. Every threshold
    in this file would be meaningless on those frames.

    So we take the LARGEST of four rigid palm spans instead. Foreshortening
    squashes the palm along one axis at a time; the wrist-to-knuckle spans and
    the across-the-knuckles span point in different directions, so when one
    collapses another survives. Same 123 hands, normalised fingertip distance:

        wrist -> middle MCP    spread (cv) 0.63,  worst case 16.5
        max of the four spans  spread (cv) 0.32,  worst case  2.8

    `detection` is an optional fallback only. If the landmarks are degenerate but
    stage 1 gave us a box, half the box diagonal is a crude but finite stand-in,
    which keeps a bad pose from dividing by zero.
    """
    points = _as_points(pose)
    if points is None:
        scale = 0.0
    else:
        scale = max(_dist(points, a, b) for a, b in PALM_SPANS)

    if scale < MIN_SCALE_PX and detection is not None:
        box = getattr(detection, "box", None)
        if box is not None:
            x1, y1, x2, y2 = box
            scale = max(scale, 0.5 * float(np.hypot(x2 - x1, y2 - y1)))
    return float(scale)


# ---------------------------------------------------------------------------
# per-finger geometry
# ---------------------------------------------------------------------------
def _four_finger_score(points, chain, scale):
    """Continuous 0..1 extension for index / middle / ring / pinky.

    Two independent measurements, averaged, because each one alone has a blind
    spot:

    1. REACH -- how much further from the wrist the tip is than the PIP knuckle,
       over hand size. Extended, the tip is the far end of a lever and clears the
       knuckle easily; curled, it swings back inside and the difference goes
       negative. This is the "compare distances along the finger chain" test, and
       it is the more reliable of the two because it is measured against the
       wrist, a landmark the network localises well. Its blind spot: a finger
       pointing straight at the lens projects to almost nothing, and reach goes
       to zero whether it is curled or not.

    2. STRAIGHTNESS -- the tip-to-knuckle distance over the summed length of the
       three bones. A straight finger scores ~1.0 because the bones lie along one
       line; a rolled-up one scores ~0.3 because they cancel out. This is a
       property of the finger alone, so it survives the head-on view that defeats
       reach. Its blind spot: a finger held straight but folded flat at the big
       knuckle (fingers down, hand relaxed) still measures straight.

    They disagree only in the poses where one of them is lying, so averaging
    gives a score that is confidently 0 or 1 on real gestures and lands in the
    mushy middle exactly when the hand really is between two shapes. On the 492
    real finger measurements above the blend is sharply bimodal -- 88% of them
    land outside the 0.3-0.7 band.
    """
    mcp, pip, dip, tip = chain
    reach = (_dist(points, tip, WRIST) - _dist(points, pip, WRIST)) / scale

    bones = (_dist(points, mcp, pip) + _dist(points, pip, dip) + _dist(points, dip, tip))
    straight = _dist(points, tip, mcp) / bones if bones > 1e-9 else 0.0

    return 0.5 * _ramp(reach, REACH_LO, REACH_HI) + 0.5 * _ramp(straight, STRAIGHT_LO, STRAIGHT_HI)


def _thumb_score(points, scale):
    """Continuous 0..1 extension for the thumb, which needs its own rule.

    THE PROBLEM. The other four fingers curl into the palm, so "tip far from the
    wrist" separates open from closed cleanly. The thumb does not curl inward --
    it swings across the front of the palm, pivoting at the wrist end and staying
    nearly straight the whole way. Fold a thumb into a fist and its tip lands over
    the index and middle knuckles, still further from the wrist than a curled
    finger's tip, and with its own bones still in a line.

    So both halves of the four-finger test go quiet on it. Against a curled index
    finger, a tucked thumb keeps 72% straightness where the finger drops to 28%,
    and gives up only a fifth as much reach. Run that score on the thumb chain
    over the real EgoHands landmarks and it reports "extended" for 98% of hands
    -- it is a constant, not a measurement, and every fist it sees becomes a
    thumbs_up. The sideways rule below reports 54% on the same hands, and the two
    disagree on 43% of them.

    So the thumb is measured sideways instead of outward.

    SIGNAL 1, ABDUCTION. Take the knuckle line, index MCP -> pinky MCP, and ask
    how far past the index knuckle the thumb tip sits in the opposite direction.
    An extended thumb sticks out beyond the edge of the hand, so it is positive;
    a folded thumb lies over the palm, between the knuckles, so it is zero or
    negative. Using the hand's own knuckle line rather than the image axes is
    what keeps this rotation invariant, and because the line is defined index ->
    pinky it is identical for a left and a right hand, and for a mirrored frame.
    That matters: this pipeline flips the webcam image, which swaps handedness.

    SIGNAL 2, SPLAY. From the far corner of the palm (the pinky knuckle), is the
    thumb TIP further away than the thumb's own IP joint? Extended, the thumb
    points away from the hand and the tip is the far end, so yes. Folded across
    the palm, the tip has swung toward the pinky side and ends up nearer than the
    joint behind it, so no. This one uses no direction at all, only three
    distances, so it still works when the knuckle line foreshortens to a point
    and signal 1 loses its axis.

    Averaged, for the same reason as the four fingers: they fail in different
    views, and the average degrades to an honest "not sure" instead of a
    confident wrong answer.
    """
    knuckle = points[PINKY_MCP] - points[INDEX_MCP]
    span = float(np.hypot(*knuckle))
    if span > 1e-9:
        # negative dot product = tip lies on the far side of the index knuckle,
        # away from the pinky, i.e. sticking out of the side of the hand
        abduction = -float(np.dot(points[THUMB_TIP] - points[INDEX_MCP], knuckle / span)) / scale
        abduct_score = _ramp(abduction, ABDUCT_LO, ABDUCT_HI)
    else:
        abduct_score = None  # knuckle line degenerate: signal 2 carries the answer

    splay = (_dist(points, THUMB_TIP, PINKY_MCP) - _dist(points, THUMB_IP, PINKY_MCP)) / scale
    splay_score = _ramp(splay, SPLAY_LO, SPLAY_HI)

    if abduct_score is None:
        return splay_score
    return 0.5 * abduct_score + 0.5 * splay_score


def _scores(points, scale):
    """The five extension scores, given points and a scale already computed.

    Split out so classify() and extension_scores() cannot drift apart: a new
    signal added here reaches both, and neither pays for a second hand_scale().
    """
    scores = {"thumb": _thumb_score(points, scale)}
    for name in ("index", "middle", "ring", "pinky"):
        scores[name] = _four_finger_score(points, FINGERS[name], scale)
    return scores


def extension_scores(pose, detection=None):
    """Per-finger extension as continuous values in [0, 1], not booleans.

    finger_states() is the API the contract asks for, but thresholding to a bool
    throws away exactly the information a confidence needs: 0.51 and 0.99 are
    both True and only one of them deserves to be believed. classify() scores
    against these instead, which is why a half-curled hand reports a low number
    rather than a crisp wrong answer.

    Returns {} when the pose is unusable, so callers can test truthiness.
    """
    points = _as_points(pose)
    if points is None:
        return {}
    scale = hand_scale(points, detection)
    if scale < MIN_SCALE_PX:
        return {}

    return _scores(points, scale)


def finger_states(pose, detection=None):
    """Which fingers are extended: {"thumb": True, "index": False, ...}.

    The boolean view of extension_scores(). Returns all-False when the pose is
    unusable rather than raising, because in a 30 fps loop the useful behaviour
    for one bad frame is "no gesture this frame", not a traceback. Callers that
    need to tell "closed hand" from "no reading" should check
    extension_scores() != {} or the confidence coming out of classify().
    """
    scores = extension_scores(pose, detection)
    if not scores:
        return {name: False for name in _FINGER_NAMES}
    return {name: bool(score >= 0.5) for name, score in scores.items()}


def pinch_distance(pose, detection=None):
    """Thumb tip to index tip, divided by hand size. Dimensionless.

    ~0.1 is pinched shut, ~0.3 is the tips a fingertip's width apart, ~1.5 is a
    spread hand. Because it is normalised it means the same thing at any distance
    from the camera, which is what makes it usable directly as a UI axis -- a
    volume slider or a zoom factor -- and not just as a gesture flag.

    Returns inf for an unusable pose: no reading is not the same as zero, and inf
    makes every "is it pinched" comparison come out False by default instead of
    firing continuously on a hand the pose stage lost.
    """
    points = _as_points(pose)
    if points is None:
        return float("inf")
    scale = hand_scale(points, detection)
    if scale < MIN_SCALE_PX:
        return float("inf")
    return _dist(points, THUMB_TIP, INDEX_TIP) / scale


def _pinch_score(points, scale):
    """How much this pose looks like a pinch, 0..1.

    Pinch is deliberately NOT scored as a finger-state pattern. In a pinch the
    thumb and index are each half folded and touching each other, which the
    extended/curled vocabulary cannot express: score it as states and it comes
    out as a mediocre match to three different templates. So it is measured
    directly, from the gap between the two tips.

    The trap is the fist, which also parks the thumb tip within a fingertip's
    width of the index tip -- on the articulated model, a clenched hand's tip gap
    runs 0.08 to 0.35, straddling the whole pinch range. Gap alone would call
    every fist a pinch.

    What separates them is the SHAPE OF THE INDEX FINGER, not where the thumb is.
    Pinching, the index is arched forward and its tip stands off from its own
    knuckle; clenched, the tip is rolled back down against that knuckle. That
    distance, over hand size, is 0.30-0.44 for a fist and 0.53-0.82 for a pinch,
    with no overlap, so the gap score is gated by it. Measuring against the index
    knuckle rather than the wrist matters: the wrist version overlaps (0.72 fist
    vs 0.81 hard pinch) because a deep pinch pulls the fingertip back toward the
    palm too.
    """
    gap = _dist(points, THUMB_TIP, INDEX_TIP) / scale
    closeness = 1.0 - _ramp(gap, PINCH_ON, PINCH_OFF)

    arch = _dist(points, INDEX_TIP, INDEX_MCP) / scale
    out_front = _ramp(arch, PINCH_ARCH_LO, PINCH_ARCH_HI)

    return closeness * out_front


def _template_match(scores, template):
    """Mean agreement between the measured fingers and what a gesture wants.

    Per finger: 1 - |measured - wanted|. A finger measured at 0.9 against a
    wanted 1 contributes 0.9; measured at 0.5 -- genuinely ambiguous -- it
    contributes 0.5 whichever way the template leans, which is the property that
    makes the mean usable as a confidence. "Don't care" fingers are dropped from
    the mean rather than scored as agreeing, so they neither help nor hurt.
    """
    total, count = 0.0, 0
    for name, wanted in template.items():
        if wanted is None:
            continue
        total += 1.0 - abs(scores[name] - (1.0 if wanted else 0.0))
        count += 1
    return total / count if count else 0.0


def classify(pose, detection=None):
    """Name the gesture in this pose. Returns (name, confidence).

    name is one of GESTURES; confidence is a real number in [0, 1] measuring how
    cleanly the pose matches, NOT a constant. It is the mean per-finger agreement
    with the winning template, so a hand halfway between a fist and a point
    reports ~0.6 and a caller can threshold on it. A pose that fits nothing well
    returns ("none", how sure we are that it is nothing).

    `detection` is optional and only supplies a fallback hand size when the
    landmarks are too degenerate to measure one; the gesture decision itself is
    made entirely from the 21 points.

    Costs ~35 us per hand, so it is free next to the pose stage that feeds it.
    """
    points = _as_points(pose)
    if points is None:
        return "none", 0.0

    scale = hand_scale(points, detection)
    if scale < MIN_SCALE_PX:
        # too small to measure: the landmark noise is the same size as the
        # signal, so any name we returned would be a guess dressed up as a result
        return "none", 0.0

    scores = _scores(points, scale)

    best_name, best_score = "none", 0.0
    for name, template in GESTURE_TEMPLATES.items():
        match = _template_match(scores, template)
        if match > best_score:
            best_name, best_score = name, match

    pinch = _pinch_score(points, scale)
    if pinch > best_score:
        best_name, best_score = "pinch", pinch

    if best_score < MIN_MATCH:
        # Nothing fits. Report how sure we are of that: the worse the best
        # template fitted, the more confidently this is "none".
        return "none", float(min(max(1.0 - best_score, 0.0), 1.0))

    return best_name, float(min(max(best_score, 0.0), 1.0))
