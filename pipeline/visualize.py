"""
pipeline/visualize.py -- draw the pipeline's output back onto the frame.

    WEBCAM -> frame -> 1 DETECT -> 2 SEGMENT -> 3 POSE -> 4 TRACK -> [ visualize ] -> screen

Three entry points, smallest to largest:

    draw_hand(frame, track, show=...)      one hand: mask, trail, box, skeleton, label
    draw_hud(frame, tracks, fps, extra)    the corner readout: fps, count, movement
    render(frame, tracks, fps, show=...)   both, which is what the runner wants

The target look, from the user's sketch:

    +---------------------------------+
    |          GREEN MASK OVERLAY     |
    |       skeleton dots and lines   |
    |  Hand: 98%                      |
    |  Index: up 12 px                |
    |  Hand velocity: 45 px/s         |
    +---------------------------------+

DESIGN NOTES -- why this file looks the way it does:

* Every draw is IN PLACE on the caller's array, and the same array is returned. At
  1280x720 a defensive frame.copy() is 2.7 MB per call; at 30 fps that is 80 MB/s of
  pointless allocation. If you need the original, copy it yourself before calling.

* The mask is composited ONLY inside its own ROI. HandMask deliberately stores an ROI
  plus an origin instead of a full-frame array, and the drawing code has to honour
  that or it throws the saving away: blending a full 720x1280 plate costs ~2 ms per
  hand, blending a 250x250 one costs ~0.05 ms. Same picture, 40x the budget.

* The overlay is semi-transparent (alpha 0.4). A solid fill hides the hand, which is
  exactly the thing the user is looking at -- they want to check the mask against the
  fingers underneath, and a solid green blob answers no question at all.

* Text is drawn twice: a thick black pass, then a thin coloured pass. Webcam
  backgrounds swing from a white wall to a black shirt inside one frame, so any single
  colour is invisible somewhere. The outline makes the text readable on both.

* Colours are a deterministic function of track_id (see color_for_track). A colour
  that changes per frame makes two hands impossible to tell apart while they cross,
  which is the one moment tracking output is worth looking at.

COORDINATE RULE: every coordinate here is FULL-FRAME pixels. HandMask.mask is the one
exception in the whole pipeline -- it is ROI-local and carries its origin with it --
and draw_mask() is the only function that is allowed to know that.

See also pipeline/types.py (the contract), run_pipeline.py (the runner).
"""

import colorsys

import cv2
import numpy as np

try:                                    # normal package import
    from .types import CONNECTIONS, FINGERTIPS, WRIST
except ImportError:                     # running this file from inside pipeline/
    from pipeline.types import CONNECTIONS, FINGERTIPS, WRIST

# ---------------------------------------------------------------- tunables

# Per-hand layer names understood by draw_hand(). render() also takes "hud" (draw the
# corner readout, which it does anyway) and "nohud" (suppress it).
DEFAULT_SHOW = ("mask", "skeleton", "box", "trail", "label")
NO_HUD = "nohud"

MASK_ALPHA = 0.4            # 0 = invisible, 1 = solid; 0.4 keeps the hand readable
MASK_OUTLINE = True         # a 1px hard edge on the blob, so its border is legible

BONE_THICKNESS = 2
JOINT_RADIUS = 3
TIP_RADIUS = 5              # fingertips are what gestures are made of -- make them big
WRIST_RADIUS = 5
TIP_COLOR = (60, 255, 255)  # BGR yellow, fixed so a fingertip reads as a fingertip
                            # regardless of which colour the track happens to own

TRAIL_MAX_THICKNESS = 3
TRAIL_MIN_FADE = 0.20       # oldest trail point keeps 20% brightness, not 0%: a fully
                            # black tail vanishes against a dark background
TRAIL_STEPS = 6             # fade bands. A 64-point trail drawn segment-by-segment is
                            # 63 antialiased cv2.line calls (0.19 ms per hand); six
                            # polylines look the same and cost 0.05 ms.

BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.55
TEXT_THICKNESS = 1
OUTLINE_EXTRA = 2           # black pass is TEXT_THICKNESS + this
OUTLINE_COLOR = (0, 0, 0)
HUD_COLOR = (255, 255, 255)
HUD_ORIGIN = (10, 26)
HUD_LINE_HEIGHT = 22

MOVEMENT_WINDOW = 5         # frames back the "moved up 12 px" readout looks
MOVEMENT_DEADZONE = 2.0     # px; below this a hand is "still", not jittering

# cv2 drawing takes C ints and a NaN cannot be cast to one at all, so coordinates are
# clamped into a range that is far off-screen but nowhere near overflow.
_COORD_LIMIT = 100000

_COMPASS = ("right", "up-right", "up", "up-left",
            "left", "down-left", "down", "down-right")


# ---------------------------------------------------------------- colours

_COLOR_CACHE = {}
_COLOR_CACHE_MAX = 512

_PLATE_CACHE = {}
_PLATE_CACHE_MAX = 16


def _stable_hash(text):
    """FNV-1a. Python's built-in hash() is salted per process, so a colour derived
    from it would differ between the runner and any tool that re-renders the same
    log. This one does not."""
    value = 0x811C9DC5
    for byte in str(text).encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def color_for_track(track_id):
    """A stable, distinct BGR colour for a track id.

    Deterministic on purpose: hand #1 must be the same colour in frame 900 as in
    frame 1, or the overlay stops meaning anything the moment two hands cross.

    The hue is stepped by the golden-ratio conjugate rather than by a fixed
    increment, which is the standard trick for "give me N maximally separated
    colours without knowing N in advance": ids 0,1,2,3 land at 0, 222, 84 and 306
    degrees instead of four neighbouring greens.
    """
    cached = _COLOR_CACHE.get(track_id)
    if cached is not None:
        return cached

    if isinstance(track_id, (int, np.integer)):
        key = int(track_id)
    else:
        key = _stable_hash(track_id)

    # Integer Fibonacci hashing, not (key * 0.618) % 1.0. The float form silently
    # collapses for large keys: float64 has 53 bits of mantissa, so once key is around
    # 2**63 (which is exactly what a string hash looks like) the product has no
    # fractional part left at all, every id lands on hue 0, and every hand is red.
    # Folding the high 32 bits in first is a no-op for any id below 2**32 -- so the
    # golden-ratio spacing of ids 0, 1, 2, 3 is untouched -- and it stops a key that
    # happens to be a multiple of 2**32 from multiplying out to a flat zero.
    key = (key ^ (key >> 32)) & 0xFFFFFFFF
    hue = ((key * 2654435769) & 0xFFFFFFFF) / 4294967296.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    color = (int(blue * 255), int(green * 255), int(red * 255))

    if len(_COLOR_CACHE) >= _COLOR_CACHE_MAX:
        _COLOR_CACHE.clear()            # pure function, so dropping it costs nothing
    _COLOR_CACHE[track_id] = color
    return color


def _scale_color(color, factor):
    return tuple(int(max(0, min(255, channel * factor))) for channel in color)


def _solid(color, shape):
    """A view onto a cached solid-colour plate of the requested shape.

    np.full() of an ROI is only ~100 us, but it is 100 us *per hand per frame* spent
    writing the same three bytes over and over. One oversized plate per colour, sliced
    to size, costs nothing after the first frame.
    """
    height, width = shape[:2]
    channels = shape[2] if len(shape) > 2 else 1
    color = tuple(color)                # a list colour would be an unhashable key
    key = (color, channels)
    plate = _PLATE_CACHE.get(key)

    if plate is None or plate.shape[0] < height or plate.shape[1] < width:
        grow_h = max(height, 720 if plate is None else plate.shape[0])
        grow_w = max(width, 1280 if plate is None else plate.shape[1])
        dims = (grow_h, grow_w, channels) if channels > 1 else (grow_h, grow_w)
        plate = np.empty(dims, np.uint8)
        plate[:] = color[:channels] if channels > 1 else color[0]
        if len(_PLATE_CACHE) >= _PLATE_CACHE_MAX:
            _PLATE_CACHE.clear()        # long sessions mint new ids forever
        _PLATE_CACHE[key] = plate

    # A writable view into the shared cache is a landmine: one caller writing through
    # it permanently changes that colour for every subsequent hand in the session.
    # The view stays zero-copy; it is just no longer a way to corrupt the cache.
    view = plate[:height, :width]
    view.flags.writeable = False
    return view


# ---------------------------------------------------------------- geometry helpers

def _ipoint(point):
    """(x, y) -> integer pixel tuple, or None if it is not a drawable number.

    A pose that failed to converge can hand back NaN, and int(nan) raises. Points far
    outside the frame are fine -- cv2 clips lines and circles itself -- they just have
    to stay inside C int range, hence the clamp.
    """
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, IndexError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (int(max(-_COORD_LIMIT, min(_COORD_LIMIT, x))),
            int(max(-_COORD_LIMIT, min(_COORD_LIMIT, y))))


def _box_corners(box):
    """Normalise a detection box to two integer corners, low corner first.

    Detectors do produce inverted boxes when a hand runs off the edge, and an
    inverted rectangle draws as nothing at all, so the corners are sorted rather
    than trusted.
    """
    if box is None or len(box) < 4:
        return None
    first = _ipoint((box[0], box[1]))
    second = _ipoint((box[2], box[3]))
    if first is None or second is None:
        return None
    return ((min(first[0], second[0]), min(first[1], second[1])),
            (max(first[0], second[0]), max(first[1], second[1])))


# ---------------------------------------------------------------- text

def draw_text(frame, text, org, color=HUD_COLOR, scale=TEXT_SCALE,
              thickness=TEXT_THICKNESS):
    """Text with a black outline, so it survives a white wall and a black shirt.

    Thick black pass first, thin coloured pass on top. Two putText calls is far
    cheaper than the alternative (a translucent panel behind every label) and does not
    hide the hand.
    """
    if not text:
        return frame
    # The outline is deliberately NOT antialiased. A thick antialiased putText is
    # 0.25 ms a line -- with seven lines on screen that alone would eat 1.8 ms of a
    # 3 ms frame budget -- and it is a black halo underneath antialiased glyphs, so
    # its jaggies are invisible. The coloured pass on top keeps LINE_AA.
    cv2.putText(frame, text, org, FONT, scale, OUTLINE_COLOR,
                thickness + OUTLINE_EXTRA, cv2.LINE_8)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------- movement wording

def movement_text(trajectory, window=MOVEMENT_WINDOW):
    """"up 12 px" for a path that recently went up 12 pixels; "still" for jitter.

    The sketch asks for a plain-English movement line. Direction is quantised to eight
    compass words because a live number in degrees is unreadable at 30 fps, and the
    deadzone stops a motionless hand from flickering between "up" and "down" on
    one-pixel detector noise.

    Screen y grows downward; the sign is flipped here so "up" means up on screen.
    """
    if not trajectory:
        return "still"
    points = [p for p in (_ipoint(q) for q in trajectory) if p is not None]
    if len(points) < 2:
        return "still"

    recent = points[-1]
    past = points[-min(len(points), window + 1)]
    dx = recent[0] - past[0]
    dy = recent[1] - past[1]
    distance = float(np.hypot(dx, dy))
    if distance < MOVEMENT_DEADZONE:
        return "still"

    sector = int(round(np.degrees(np.arctan2(-dy, dx)) % 360.0 / 45.0)) % 8
    return f"{_COMPASS[sector]} {distance:.0f} px"


def label_lines(track):
    """The two lines that sit on the hand itself: who it is, and what it is doing."""
    detection = getattr(track, "detection", None)
    handedness = getattr(detection, "handedness", None) or "unknown"
    name = "hand" if handedness == "unknown" else handedness
    score = float(getattr(detection, "score", 0.0) or 0.0)
    first = f"#{getattr(track, 'track_id', '?')} {name} {score * 100:.0f}%"

    bits = []
    gesture = getattr(track, "gesture", None)
    if gesture and gesture != "none":
        bits.append(f"{gesture} {float(getattr(track, 'gesture_confidence', 0.0)) * 100:.0f}%")
    bits.append(f"{float(getattr(track, 'speed', 0.0)):.0f} px/s")
    return [first, "  ".join(bits)]


def hand_summary(track):
    """One HUD row per hand: identity, gesture, where it moved, how fast."""
    detection = getattr(track, "detection", None)
    handedness = getattr(detection, "handedness", None) or "unknown"
    name = "hand" if handedness == "unknown" else handedness
    parts = [f"#{getattr(track, 'track_id', '?')} {name}"]

    gesture = getattr(track, "gesture", None)
    if gesture and gesture != "none":
        parts.append(f"{gesture} {float(getattr(track, 'gesture_confidence', 0.0)) * 100:.0f}%")

    parts.append(movement_text(getattr(track, "trajectory", None)))
    parts.append(f"{float(getattr(track, 'speed', 0.0)):.0f} px/s")
    return "   ".join(parts)


def hud_lines(tracked_hands, fps=None, extra=None):
    """Build the HUD text. Split out from the drawing so it can be asserted on."""
    hands = list(tracked_hands or ())
    head = []
    if fps is not None:
        head.append(f"{float(fps):.1f} fps")
    head.append(f"hands: {len(hands)}")
    lines = ["   ".join(head)]
    lines.extend(hand_summary(track) for track in hands)

    if extra:
        if isinstance(extra, dict):
            lines.extend(f"{key}: {value}" for key, value in extra.items())
        elif isinstance(extra, str):
            lines.append(extra)
        else:
            lines.extend(str(item) for item in extra)
    return lines


# ---------------------------------------------------------------- layers

def draw_mask(frame, hand_mask, color, alpha=MASK_ALPHA, outline=MASK_OUTLINE):
    """Composite an ROI-local HandMask onto the frame, semi-transparently.

    This is the only function in the module that touches ROI coordinates, and the
    whole reason it is fast: everything happens inside the intersection of the mask's
    ROI and the frame, so cost scales with the hand, not with the frame.

    The ROI is intersected rather than assumed valid because a mask whose origin is
    negative, or which hangs off the right edge, is completely normal for a hand
    walking out of shot -- and slicing a numpy array with a negative index silently
    wraps to the far side of the image, painting the blob on the wrong edge.
    """
    if hand_mask is None:
        return frame
    mask = getattr(hand_mask, "mask", None)
    if mask is None or mask.size == 0 or mask.ndim < 2:
        return frame

    origin = getattr(hand_mask, "origin", (0, 0)) or (0, 0)
    corner = _ipoint(origin)
    if corner is None:
        return frame
    ox, oy = corner

    frame_h, frame_w = frame.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + mask_w, frame_w), min(oy + mask_h, frame_h)
    if x1 <= x0 or y1 <= y0:
        return frame                    # entirely off-screen, nothing to blend

    sub = mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    if sub.ndim > 2:
        # The contract says (h, w), but get_segmentation_mask() in this very repo
        # hands back (h, w, 3), so a segmenter built on it will eventually pass one
        # through. Taking a channel is obviously right and beats an opaque
        # "mask must be CV_8UC1" from deep inside copyTo.
        sub = sub[:, :, 0]
    if sub.dtype != np.uint8:
        sub = sub.astype(np.uint8)      # tolerate a bool mask rather than crash

    roi = frame[y0:y1, x0:x1]
    plate = _solid(color, roi.shape)
    blended = cv2.addWeighted(roi, 1.0 - alpha, plate, alpha, 0.0)
    cv2.copyTo(blended, sub, roi)       # writes through the view, only where mask != 0

    if outline:
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(roi, contours, -1, color, 1, cv2.LINE_8)
    return frame


def draw_skeleton(frame, pose, color):
    """21 joints and the bones between them, in full-frame pixels.

    Bones are drawn dimmer than the joints so the dots stay readable on top of the
    lines, and fingertips get their own colour and radius: gestures are read off the
    tips, so they have to be pickable out of twenty-one identical dots at a glance.
    """
    if pose is None:
        return frame
    points = getattr(pose, "points", None)
    if points is None:
        return frame
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        return frame

    pixels = [_ipoint(point) for point in array]
    count = len(pixels)
    bone_color = _scale_color(color, 0.75)

    for start, end in CONNECTIONS:
        if start < count and end < count:
            a, b = pixels[start], pixels[end]
            if a is not None and b is not None:
                cv2.line(frame, a, b, bone_color, BONE_THICKNESS, cv2.LINE_AA)

    for index, pixel in enumerate(pixels):
        if pixel is None:
            continue
        if index in FINGERTIPS:
            cv2.circle(frame, pixel, TIP_RADIUS, TIP_COLOR, -1, cv2.LINE_AA)
        elif index == WRIST:
            cv2.circle(frame, pixel, WRIST_RADIUS, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pixel, JOINT_RADIUS, color, -1, cv2.LINE_AA)
    return frame


def draw_trail(frame, trajectory, color):
    """The path the hand took, fading and thinning into the past.

    A trail drawn at constant weight reads as a scribble with no direction. Fading it
    encodes time in brightness, so the bright thick end is unambiguously *now* and
    the shape of a swipe is readable from one still frame.
    """
    if not trajectory:
        return frame
    points = [p for p in (_ipoint(q) for q in trajectory) if p is not None]
    if not points:
        return frame

    total = len(points)
    if total == 1:
        cv2.circle(frame, points[0], 4, color, -1, cv2.LINE_AA)
        return frame

    # Banded rather than per-segment: the eye cannot resolve 64 brightness steps in a
    # 100 px tail anyway, and this turns 63 draw calls into 6. Bands share an endpoint
    # so there is no gap where one hands over to the next.
    path = np.asarray(points, dtype=np.int32)
    bands = min(TRAIL_STEPS, total - 1)
    edges = np.linspace(0, total - 1, bands + 1).round().astype(int)
    for band in range(bands):
        low, high = int(edges[band]), int(edges[band + 1])
        if high <= low:
            continue
        age = high / (total - 1)        # 0 = oldest band, 1 = newest
        fade = TRAIL_MIN_FADE + (1.0 - TRAIL_MIN_FADE) * age
        thickness = 1 + int(round(age * (TRAIL_MAX_THICKNESS - 1)))
        cv2.polylines(frame, [path[low:high + 1]], False,
                      _scale_color(color, fade), thickness, cv2.LINE_AA)

    cv2.circle(frame, points[-1], 4, color, -1, cv2.LINE_AA)
    return frame


def draw_box(frame, detection, color):
    """The detector's rectangle. Thin, because the mask already says where the hand is."""
    corners = _box_corners(getattr(detection, "box", None))
    if corners is None:
        return frame
    cv2.rectangle(frame, corners[0], corners[1], color, BOX_THICKNESS)
    return frame


def _label_anchor(track, frame_shape):
    """Where the per-hand label goes: above the box, or inside it near the top edge.

    Clamped into the frame because a label for a hand at y=5 would otherwise be drawn
    at a negative y and simply never appear -- the one hand you most want labelled.
    """
    frame_h, frame_w = frame_shape[:2]
    corners = _box_corners(getattr(getattr(track, "detection", None), "box", None))
    if corners is not None:
        x, y = corners[0][0], corners[0][1]
    else:
        pose = getattr(track, "pose", None)
        wrist = None
        if pose is not None and getattr(pose, "points", None) is not None:
            array = np.asarray(pose.points, dtype=np.float64)
            if array.ndim == 2 and array.shape[0] > WRIST and array.shape[1] >= 2:
                wrist = _ipoint(array[WRIST])
        if wrist is None:
            return None
        x, y = wrist[0], wrist[1]

    x = int(max(2, min(x, frame_w - 4)))
    y = int(max(HUD_LINE_HEIGHT, min(y, frame_h - 4)))
    return (x, y)


def draw_label(frame, track, color):
    """Track id, handedness, gesture, confidence and speed, pinned to the hand."""
    anchor = _label_anchor(track, frame.shape)
    if anchor is None:
        return frame
    x, y = anchor
    lines = label_lines(track)
    top = y - HUD_LINE_HEIGHT * len(lines) - 4
    if top < 4:                         # not enough room above: sit inside the box
        top = y + 4
    for offset, text in enumerate(lines):
        draw_text(frame, text, (x, top + HUD_LINE_HEIGHT * (offset + 1)), color)
    return frame


# ---------------------------------------------------------------- public API

def draw_hand(frame, tracked_hand, show=DEFAULT_SHOW):
    """Draw one tracked hand onto the frame, in place, and return the frame.

    Layers are painted bottom-up -- mask, trail, box, skeleton, label -- so the
    translucent blob never lands on top of the skeleton it is supposed to sit behind,
    and the text ends up above everything else where it can still be read.

    `show` is any container of layer names; anything absent is skipped, which is how
    the runner offers per-key toggles without this module knowing about keys.
    """
    if frame is None or getattr(frame, "size", 0) == 0 or tracked_hand is None:
        return frame

    color = color_for_track(getattr(tracked_hand, "track_id", 0))

    if "mask" in show:
        draw_mask(frame, getattr(tracked_hand, "mask", None), color)
    if "trail" in show:
        draw_trail(frame, getattr(tracked_hand, "trajectory", None), color)
    if "box" in show:
        draw_box(frame, getattr(tracked_hand, "detection", None), color)
    if "skeleton" in show:
        draw_skeleton(frame, getattr(tracked_hand, "pose", None), color)
    if "label" in show:
        draw_label(frame, tracked_hand, color)
    return frame


def draw_hud(frame, tracked_hands, fps=None, extra=None):
    """The top-left readout: fps, hand count, and a movement line per hand.

    Hand rows are tinted with that hand's own colour, so the row and the hand on
    screen identify each other without a legend.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    hands = list(tracked_hands or ())
    lines = hud_lines(hands, fps, extra)
    x, y = HUD_ORIGIN
    for index, text in enumerate(lines):
        # line 0 is the fps/count header; lines 1..n line up with the hands
        color = HUD_COLOR
        if 1 <= index <= len(hands):
            color = color_for_track(getattr(hands[index - 1], "track_id", 0))
        draw_text(frame, text, (x, y + HUD_LINE_HEIGHT * index), color)
    return frame


def render(frame, tracked_hands, fps=None, show=DEFAULT_SHOW, extra=None):
    """Everything: every hand, then the HUD on top. In place; returns the frame.

    `show` whitelists the PER-HAND layers -- show=("mask",) is a mask-only debug view.
    The HUD is not one of them: fps and hand count are frame-level facts, and a caller
    that switches off the skeleton has not asked to lose its frame rate readout. So
    the HUD is drawn whenever anything is drawn at all, and is switched off explicitly
    with NO_HUD in `show`, or implicitly by show=() meaning "draw nothing".

    That default also keeps this honest for callers that enumerate the layer names
    themselves (run_pipeline.py builds --show from its own list of the five hand
    layers); requiring an opt-in token would have silently dropped their HUD.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame
    # Materialise once. Iterating a generator here would exhaust it, and draw_hud
    # would then report "hands: 0" and lose every per-hand row.
    tracks = list(tracked_hands or ())
    for track in tracks:
        draw_hand(frame, track, show=show)
    if show and NO_HUD not in show:
        draw_hud(frame, tracks, fps, extra)
    return frame
