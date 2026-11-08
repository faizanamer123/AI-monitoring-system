"""
Places marked once per camera: doors for enter and exit, seats for sitting.

The frame edge is a poor stand-in for a door. Someone coming through a door in the middle
of the view appears mid-frame and is ignored, while anyone walking past a corner counts.
A camera does not move, so its doors can be marked once.

Seats do the same job for posture. People only sit where there is something to sit on, so
a "sit down" anywhere else is a misread pose, and from behind a chair, where the legs are
hidden, being in a seat zone is the best evidence available.

    # zones.yaml -- coordinates are fractions of the frame, so resolution does not matter
    G421:
      doors:
        - [0.47, 0.10, 0.60, 0.40]     # x1, y1, x2, y2
      seats:
        - [0.11, 0.27, 0.27, 0.42]

Zones should include the floor in front, because a person is judged by where their feet
are. Cameras are matched by name: a key appearing in the video or camera name.
"""

import yaml


def load_zones(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def doors_for(zones, name, size, kind="doors"):
    """Zone rectangles in pixels for this camera, or [] if none are marked. kind is
    "doors" or "seats"."""
    width, height = size
    for key, camera in (zones or {}).items():
        if str(key) in name:
            return [(x1 * width, y1 * height, x2 * width, y2 * height)
                    for x1, y1, x2, y2 in camera.get(kind, [])]
    return []


def at_door(box, doors):
    """Are this person's feet, or failing that the middle of their box, in a door zone?
    Feet are the better signal, but they are often cut off in a doorway."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    points = ((cx, y2), (cx, (y1 + y2) / 2))
    return any(a <= x <= c and b <= y <= d for a, b, c, d in doors for x, y in points)
