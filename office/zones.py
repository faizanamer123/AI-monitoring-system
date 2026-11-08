"""
Places marked once per camera: currently doors, for enter and exit.

The frame edge is a poor stand-in for a door. Someone coming through a door in the middle
of the view appears mid-frame and is ignored, while anyone walking past a corner counts.
A camera does not move, so its doors can be marked once.

    # zones.yaml -- coordinates are fractions of the frame, so resolution does not matter
    G421:
      doors:
        - [0.47, 0.10, 0.60, 0.40]     # x1, y1, x2, y2

Zones should include the floor in front of the door, because a person is judged by where
their feet are. Cameras are matched by name: a key appearing in the video or camera name.
"""

import yaml


def load_zones(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def doors_for(zones, name, size):
    """Door rectangles in pixels for this camera, or [] if none are marked."""
    width, height = size
    for key, camera in (zones or {}).items():
        if str(key) in name:
            return [(x1 * width, y1 * height, x2 * width, y2 * height)
                    for x1, y1, x2, y2 in camera.get("doors", [])]
    return []


def at_door(box, doors):
    """Are this person's feet, or failing that the middle of their box, in a door zone?
    Feet are the better signal, but they are often cut off in a doorway."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    points = ((cx, y2), (cx, (y1 + y2) / 2))
    return any(a <= x <= c and b <= y <= d for a, b, c, d in doors for x, y in points)
