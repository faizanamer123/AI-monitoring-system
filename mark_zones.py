"""
Mark a camera's doors once, by dragging rectangles over a frame.

    python mark_zones.py config/zones.yaml office_cam --camera 0
    python mark_zones.py config/zones.yaml G421 --video clip.avi

Drag a box around each door, including the floor in front of it, and press Enter after
each one. Press Esc when done. NAME is matched against the video or camera name.
"""

import argparse
from pathlib import Path

import cv2
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zones")
    parser.add_argument("name")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read a frame")
    height, width = frame.shape[:2]
    boxes = cv2.selectROIs("mark doors: drag, Enter after each, Esc to finish", frame)
    cv2.destroyAllWindows()

    path = Path(args.zones)
    zones = (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}
    zones[args.name] = {"doors": [[round(float(x) / width, 3), round(float(y) / height, 3),
                                   round(float(x + w) / width, 3), round(float(y + h) / height, 3)]
                                  for x, y, w, h in boxes]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(zones, sort_keys=False))
    print(f"wrote {len(boxes)} door(s) for {args.name} to {path}")


if __name__ == "__main__":
    main()
