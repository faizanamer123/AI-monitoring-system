"""
Draw a scored run on its video, so results can be checked by eye.

    python render_results.py VIDEO TRACKS.npz EVENTS.csv LABEL_DIR OUT.mp4 --frame-size 320x240

Replays the saved tracks through the same rules the run used, and adds a banner comparing
the event MM-Office labels at that moment with the event the system reported.
"""

import argparse
import csv
import re
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import evaluate_meva
import evaluate_office
from office.cache import cached_frame_size, read_tracks
from office_monitor import Rules, draw, frame_size

BANNER = 62
SHOW_EVENT_S = 1.5
SCALE_FONT = cv2.FONT_HERSHEY_SIMPLEX


class NoStore:
    def write(self, events):
        pass


def reported_events(events_csv, source):
    """[(label, start, end)] of the latest session's moments and activities for one video."""
    with open(events_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == source]
    latest = max(r["session"] for r in rows) if rows else None
    return [(r["label"], float(r["start"]), float(r["end"])) for r in rows
            if r["session"] == latest and r["kind"] in ("change", "activity")]


def banner(width, t, truth, reported):
    strip = np.zeros((BANNER, width, 3), np.uint8)
    labelled = sorted(c for c, s, e in truth if s <= t <= e)
    said = sorted({c for c, s, e in reported if s - SHOW_EVENT_S <= t <= e + SHOW_EVENT_S})
    lines =((f"LABEL:  {', '.join(labelled) or '-'}", (120, 220, 255), 24),
             (f"SYSTEM: {', '.join(said) or '-'}", (120, 255, 120), 52))
    for text, color, y in lines:
        cv2.putText(strip, text, (10, y), SCALE_FONT, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(strip, f"{t:5.1f} s", (width - 110, 24), SCALE_FONT, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return strip


def enlarge(person, scale):
    keypoints = person.keypoints.copy()
    keypoints[:, :2] *= scale
    return replace(person, box=tuple(int(v * scale) for v in person.box), keypoints=keypoints)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("tracks")
    parser.add_argument("events")
    parser.add_argument("labels")
    parser.add_argument("out")
    parser.add_argument("--frame-size", type=frame_size, default=None)
    parser.add_argument("--scale", type=float, default=3.0, help="resize factor for the output")
    parser.add_argument("--confirm-s", type=float, default=3.0)
    parser.add_argument("--stitch-gap", type=float, default=5.0)
    parser.add_argument("--from-s", type=float, default=0.0, help="start of the part to write")
    parser.add_argument("--to-s", type=float, default=float("inf"))
    parser.add_argument("--meva", action="store_true", help="labels are a MEVA .activities.yml")
    args = parser.parse_args()

    source = Path(args.video).stem
    if args.meva:
        truth = evaluate_meva.read_labels(args.labels)
    else:
        recid = int(re.search(r"recid(\d+)", source)[1])
        truth = evaluate_office.ground_truth(recid, args.labels)
    reported = reported_events(args.events, source)
    size = cached_frame_size(args.tracks) or args.frame_size
    rules = Rules(source, NoStore(), args.confirm_s, args.stitch_gap, size)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame, position, writer = None, -1, None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for t, raw, things in read_tracks(args.tracks):
        while position < round(t * fps):
            ok, frame = cap.read()
            position += 1
            if not ok:
                break
        if frame is None:
            break
        people, labels, activities = rules.step(t, raw, things)
        if not args.from_s <= t <= args.to_s:      # rules still run, so ids and states are right
            continue
        big = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
        shown = [replace(th, box=tuple(int(v * args.scale) for v in th.box)) for th in things]
        big = draw(big, [enlarge(p, args.scale) for p in people], labels, activities, shown, False)
        out = np.vstack([big, banner(big.shape[1], t, truth, reported)])
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / 2,
                                     (out.shape[1], out.shape[0]))
        writer.write(out)
    cap.release()
    if writer is not None:
        writer.release()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
