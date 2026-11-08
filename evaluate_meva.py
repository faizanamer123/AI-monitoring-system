"""
Score office_monitor.py on MEVA: public 1080p footage from high, fixed cameras.

    python evaluate_meva.py EVENTS.csv LABEL_DIR --report mistakes.csv

LABEL_DIR holds MEVA's <clip>.activities.yml files. Moments (sit down, stand up, enter,
exit) are matched as in evaluate_office.py: same event type, inside the labelled time
widened by --collar seconds. Phone activities last for seconds, so a detected segment
counts if it overlaps a labelled one.

Labels here belong to one camera's own clip, so nothing is marked missed for being out
of view. They are matched by event type and time only, not by person, so in a crowded
room a detection on one person can be credited to a label on another.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from evaluate_office import add, intervals, match, merge, prf, table

FPS = 30.0
LABELLED = {"person_sits_down": "sit down", "person_stands_up": "stand up",
            "person_enters_scene_through_structure": "enter",
            "person_exits_scene_through_structure": "exit",
            "person_talks_on_phone": "on_call", "person_texts_on_phone": "using_phone"}
MOMENTS = ("sit down", "stand up", "enter", "exit")
PHONE = ("on_call", "using_phone")
GROUPS = (("sit down + stand up", ("sit down", "stand up")), ("enter + exit", ("enter", "exit")),
          ("phone (call + texting)", PHONE), ("sit down", ("sit down",)), ("stand up", ("stand up",)),
          ("enter", ("enter",)), ("exit", ("exit",)), ("on_call", ("on_call",)),
          ("using_phone", ("using_phone",)))
ACTIVITY = re.compile(r"act2'?:\s*\{\s*'?([a-z_]+)'?\s*:.*?tsr0'?:\s*\[\s*(\d+)\s*,\s*(\d+)")


def read_labels(path):
    """[(event, start seconds, end seconds), ...] for the activities we score."""
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        found = ACTIVITY.search(line)
        if found and found[1] in LABELLED:
            out.append((LABELLED[found[1]], int(found[2]) / FPS, int(found[3]) / FPS))
    return out


def match_segments(dets, truth, collar, outcomes=None):
    """(event, start, end) detections against labels: a detection counts if it overlaps an
    unused label of its type, widened by the collar. -> {event: [tp, fp, fn]}"""
    counts = defaultdict(lambda: [0, 0, 0])
    for cls in PHONE:
        events = [(s, e) for c, s, e in truth if c == cls]
        used = [False] * len(events)
        for _, a, b in sorted(d for d in dets if d[0] == cls):
            overlap = [(min(b, e + collar) - max(a, s - collar), i)
                       for i, (s, e) in enumerate(events) if not used[i]
                       and min(b, e + collar) > max(a, s - collar)]
            if overlap:
                i = max(overlap)[1]
                used[i] = True
                counts[cls][0] += 1
                if outcomes is not None:
                    outcomes.append((cls, "found", events[i][0], events[i][1], a))
            else:
                counts[cls][1] += 1
                if outcomes is not None:
                    outcomes.append((cls, "false alarm", "", "", a))
        counts[cls][2] += used.count(False)
        if outcomes is not None:
            outcomes += [(cls, "missed", s, e, "") for (s, e), hit in zip(events, used) if not hit]
    return counts


def latest_rows(events_csv):
    with open(events_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    latest = max(r["session"] for r in rows) if rows else None
    return [r for r in rows if r["session"] == latest]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events")
    parser.add_argument("labels")
    parser.add_argument("--collar", type=float, default=2.0)
    parser.add_argument("--report", help="write every found / missed / false event to this CSV")
    args = parser.parse_args()

    by_source = defaultdict(list)
    for row in latest_rows(args.events):
        by_source[row["source"]].append(row)

    pooled, by_clip, by_camera, report = {}, {}, defaultdict(dict), []
    classes = MOMENTS + PHONE
    for path in sorted(Path(args.labels).glob("*.activities.yml")):
        clip = path.name.removesuffix(".activities.yml")
        rows = by_source.get(f"{clip}.r13", [])
        truth = read_labels(path)
        camera = clip.rsplit(".", 1)[1]
        points = [(r["label"], float(r["start"])) for r in rows if r["kind"] == "change"]
        spans = [(r["label"], float(r["start"]), float(r["end"])) for r in rows
                 if r["kind"] == "activity" and r["label"] in PHONE]
        outcomes = []
        counts = match(merge(points, args.collar), truth, args.collar, outcomes, MOMENTS)
        add(counts, match_segments(spans, truth, args.collar, outcomes))
        by_clip[clip] = dict(counts)
        add(pooled, counts)
        add(by_camera[camera], counts)
        report += [(clip, camera, cls, result, "" if s == "" else round(s, 1),
                    "" if e == "" else round(e, 1), "" if when == "" else round(when, 1))
                   for cls, result, s, e, when in outcomes]

    print(f"{len(by_clip)} MEVA clips scored, collar +-{args.collar:g} s, by event type and time")
    table("ALL CLIPS", pooled, classes)
    print("\n95% ranges, resampling clips:")
    for title, members in GROUPS:
        tp, fp, fn = (sum(pooled.get(c, [0, 0, 0])[i] for c in members) for i in range(3))
        p, r, f = prf(tp, fp, fn)
        (pl, ph), (rl, rh), (fl, fh) = intervals(by_clip, members)
        print(f"  {title:<26} labelled {tp + fn:>4}  found {tp:>4}  false {fp:>5}  "
              f"precision {p:.3f} ({pl:.3f}-{ph:.3f})  recall {r:.3f} ({rl:.3f}-{rh:.3f})  "
              f"F1 {f:.3f} ({fl:.3f}-{fh:.3f})")
    for camera in sorted(by_camera):
        table(f"camera {camera}", by_camera[camera], classes)

    if args.report:
        with open(args.report, "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(["clip", "camera", "event", "result", "label_start_s", "label_end_s",
                          "detected_at_s"])
            out.writerows(sorted(report, key=lambda r: (r[0], r[3], r[2])))


if __name__ == "__main__":
    main()
