"""
Score office_monitor.py's timeline against MM-Office's labelled test events.

    python evaluate_office.py videos --split tuning > tuning.txt   # which videos to run
    python office_monitor.py --video tuning.txt --out office_runs/tuning
    python evaluate_office.py score office_runs/tuning/events.csv --split tuning

WHAT IS SCORED. MM-Office labels each test recording with events and their start/end
second. Four of its 13 classes follow from what office_monitor already outputs:

    sit down   a person's posture changes from standing/walking to sitting
    stand up   ... from sitting to standing/walking
    enter      a person id appears (not counting those already in view at the start)
    exit       a person id is lost for good

enter/exit are a baseline: a tracker that loses someone behind a partition reports an
exit and an enter that never happened.

MATCHING. A detection is correct if it falls inside a labelled event of the same class,
widened by --collar seconds either side (default 2). Each labelled event can be matched
once; extra detections are false positives. Precision, recall and F1 per class.

CAMERAS. Each recording was filmed by four cameras with ONE set of labels. Scored two
ways: per camera, and fused -- detections from all four pooled, and same-class
detections within --collar of each other merged, so four cameras seeing one sit-down
count once.

THE SPLIT. 22 recordings are for tuning rules, 66 are held out for the final number.
Adjusting thresholds while looking at the final split would inflate the result, so the
split is fixed by a seeded shuffle and never changes.
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

LABELS = Path("../mm-office/MM_Office_Dataset/label/testlabel")
VIDEOS = Path("../mm-office/MM_Office_Dataset/video/test")
CLASSES = {8: "enter", 9: "exit", 10: "stand up", 11: "sit down"}
SEATED = "sitting"
UPRIGHT = ("standing", "walking")
SOURCE = re.compile(r"(?P<split>split\d+)_(?P<camera>id\d+)_s(?P<scene>\d+)_recid(?P<recid>\d+)")
CAMERAS = ("id00", "id01", "id02", "id03")
TUNING_COUNT = 22
SEED = 0
# Recordings already watched while writing the posture rules. They go in tuning by
# construction: rules shaped on them must not be scored on them as "unseen".
SEEN_DURING_DEVELOPMENT = (8, 618)


def recordings(label_dir=LABELS):
    return sorted(int(p.stem.removeprefix("recid")) for p in Path(label_dir).glob("recid*.csv"))


def split(name, label_dir=LABELS):
    recids = recordings(label_dir)
    seen = [r for r in SEEN_DURING_DEVELOPMENT if r in recids]
    rest = [r for r in recids if r not in seen]
    random.Random(SEED).shuffle(rest)
    tuning = seen + rest[:TUNING_COUNT - len(seen)]
    final = rest[TUNING_COUNT - len(seen):]
    return sorted({"tuning": tuning, "final": final, "all": recids}[name])


def ground_truth(recid, label_dir=LABELS):
    """[(class name, start, end), ...] for the classes we score."""
    out = []
    with open(Path(label_dir) / f"recid{recid:03d}.csv") as f:
        for row in csv.DictReader(f):
            cls = int(row["eventclass"])
            if cls in CLASSES:
                out.append((CLASSES[cls], float(row["starttime"]), float(row["endtime"])))
    return out


def detections(rows, max_gap=5.0, settle=1.0):
    """events.csv rows for ONE video -> [(class name, time), ...]

    A posture change is timed at the midpoint between the old segment's end and the new
    one's start. Unknown stretches are not logged, so the two need not touch; if they
    are more than max_gap apart we do not know when the change happened and skip it.
    `appeared` within `settle` seconds of the start is someone already in the room.
    """
    out = []
    by_person = defaultdict(list)
    for r in rows:
        t = float(r["start"])
        if r["kind"] == "appeared" and t > settle:
            out.append(("enter", t))
        elif r["kind"] == "left":
            out.append(("exit", t))
        elif r["kind"] == "posture":
            by_person[r["person"]].append((t, float(r["end"]), r["label"]))
    for segments in by_person.values():
        segments.sort()
        for (_, prev_end, prev), (start, _, cur) in zip(segments, segments[1:]):
            if start - prev_end > max_gap:
                continue
            when = (prev_end + start) / 2
            if prev in UPRIGHT and cur == SEATED:
                out.append(("sit down", when))
            elif prev == SEATED and cur in UPRIGHT:
                out.append(("stand up", when))
    return out


def merge(dets, window):
    """Same-class detections closer than `window` seconds become one (the earliest)."""
    out = []
    for cls in {c for c, _ in dets}:
        last = None
        for t in sorted(t for c, t in dets if c == cls):
            if last is None or t - last > window:
                out.append((cls, t))
            last = t
    return out


def match(dets, truth, collar):
    """-> {class: [tp, fp, fn]}. Each truth event matches at most one detection, and a
    detection takes the nearest unmatched truth event whose widened window holds it."""
    counts = defaultdict(lambda: [0, 0, 0])
    for cls in set(CLASSES.values()):
        events = [(s, e) for c, s, e in truth if c == cls]
        used = [False] * len(events)
        for t in sorted(t for c, t in dets if c == cls):
            hits = [i for i, (s, e) in enumerate(events)
                    if not used[i] and s - collar <= t <= e + collar]
            if hits:
                best = min(hits, key=lambda i: abs(t - (events[i][0] + events[i][1]) / 2))
                used[best] = True
                counts[cls][0] += 1
            else:
                counts[cls][1] += 1
        counts[cls][2] += used.count(False)
    return counts


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def table(title, counts):
    print(f"\n{title}")
    print(f"  {'event':<10} {'labelled':>8} {'found':>6} {'false':>6}   "
          f"{'precision':>9} {'recall':>7} {'F1':>6}")
    total = [0, 0, 0]
    for cls in ("sit down", "stand up", "enter", "exit"):
        tp, fp, fn = counts.get(cls, [0, 0, 0])
        total = [a + b for a, b in zip(total, (tp, fp, fn))]
        p, r, f = prf(tp, fp, fn)
        print(f"  {cls:<10} {tp + fn:>8} {tp:>6} {fp:>6}   {p:>9.3f} {r:>7.3f} {f:>6.3f}")
    p, r, f = prf(*total)
    print(f"  {'all four':<10} {total[0] + total[2]:>8} {total[0]:>6} {total[1]:>6}   "
          f"{p:>9.3f} {r:>7.3f} {f:>6.3f}")


def add(into, counts):
    for cls, (tp, fp, fn) in counts.items():
        into[cls] = [a + b for a, b in zip(into.get(cls, [0, 0, 0]), (tp, fp, fn))]


def score(args):
    wanted = set(split(args.split, args.labels))
    rows = defaultdict(list)          # (recid, camera) -> rows
    with open(args.events) as f:
        for row in csv.DictReader(f):
            m = SOURCE.match(row["source"])
            if m and int(m["recid"]) in wanted:
                rows[(int(m["recid"]), m["camera"])].append(row)

    seen = sorted({recid for recid, _ in rows})
    if not seen:
        raise SystemExit(f"no {args.split} recordings in {args.events}")
    missing = len(wanted) - len(seen)

    per_camera = {cam: {} for cam in CAMERAS}
    fused = {}
    for recid in seen:
        truth = ground_truth(recid, args.labels)
        pooled = []
        for cam in CAMERAS:
            if (recid, cam) not in rows:
                continue
            dets = detections(rows[(recid, cam)])
            pooled += dets
            add(per_camera[cam], match(merge(dets, args.collar), truth, args.collar))
        add(fused, match(merge(pooled, args.collar), truth, args.collar))

    print(f"{args.split} split: {len(seen)} recordings scored"
          + (f" ({missing} not in the events file)" if missing else "")
          + f", collar +-{args.collar:g} s")
    table("FUSED (any of the four cameras)", fused)
    for cam in CAMERAS:
        if per_camera[cam]:
            table(f"camera {cam}", per_camera[cam])


def videos(args):
    """Print the test video paths for a split, one per line."""
    chosen = set(split(args.split, args.labels))
    listing = Path(args.listing).read_text(encoding="utf-8", errors="replace").split()
    for name in sorted(n for n in listing if n.endswith(".mp4") and "/video/test/" in n):
        m = SOURCE.search(name)
        if m and int(m["recid"]) in chosen:
            print(Path(args.videos) / Path(name).name)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(LABELS))
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("videos", help="list the videos in a split")
    v.add_argument("--split", default="tuning", choices=["tuning", "final", "all"])
    v.add_argument("--listing", default="../mm-office/_list.txt",
                   help="output of `fetch_mm_office.py list`")
    v.add_argument("--videos", default=str(VIDEOS))

    s = sub.add_parser("score", help="score an events.csv")
    s.add_argument("events")
    s.add_argument("--split", default="tuning", choices=["tuning", "final", "all"])
    s.add_argument("--collar", type=float, default=2.0)

    args = parser.parse_args()
    videos(args) if args.cmd == "videos" else score(args)


if __name__ == "__main__":
    main()
