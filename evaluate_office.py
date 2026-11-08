"""
Score office_monitor.py's timeline against MM-Office's labelled test events.

    python evaluate_office.py videos --split tuning > tuning.txt   # which videos to run
    python office_monitor.py --video tuning.txt --out ../results/runs/tuning
    python evaluate_office.py score ../results/runs/tuning/events.csv --split tuning

WHAT IS SCORED. MM-Office labels each test recording with events and their start/end
second. Four of its 13 classes are events office_monitor reports itself:

    sit down   a person's observed posture changes from upright to sitting
    stand up   ... from sitting to upright
    enter      a person id appears at the frame edge
    exit       a person id is lost for good at the frame edge

MATCHING. A detection is correct if it falls inside a labelled event of the same class,
widened by --collar seconds either side (default 2). Each labelled event can be matched
once; extra detections are false positives. Precision, recall and F1 per class.

CAMERAS. Each recording was filmed by four cameras with one set of labels. The product
is one camera per room, so each camera is judged alone, then the four are pooled: a
recording counts four times, once per viewpoint.

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
SOURCE =re.compile(r"(?P<split>split\d+)_(?P<camera>id\d+)_s(?P<scene>\d+)_recid(?P<recid>\d+)")
CAMERAS = ("id00", "id01", "id02", "id03")
GROUPS = (("sit down + stand up", ("sit down", "stand up")),
          ("enter + exit (edge rule)", ("enter", "exit")),
          ("sit down", ("sit down",)), ("stand up", ("stand up",)),
          ("enter", ("enter",)), ("exit", ("exit",)))
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


def detections(rows):
    """events.csv rows for ONE video -> [(class name, time), ...]

    The monitor decides what counts as an event (office/events.py), so what is scored
    here is exactly what the live system reports."""
    return [(r["label"], float(r["start"])) for r in rows if r["kind"] == "change"]


def merge(dets, window, min_cameras=1):
    """Same-class detections closer than `window` seconds become one (the earliest).

    dets may carry a camera as a third field. min_cameras then keeps only groups seen
    by at least that many different cameras: each camera's false alarms are
    independent, so any-camera fusion adds them all up, while agreement cancels them.
    """
    out = []
    for cls in {d[0] for d in dets}:
        group, last = [], None
        for t, camera in sorted((d[1], d[2] if len(d) > 2 else 0) for d in dets if d[0] == cls):
            if last is not None and t - last > window:
                if len({c for _, c in group}) >= min_cameras:
                    out.append((cls, group[0][0]))
                group = []
            group.append((t, camera))
            last = t
        if group and len({c for _, c in group}) >= min_cameras:
            out.append((cls, group[0][0]))
    return out


def match(dets, truth, collar, outcomes=None, classes=tuple(CLASSES.values())):
    """-> {class: [tp, fp, fn]}. Each truth event matches at most one detection, and a
    detection takes the nearest unmatched truth event whose widened window holds it.
    outcomes: optional list, receives (class, result, label start, label end, detected)."""
    counts = defaultdict(lambda: [0, 0, 0])
    log = outcomes.append if outcomes is not None else lambda row: None
    for cls in classes:
        events = [(s, e) for c, s, e in truth if c == cls]
        used = [False] * len(events)
        for t in sorted(t for c, t in dets if c == cls):
            hits = [i for i, (s, e) in enumerate(events)
                    if not used[i] and s - collar <= t <= e + collar]
            if hits:
                best = min(hits, key=lambda i: abs(t - (events[i][0] + events[i][1]) / 2))
                used[best] = True
                counts[cls][0] += 1
                log((cls, "found", events[best][0], events[best][1], t))
            else:
                counts[cls][1] += 1
                log((cls, "false alarm", "", "", t))
        counts[cls][2] += used.count(False)
        for i, (s, e) in enumerate(events):
            if not used[i]:
                log((cls, "missed", s, e, ""))
    return counts


def write_report(path, rows):
    """One line per labelled event and per false alarm, for checking against the video."""
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["recording", "camera", "event", "result", "label_start_s", "label_end_s",
                      "detected_at_s"])
        out.writerows(rows)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def table(title, counts, classes=("sit down", "stand up", "enter", "exit")):
    print(f"\n{title}")
    print(f"  {'event':<11} {'labelled':>8} {'found':>6} {'false':>6}   "
          f"{'precision':>9} {'recall':>7} {'F1':>6}")
    total = [0, 0, 0]
    for cls in classes:
        tp, fp, fn = counts.get(cls, [0, 0, 0])
        total = [a + b for a, b in zip(total, (tp, fp, fn))]
        p, r, f = prf(tp, fp, fn)
        print(f"  {cls:<11} {tp + fn:>8} {tp:>6} {fp:>6}   {p:>9.3f} {r:>7.3f} {f:>6.3f}")
    p, r, f = prf(*total)
    print(f"  {'all':<11} {total[0] + total[2]:>8} {total[0]:>6} {total[1]:>6}   "
          f"{p:>9.3f} {r:>7.3f} {f:>6.3f}")


def add(into, counts):
    for cls, (tp, fp, fn) in counts.items():
        into[cls] = [a + b for a, b in zip(into.get(cls, [0, 0, 0]), (tp, fp, fn))]


def intervals(by_recording, classes, samples=2000, seed=0):
    """95% ranges for precision, recall and F1, resampling whole recordings: events
    within one recording share the same people and room, so they are not independent."""
    recids = sorted(by_recording)
    rng = random.Random(seed)
    results = []
    for _ in range(samples):
        tp = fp = fn = 0
        for recid in rng.choices(recids, k=len(recids)):
            for cls in classes:
                a, b, c = by_recording[recid].get(cls, [0, 0, 0])
                tp, fp, fn = tp + a, fp + b, fn + c
        results.append(prf(tp, fp, fn))
    low, high = int(samples * 0.025), int(samples * 0.975)
    return [(sorted(v)[low], sorted(v)[high]) for v in zip(*results)]


def score(args):
    wanted = set(split(args.split, args.labels))
    rows = defaultdict(list)          # (recid, camera) -> rows
    with open(args.events) as f:
        everything = list(csv.DictReader(f))
    if everything and "session" not in everything[0]:
        raise SystemExit(f"{args.events} predates change events; replay it with this version")
    # a folder accumulates sessions, and scoring two runs at once would double every event
    latest = max(row["session"] for row in everything) if everything else None
    for row in everything:
        if row["session"] == latest:
            m = SOURCE.match(row["source"])
            if m and int(m["recid"]) in wanted:
                rows[(int(m["recid"]), m["camera"])].append(row)

    seen = sorted({recid for recid, _ in rows})
    if not seen:
        raise SystemExit(f"no {args.split} recordings in {args.events}")
    missing = len(wanted) - len(seen)

    per_camera = {cam: {} for cam in CAMERAS}
    pooled, by_recording, report = {}, {}, []
    for recid in seen:
        truth = ground_truth(recid, args.labels)
        by_recording[recid] = {}
        for cam in CAMERAS:
            # A camera with no rows produced no events -- every labelled event is a
            # miss for it. Skipping it instead silently dropped those misses and
            # flattered the weakest cameras.
            outcomes = []
            counts = match(merge(detections(rows.get((recid, cam), [])), args.collar),
                           truth, args.collar, outcomes)
            add(per_camera[cam], counts)
            add(pooled, counts)
            add(by_recording[recid], counts)
            report += [(f"recid{recid:03d}", cam, cls, result, start, end,
                        "" if when == "" else round(when, 1))
                       for cls, result, start, end, when in outcomes]

    if args.report:
        write_report(args.report, sorted(report, key=lambda r: (r[0], r[1], r[3], r[2])))
    print(f"{args.split} split: {len(seen)} recordings scored"
          + (f" ({missing} not in the events file)" if missing else "")
          + f", collar +-{args.collar:g} s, each camera judged alone")
    table("ALL CAMERAS POOLED", pooled)
    print("\n95% ranges, resampling recordings:")
    for title, classes in GROUPS:
        tp, fp, fn = (sum(pooled.get(c, [0, 0, 0])[i] for c in classes) for i in range(3))
        p, r, f = prf(tp, fp, fn)
        (pl, ph), (rl, rh), (fl, fh) = intervals(by_recording, classes)
        print(f"  {title:<26} precision {p:.3f} ({pl:.3f}-{ph:.3f})  "
              f"recall {r:.3f} ({rl:.3f}-{rh:.3f})  F1 {f:.3f} ({fl:.3f}-{fh:.3f})")
    for cam in CAMERAS:
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
    s.add_argument("--report", help="write every found / missed / false event to this CSV")

    args = parser.parse_args()
    videos(args) if args.cmd == "videos" else score(args)


if __name__ == "__main__":
    main()
