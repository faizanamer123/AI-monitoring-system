"""
Office activity monitor: track every person, label their posture, log a timeline.

    python office_monitor.py                                # webcam
    python office_monitor.py --video clip.mp4 --show        # a file, with a live window
    python office_monitor.py --video folder/ --save-video   # every video in a folder
    python office_monitor.py --camera 0 --blur-faces --show

    python office_monitor.py --video list.txt --save-tracks tracks/   # cache model output
    python office_monitor.py --tracks tracks/ --out runs/try2          # rules only, fast

Writes to --out (default office_runs/):
    events.db / events.csv   every event: appeared, posture segments, left
    summary.csv              seconds per person per posture
    <name>.mp4               annotated video, ONLY with --save-video

--save-tracks stores what the pose model and tracker saw, per video. --tracks then
re-runs only the rules from that cache: seconds instead of hours, so a rule can be
tuned and re-scored without the model (see office/cache.py).

No video is kept unless you ask for it: the event log alone answers "who sat, stood or
walked, and for how long". --blur-faces blurs heads in anything shown or saved.

Press q or Esc in the window to stop.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

from office.cache import TrackWriter, read_tracks
from office.events import EventLog, EventStore, summarise
from office.perception import (DEFAULT_WEIGHTS, L_EAR, NOSE, R_EAR, SKELETON,
                               PersonTracker, drop_nested)
from office.posture import SITTING, STANDING, UNKNOWN, WALKING, PostureTracker

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
COLORS = {SITTING: (255, 160, 0), STANDING: (0, 200, 0), WALKING: (0, 140, 255),
          UNKNOWN: (160, 160, 160)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", help="a video file, a folder of them, or a .txt listing "
                                        "one video path per line (default: webcam)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.3, help="person detection threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--every", type=int, default=2, help="process every Nth frame")
    parser.add_argument("--out", default="office_runs")
    parser.add_argument("--save-video", action="store_true", help="also write annotated video")
    parser.add_argument("--show", action="store_true", help="display a live window")
    parser.add_argument("--blur-faces", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="with a folder: first N videos")
    parser.add_argument("--save-tracks", metavar="DIR",
                        help="also cache the tracker's output per video here")
    parser.add_argument("--tracks", metavar="DIR",
                        help="skip the model: replay cached tracks from this folder")
    args = parser.parse_args()
    if args.every < 1:
        parser.error("--every must be at least 1")
    if args.tracks and (args.video or args.save_video or args.show or args.save_tracks):
        parser.error("--tracks replays a cache; it cannot be combined with --video, "
                     "--save-video, --show or --save-tracks")
    return args


def blur_face(frame, person):
    """Blur around whichever head keypoints are visible. Sized from the ear-to-ear or
    box width, since a fixed pixel radius is either too small near the camera or
    swallows the whole body far away."""
    kp = person.keypoints
    head = kp[NOSE:R_EAR + 1]
    seen = head[head[:, 2] > 0.3]
    if len(seen) == 0:
        return
    cx, cy = seen[:, :2].mean(axis=0)
    ears = kp[[L_EAR, R_EAR]]
    span = abs(ears[0, 0] - ears[1, 0]) if (ears[:, 2] > 0.3).all() else 0
    r = int(max(span * 1.2, (person.box[2] - person.box[0]) * 0.3, 6))
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(int(cx) - r, 0), max(int(cy) - r, 0), min(int(cx) + r, w), min(int(cy) + r, h)
    if x2 > x1 and y2 > y1:
        k = max(3, (r // 2) * 2 + 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (k, k), 0)


def draw(frame, people, labels, blur):
    for person in people:
        if blur:
            blur_face(frame, person)
        label = labels.get(person.track_id, UNKNOWN)
        color = COLORS[label]
        kp = person.keypoints
        for a, b in SKELETON:
            if kp[a, 2] > 0.4 and kp[b, 2] > 0.4:
                cv2.line(frame, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]),
                         color, 1, cv2.LINE_AA)
        x1, y1, x2, y2 = person.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"P{person.track_id} {label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def run_source(source, name, args, store, out):
    """One video file or camera, start to finish. Returns its events."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source!r}")
    live = isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = PersonTracker(args.weights, args.conf, args.imgsz, args.device)
    rules = Rules(name, store)
    cache = TrackWriter(Path(args.save_tracks) / f"{name}.npz", fps) if args.save_tracks else None
    writer = None
    started = time.perf_counter()
    frame_no = processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if (frame_no - 1) % args.every:
                continue
            # video time for files (reproducible), wall time for a live camera
            t = time.perf_counter() - started if live else (frame_no - 1) / fps

            raw = tracker.update(frame, dedupe=False)
            if cache is not None:
                cache.add(t, raw)
            people, labels = rules.step(t, raw)
            processed += 1

            if args.save_video or args.show:
                frame = draw(frame, people, labels, args.blur_faces)
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out / f"{name}.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps / args.every,
                                                 (frame.shape[1], frame.shape[0]))
                    writer.write(frame)
                if args.show:
                    cv2.imshow("office monitor  (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        if cache is not None:
            cache.close()

    events = rules.finish()
    elapsed = time.perf_counter() - started
    people = len({e.person for e in events})
    print(f"{name}: {processed} frames in {elapsed:.0f}s ({processed / max(elapsed, 1e-6):.1f} fps), "
          f"{people} people, {len(events)} events")
    return events


class Rules:
    """Everything after the model: duplicate removal, posture, timeline.

    Live runs and cache replays both go through this one class, so identical tracks
    always give identical events -- otherwise a result tuned on the cache would not be
    the result the live system produces."""

    def __init__(self, name, store):
        self.postures = PostureTracker()
        self.log = EventLog(name)
        self.store = store

    def step(self, t, raw_people):
        people = drop_nested(raw_people)
        labels = {p.track_id: self.postures.update(p, t) for p in people}
        self.store.write(self.log.update(t, labels,
                                        {i: self.postures.onset(i) for i in labels}))
        for gone in set(self.postures.history) - set(self.log.tracks):
            self.postures.forget(gone)
        return people, labels

    def finish(self):
        self.store.write(self.log.close())
        return self.log.events


def replay_source(path, store):
    """Rules only, from a cached .npz: no model, no video."""
    rules = Rules(path.stem, store)
    for t, raw in read_tracks(path):
        rules.step(t, raw)
    return rules.finish()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.tracks:
        caches = sorted(Path(args.tracks).glob("*.npz"))[: args.limit]
        if not caches:
            raise SystemExit(f"no .npz track caches in {args.tracks}")
        store = EventStore(out)
        started = time.perf_counter()
        try:
            events = [e for path in caches for e in replay_source(path, store)]
        finally:
            store.close()
        print(f"replayed {len(caches)} cached videos in {time.perf_counter() - started:.1f}s")
        write_summary(out, events)
        return

    if args.video:
        path = Path(args.video)
        if path.is_dir():
            sources = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.suffix.lower() == ".txt":
            sources = [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
        else:
            sources = [path]
        sources = [(str(p), p.stem) for p in sources[: args.limit]]
        if not sources:
            raise SystemExit(f"no videos found at {args.video}")
    else:
        sources = [(args.camera, f"camera{args.camera}")]

    store = EventStore(out)
    events = []
    try:
        for source, name in sources:
            events += run_source(source, name, args, store, out)
    finally:
        store.close()
    write_summary(out, events)
    if args.save_video:
        print(f"annotated videos in {out}/")
    if args.save_tracks:
        print(f"track caches in {args.save_tracks}/")


def write_summary(out, events):
    with open(out / "summary.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["source", "person", "sitting_s", "standing_s", "walking_s"])
        for (source, person), totals in sorted(summarise(events).items()):
            rows.writerow([source, person, *(round(totals.get(k, 0.0), 1)
                                              for k in (SITTING, STANDING, WALKING))])
    print(f"wrote {out}/ (events.db, events.csv, summary.csv)")


if __name__ == "__main__":
    main()
