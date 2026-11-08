"""
Train and test the skeleton model for sit down / stand up on MEVA.

    python train_skeleton.py prepare TRACKS_DIR LABEL_DIR CACHE_DIR
    python train_skeleton.py train CACHE_DIR RUN_DIR [--test-only-cameras G331 --train-cameras G421]

Labels name a person id, and MEVA gives that person's box per frame, so each labelled
event attaches to the YOLO track overlapping it. Positives are windows around those
moments. Negatives are other windows from the same tracks: mostly ordinary sitting,
standing and walking, which is what the model must not fire on.

The test clips are the 60 the rules were scored on, so the two are directly comparable,
and the model never trains on them. Validation is a held-out recording day inside the
training clips, and picks the decision threshold.
"""

import argparse
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_meva import read_labels
from evaluate_office import add, intervals, match, merge, prf, table
from office.cache import read_tracks
from office.perception import drop_nested
from office.skeleton import CLASSES, FRAMES, SkeletonTCN, probabilities, window_features
from office.stitch import IdStitcher

FPS = 30.0
EVENTS = {"person_sits_down": "sit down", "person_stands_up": "stand up"}
MOMENTS = ("sit down", "stand up")
ACT = re.compile(r"act2'?:\s*\{\s*'?([a-z_]+)'?\s*:")
ACTOR = re.compile(r"actors'?:\s*\[\s*\{\s*'?id1'?:\s*(\d+)")
SPAN = re.compile(r"tsr0'?:\s*\[\s*(\d+)\s*,\s*(\d+)")
GEOM = re.compile(r"g0'?:\s*'?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
ID1 = re.compile(r"id1'?:\s*(\d+)")
TS0 = re.compile(r"ts0'?:\s*(\d+)")
VALIDATION_DAY = "2018-03-13"


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


def read_events(activities):
    """[(event, start s, end s, actor id)] for sit down / stand up."""
    out = []
    for line in Path(activities).read_text(errors="replace").splitlines():
        act, actor, span = ACT.search(line), ACTOR.search(line), SPAN.search(line)
        if act and act[1] in EVENTS and actor and span:
            out.append((EVENTS[act[1]], int(span[1]) / FPS, int(span[2]) / FPS, int(actor[1])))
    return out


def read_geom(path):
    """{actor id: (frame numbers, boxes)}"""
    tracks = defaultdict(list)
    for line in Path(path).read_text(errors="replace").splitlines():
        g, i, ts = GEOM.search(line), ID1.search(line), TS0.search(line)
        if g and i and ts:
            tracks[int(i[1])].append((int(ts[1]), *map(int, g.groups())))
    return {k: (np.array([r[0] for r in v]), np.array([r[1:] for r in v])) for k, v in tracks.items()}


def load_people(track_path):
    """{track id: (times, boxes, keypoints)}, duplicates removed and broken tracks re-joined
    exactly as the live rules do."""
    stitcher, people = IdStitcher(5.0), defaultdict(lambda: ([], [], []))
    for t, raw, _ in read_tracks(track_path):
        for p in stitcher.update(drop_nested(raw), t):
            times, boxes, keypoints = people[p.track_id]
            times.append(t)
            boxes.append(p.box)
            keypoints.append(p.keypoints)
    return {k: (np.array(v[0]), np.array(v[1], np.float32), np.array(v[2], np.float32))
            for k, v in people.items() if len(v[0]) >= FRAMES // 2}


def attach(events, geom, people):
    """[(event, centre time, track id)]: each labelled event on the YOLO track covering it."""
    attached, missed = [], 0
    for cls, start, end, actor in events:
        centre = (start + end) / 2
        best, best_iou = None, 0.3
        if actor in geom:
            frames, boxes = geom[actor]
            nearest = int(np.argmin(np.abs(frames - centre * FPS)))
            if abs(frames[nearest] - centre * FPS) <= 15:
                for track, (times, track_boxes, _) in people.items():
                    i = int(np.argmin(np.abs(times - centre)))
                    if abs(times[i] - centre) <= 0.4:
                        overlap = iou(boxes[nearest], track_boxes[i])
                        if overlap > best_iou:
                            best, best_iou = track, overlap
        if best is None:
            missed += 1
        else:
            attached.append((cls, centre, best))
    return attached, missed


def prepare(args):
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    total, missed = 0, 0
    for path in sorted(Path(args.tracks).glob("*.npz")):
        clip = path.stem.removesuffix(".r13")
        activities = Path(args.labels) / f"{clip}.activities.yml"
        geom_path = Path(args.labels) / f"{clip}.geom.yml"
        if not activities.is_file() or not geom_path.is_file():
            print("no labels for", clip)
            continue
        people = load_people(path)
        attached, lost = attach(read_events(activities), read_geom(geom_path), people)
        total += len(attached)
        missed += lost
        with open(out / f"{clip}.pkl", "wb") as f:
            pickle.dump({"people": people, "attached": attached,
                         "truth": read_labels(activities)}, f)
    print(f"{total} events attached to a track, {missed} with no matching track")


def load_clip(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def training_windows(clip, rng, negatives_per_positive):
    """Windows and labels from one clip."""
    people, attached = clip["people"], clip["attached"]
    xs, ys = [], []
    for cls, centre, track in attached:
        times, boxes, keypoints = people[track]
        for jitter in (-0.5, -0.25, 0.0, 0.25, 0.5):
            x = window_features(times, boxes, keypoints, centre + jitter)
            if x is not None:
                xs.append(x)
                ys.append(CLASSES.index(cls))
    positives = len(xs)
    candidates = []
    for track, (times, boxes, keypoints) in people.items():
        busy = [c for _, c, t in attached if t == track]
        for centre in np.arange(times[0] + 1.5, times[-1] - 1.5, 1.0):
            if all(abs(centre - c) > 2.5 for c in busy):
                candidates.append((track, centre))
    rng.shuffle(candidates)
    wanted = max(positives, 5) * negatives_per_positive
    for track, centre in candidates[:wanted]:
        times, boxes, keypoints = people[track]
        x = window_features(times, boxes, keypoints, float(centre))
        if x is not None:
            xs.append(x)
            ys.append(0)
    return xs, ys


def fit(train_paths, epochs, seed, negatives_per_positive):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    xs, ys = [], []
    for path in train_paths:
        x, y = training_windows(load_clip(path), rng, negatives_per_positive)
        xs += x
        ys += y
    X, Y = torch.from_numpy(np.stack(xs)), torch.tensor(ys)
    print(f"training windows: {len(Y)}  " + "  ".join(f"{c}: {(Y == i).sum().item()}"
                                                      for i, c in enumerate(CLASSES)), flush=True)
    counts = torch.bincount(Y, minlength=len(CLASSES)).float()
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / counts).sqrt())
    model = SkeletonTCN()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=True)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += float(loss) * len(yb)
        print(f"epoch {epoch + 1}/{epochs}  loss {running / len(Y):.4f}", flush=True)
    return model


def clip_scores(model, clip):
    """{event: [(track, time, probability), ...]} from windows every half second."""
    people = clip["people"]
    index, windows = [], []
    for track, (times, boxes, keypoints) in people.items():
        for centre in np.arange(times[0] + 1.5, times[-1] - 1.5, 0.5):
            x = window_features(times, boxes, keypoints, float(centre))
            if x is not None:
                windows.append(x)
                index.append((track, float(centre)))
    probs = probabilities(model, np.stack(windows)) if windows else np.zeros((0, len(CLASSES)))
    return {cls: [(track, centre, float(p[i])) for (track, centre), p in zip(index, probs)]
            for i, cls in enumerate(CLASSES) if i}


def detect(scores, threshold, spacing=3.0):
    """Peak-picking per track: the best window, then nothing else within `spacing` s."""
    points = []
    for cls, rows in scores.items():
        by_track = defaultdict(list)
        for track, time, prob in rows:
            if prob >= threshold:
                by_track[track].append((prob, time))
        for values in by_track.values():
            kept = []
            for prob, time in sorted(values, reverse=True):
                if all(abs(time - other) > spacing for other in kept):
                    kept.append(time)
            points += [(cls, t) for t in kept]
    return points


def evaluate(scored_clips, threshold, collar=2.0):
    per_clip, pooled = {}, {}
    for name, (scores, truth) in scored_clips.items():
        counts = match(merge(detect(scores, threshold), collar), truth, collar, None, MOMENTS)
        per_clip[name] = dict(counts)
        add(pooled, counts)
    return pooled, per_clip


def f1_of(pooled):
    tp, fp, fn = (sum(pooled.get(c, [0, 0, 0])[i] for c in MOMENTS) for i in range(3))
    return prf(tp, fp, fn)


def train(args):
    cache, run = Path(args.cache), Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    test_names = set(Path(args.test_list).read_text().split())
    paths = sorted(cache.glob("*.pkl"))
    cameras = lambda p: p.stem.rsplit(".", 1)[1]
    test_paths = [p for p in paths if p.stem in test_names
                  and (not args.test_cameras or cameras(p) in args.test_cameras)]
    rest = [p for p in paths if p.stem not in test_names
            and (not args.train_cameras or cameras(p) in args.train_cameras)]
    val_paths = [p for p in rest if p.name.startswith(VALIDATION_DAY)]
    train_paths = [p for p in rest if not p.name.startswith(VALIDATION_DAY)]
    print(f"clips: {len(train_paths)} train, {len(val_paths)} validation, {len(test_paths)} test",
          flush=True)

    model = fit(train_paths, args.epochs, args.seed, args.negatives)
    torch.save(model.state_dict(), run / "skeleton_tcn.pt")

    scored_val = {p.stem: (clip_scores(model, c := load_clip(p)), c["truth"]) for p in val_paths}
    grid = np.arange(0.3, 0.96, 0.05)
    results = [(t, *f1_of(evaluate(scored_val, t)[0])) for t in grid]
    best = max(results, key=lambda r: r[3])
    print("\nvalidation (threshold, precision, recall, F1):")
    for t, p, r, f in results:
        print(f"  {t:.2f}  {p:.3f}  {r:.3f}  {f:.3f}{'   <- chosen' if t == best[0] else ''}")

    scored_test = {p.stem: (clip_scores(model, c := load_clip(p)), c["truth"]) for p in test_paths}
    pooled, per_clip = evaluate(scored_test, best[0])
    p, r, f = f1_of(pooled)
    (pl, ph), (rl, rh), (fl, fh) = intervals(per_clip, MOMENTS)
    lines = [f"test: {len(test_paths)} clips, threshold {best[0]:.2f}",
             f"  sit down + stand up  precision {p:.3f} ({pl:.3f}-{ph:.3f})  "
             f"recall {r:.3f} ({rl:.3f}-{rh:.3f})  F1 {f:.3f} ({fl:.3f}-{fh:.3f})"]
    print("\n" + "\n".join(lines))
    table("TEST", pooled, MOMENTS)
    (run / "scores.txt").write_text("\n".join(lines) + "\n")
    with open(run / "test_scores.pkl", "wb") as out:
        pickle.dump(scored_test, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("tracks")
    p.add_argument("labels")
    p.add_argument("cache")
    t = sub.add_parser("train")
    t.add_argument("cache")
    t.add_argument("run")
    t.add_argument("test_list", help="text file listing the test clip names")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--negatives", type=int, default=6, help="negative windows per positive")
    t.add_argument("--train-cameras", nargs="*", default=None)
    t.add_argument("--test-cameras", nargs="*", default=None)
    args = parser.parse_args()
    prepare(args) if args.cmd == "prepare" else train(args)


if __name__ == "__main__":
    main()
