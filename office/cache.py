"""
Save what the pose model + tracker saw, so the rules can be re-run without the model.

The model is the slow part: ~85 s per MM-Office clip on a laptop CPU. The posture rules
and event logic take milliseconds. Caching the tracker's output per video means a rule
change is re-scored across all 88 test recordings in seconds instead of hours -- only a
change to the model or tracker needs a fresh run.

One .npz per video:
    t          (F,)        time in seconds of each processed frame
    fps        ()          the video's frame rate
    frame      (P,)        index into t, one entry per detected person
    track_id   (P,)
    box        (P, 4)      x1, y1, x2, y2
    score      (P,)
    keypoints  (P, 17, 3)  x, y, confidence

The cache holds the tracker's RAW output, duplicate boxes included -- deduplication is a
rule, and rules are what the cache exists to let us change.
"""

from pathlib import Path

import numpy as np

from .perception import Person


class TrackWriter:
    def __init__(self, path, fps):
        self.path = Path(path)
        self.fps = fps
        self.t, self.rows = [], []

    def add(self, t, people):
        index = len(self.t)
        self.t.append(t)
        self.rows += [(index, p) for p in people]

    def close(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        people = [p for _, p in self.rows]
        np.savez_compressed(
            self.path,
            t=np.asarray(self.t, np.float64),
            fps=np.float64(self.fps),
            frame=np.asarray([i for i, _ in self.rows], np.int32),
            track_id=np.asarray([p.track_id for p in people], np.int32),
            box=np.asarray([p.box for p in people], np.int32).reshape(-1, 4),
            score=np.asarray([p.score for p in people], np.float32),
            keypoints=np.asarray([p.keypoints for p in people], np.float32).reshape(-1, 17, 3),
        )


def read_tracks(path):
    """Yield (t, [Person, ...]) for every processed frame, in order, including frames
    where nobody was seen -- the event log needs those to notice people leaving."""
    # dict() loads every array once. Indexing an NpzFile directly decompresses the whole
    # array again on every access -- per person, per frame -- which made a 70 s clip
    # take 15 s to replay.
    with np.load(path) as npz:
        data = dict(npz)
    t, frame = data["t"], data["frame"]
    order = np.argsort(frame, kind="stable")
    starts = np.searchsorted(frame[order], np.arange(len(t) + 1))
    for i, when in enumerate(t):
        rows = order[starts[i]:starts[i + 1]]
        yield float(when), [
            Person(track_id=int(data["track_id"][r]), box=tuple(int(v) for v in data["box"][r]),
                   score=float(data["score"][r]), keypoints=data["keypoints"][r])
            for r in rows
        ]
