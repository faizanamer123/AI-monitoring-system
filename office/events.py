"""
Turn per-frame labels into a timeline of events.

    every frame:   {3: "sitting", 5: "walking"}
    becomes:       P3 appeared 12.0 | P3 sitting 12.0-48.5 | P3 standing 48.5-50.1 | P3 left 61.0

Event kinds:
  appeared   a new person id, instantaneous
  posture    a stretch of sitting / standing / walking, with start, end and duration
  activity   a stretch of on_call / using_phone / working, in parallel with posture --
             "sitting" and "on_call" overlap, which is the point
  left       the id was not seen for `lost_after` seconds; timed at the last sighting

PRIVACY. Only these rows are stored, never frames. The database answers "how long did
P3 sit" without holding a single image of P3.

Ids come from the tracker, so they are per-video and anonymous: P3 is "the third track",
not a named employee. If the tracker loses someone behind a pillar and re-finds them,
they come back as a new id -- the timeline shows a left + appeared pair.
"""

import csv
import sqlite3
from dataclasses import dataclass

from .posture import UNKNOWN


@dataclass
class Event:
    source: str
    person: int
    kind: str           # appeared / posture / left
    label: str          # the posture, or "" for appeared / left
    start: float        # seconds from the start of the source
    end: float

    @property
    def duration(self):
        return self.end - self.start


class EventLog:
    """Feed update(t, {track_id: label}) once per processed frame, then close(t)."""

    def __init__(self, source, lost_after=2.0, min_segment=1.0, hold_unknown=3.0,
                 kind="posture", presence=True):
        """kind names the segments this log writes ("posture", "activity"). One person
        has one timeline per kind; only one of them should own presence, or every
        arrival is logged once per kind."""
        self.source = source
        self.kind = kind
        self.presence = presence
        self.lost_after = lost_after
        self.min_segment = min_segment
        # A seated person whose knees vanish behind the desk for a moment reads as
        # unknown. Up to this many seconds of unknown keep the last real posture
        # rather than cutting the sitting stretch in two.
        self.hold_unknown = hold_unknown
        self.tracks = {}    # id -> {"label", "since", "last_seen", "last_known"}
        self.events = []

    def update(self, t, labels, onsets=None):
        """Returns the events this call closed, for live display.

        onsets: optional {id: time} for labels that truly began before `t` (see
        PostureTracker.onset). A new segment is backdated to it, never earlier than
        the segment it replaces began."""
        onsets = onsets or {}
        before = len(self.events)
        for track_id, label in labels.items():
            track = self.tracks.get(track_id)
            if track is None:
                if self.presence:
                    self.events.append(Event(self.source, track_id, "appeared", "", t, t))
                self.tracks[track_id] = {"label": label, "since": t, "last_seen": t,
                                         "last_known": t}
                continue
            track["last_seen"] = t
            end = t
            if label == UNKNOWN and track["label"] != UNKNOWN:
                if t - track["last_known"] <= self.hold_unknown:
                    continue
                end = track["last_known"]   # the posture was last actually seen here
            if label != UNKNOWN:
                track["last_known"] = t
                if onsets.get(track_id) is not None:
                    end = min(t, max(onsets[track_id], track["since"]))
            if label != track["label"]:
                self._close_segment(track_id, track, end)
                track["label"], track["since"] = label, end

        for track_id in [i for i, tr in self.tracks.items()
                         if i not in labels and t - tr["last_seen"] > self.lost_after]:
            self._drop(track_id)
        return self.events[before:]

    def close(self):
        """End of the source: close every open segment. Returns the events it closed.
        No `left` events -- the people did not leave, the recording stopped."""
        before = len(self.events)
        for track_id, track in self.tracks.items():
            self._close_segment(track_id, track, self._segment_end(track))
        self.tracks.clear()
        return self.events[before:]

    def _close_segment(self, track_id, track, end):
        # UNKNOWN stretches are gaps in knowledge, not an activity. Very short segments
        # are flicker the smoothing let through; dropping them keeps the timeline
        # readable at the cost of a second's precision at each boundary.
        if track["label"] != UNKNOWN and end - track["since"] >= self.min_segment:
            self.events.append(Event(self.source, track_id, self.kind, track["label"],
                                     track["since"], end))

    @staticmethod
    def _segment_end(track):
        # a posture carried through trailing unknown frames ends where it was last seen
        return track["last_seen"] if track["label"] == UNKNOWN else track["last_known"]

    def _drop(self, track_id):
        track = self.tracks.pop(track_id)
        self._close_segment(track_id, track, self._segment_end(track))
        if self.presence:
            self.events.append(Event(self.source, track_id, "left", "",
                                     track["last_seen"], track["last_seen"]))

    def active(self):
        """{id: (label, seconds in it so far)} for people currently in view."""
        return {i: (tr["label"], tr["last_seen"] - tr["since"]) for i, tr in self.tracks.items()}


def summarise(events):
    """{(source, person): {label: seconds}} for postures and activities -- the "how was
    time spent" table. Labels never collide across kinds, so one dict holds both."""
    totals = {}
    for e in events:
        if e.kind in ("posture", "activity"):
            person = totals.setdefault((e.source, e.person), {})
            person[e.label] = person.get(e.label, 0.0) + e.duration
    return totals


class EventStore:
    """events.db (SQLite, for the dashboard to query) plus events.csv (to eyeball)."""

    COLUMNS = ("source", "person", "kind", "label", "start", "end", "duration")

    def __init__(self, out_dir):
        self.db = sqlite3.connect(out_dir / "events.db")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events (source TEXT, person INTEGER, kind TEXT, "
            "label TEXT, start REAL, \"end\" REAL, duration REAL)")
        # one run per output folder, matching the CSV, which is rewritten each time
        self.db.execute("DELETE FROM events")
        self.csv_file = open(out_dir / "events.csv", "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(self.COLUMNS)

    def write(self, events):
        rows = [(e.source, e.person, e.kind, e.label, round(e.start, 2), round(e.end, 2),
                 round(e.duration, 2)) for e in events]
        self.db.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        self.db.commit()
        self.csv.writerows(rows)
        self.csv_file.flush()

    def close(self):
        self.db.close()
        self.csv_file.close()
