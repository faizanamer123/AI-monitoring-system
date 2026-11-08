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
  change     a moment: sit down, stand up, enter or exit

PRIVACY. Only these rows are stored, never frames. The database answers "how long did
P3 sit" without holding a single image of P3.

Ids come from the tracker, so they are per-video and anonymous: P3 is "the third track",
not a named employee. If the tracker loses someone behind a pillar and re-finds them,
they come back as a new id -- the timeline shows a left + appeared pair.
"""

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from .posture import SITTING, UNKNOWN


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


class PostureChanges:
    """Per-person sit-down / stand-up detection, online, from OBSERVED posture only.

    Pass UNKNOWN for anything not actually seen, including sitting inferred from
    stillness: that inference is fine for a timeline, but it fires when someone merely
    stops moving, and on MM-Office it was the largest single source of false sit-downs.

    Standing and walking are one "upright" state, so stand up then walk away is a single
    change. A new state must persist for `confirm_s` to replace the old one, so a flicker
    is no change at all. If the two states are more than `max_gap` apart we do not know
    when the change happened, and no event is reported.
    """

    def __init__(self, confirm_s=1.0, max_gap=5.0):
        self.confirm_s = confirm_s
        self.max_gap = max_gap
        self.people = {}

    def update(self, track_id, label, t):
        """Returns ("sit down" or "stand up", when it happened), or None."""
        if label == UNKNOWN:
            return None
        seated = label == SITTING
        person = self.people.setdefault(track_id, {"state": None, "last": None,
                                                   "candidate": None, "since": None, "seen": None})
        if seated == person["state"]:
            person["last"], person["candidate"] = t, None
            return None
        if person["candidate"] != seated or t - person["seen"] > self.max_gap:
            person["candidate"], person["since"] = seated, t
        person["seen"] = t
        if t - person["since"] < self.confirm_s:
            return None
        before, last, since = person["state"], person["last"], person["since"]
        person.update(state=seated, last=t, candidate=None)
        if before is None or since - last > self.max_gap:
            return None
        return ("sit down" if seated else "stand up"), (last + since) / 2

    def forget(self, track_id):
        self.people.pop(track_id, None)


class EventLog:
    """Feed update(t, {track_id: label}) once per processed frame, then close(t)."""

    def __init__(self, source, lost_after=2.0, min_segment=1.0, hold_unknown=3.0,
                 kind="posture", presence=True, confirm_s=0.0, settle=1.0):
        """kind names the segments this log writes ("posture", "activity"). One person
        has one timeline per kind; only one of them should own presence, or every
        arrival is logged once per kind.

        confirm_s: an id writes nothing until it has existed this long, then its events
        are released, backdated. Most ids that die young are a chair or a coat detected
        for a few frames, and a real person is always around longer than that.

        settle: someone at the frame edge this soon after the start was already in the
        room, so it is not an enter."""
        self.started = None
        self.settle = settle
        self.source = source
        self.kind = kind
        self.presence = presence
        self.confirm_s = confirm_s
        self.lost_after = lost_after
        self.min_segment = min_segment
        # A seated person whose knees vanish behind the desk for a moment reads as
        # unknown. Up to this many seconds of unknown keep the last real posture
        # rather than cutting the sitting stretch in two.
        self.hold_unknown = hold_unknown
        self.tracks = {}    # id -> {"label", "since", "last_seen", "last_known"}
        self.events = []

    def update(self, t, labels, onsets=None, edges=None):
        """Returns the events this call closed, for live display.

        onsets: optional {id: time} for labels that truly began before `t` (see
        PostureTracker.onset). A new segment is backdated to it, never earlier than
        the segment it replaces began.

        edges: optional {id: bool}, is this person touching the frame border. Recorded
        as the label of appeared / left ("edge" or "inside"): someone walking in arrives
        at the border, while a person re-detected mid-room did not enter anything."""
        onsets = onsets or {}
        edges = edges or {}
        before = len(self.events)
        if self.started is None:
            self.started = t
        for track_id, label in labels.items():
            track = self.tracks.get(track_id)
            if track is None:
                track = {"label": label, "since": t, "last_seen": t, "last_known": t,
                         "first_seen": t, "confirmed": self.confirm_s <= 0, "pending": [],
                         "at_edge": edges.get(track_id)}
                self.tracks[track_id] = track
                if self.presence:
                    self._emit(track, Event(self.source, track_id, "appeared",
                                            self._place(track), t, t))
                    if track["at_edge"] and t - self.started > self.settle:
                        self._emit(track, Event(self.source, track_id, "change", "enter", t, t))
                continue
            track["last_seen"] = t
            track["at_edge"] = edges.get(track_id)
            if not track["confirmed"] and t - track["first_seen"] >= self.confirm_s:
                track["confirmed"] = True
                self.events += track["pending"]
                track["pending"] = []
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
            self._emit(track, Event(self.source, track_id, self.kind, track["label"],
                                    track["since"], end))

    def _emit(self, track, event):
        # an unconfirmed id's events wait; if it dies young they are simply never released
        (self.events if track["confirmed"] else track["pending"]).append(event)

    @staticmethod
    def _place(track):
        return "" if track["at_edge"] is None else ("edge" if track["at_edge"] else "inside")

    @staticmethod
    def _segment_end(track):
        # a posture carried through trailing unknown frames ends where it was last seen
        return track["last_seen"] if track["label"] == UNKNOWN else track["last_known"]

    def _drop(self, track_id):
        track = self.tracks.pop(track_id)
        self._close_segment(track_id, track, self._segment_end(track))
        if self.presence:
            self._emit(track, Event(self.source, track_id, "left", self._place(track),
                                    track["last_seen"], track["last_seen"]))
            if track["at_edge"]:
                self._emit(track, Event(self.source, track_id, "change", "exit",
                                        track["last_seen"], track["last_seen"]))

    def add_change(self, track_id, label, when):
        """Record a sit down / stand up, held back like everything else until the id
        is confirmed."""
        track = self.tracks.get(track_id)
        if track is not None:
            self._emit(track, Event(self.source, track_id, "change", label, when, when))

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

    COLUMNS = ("session", "source", "person", "kind", "label", "start", "end", "duration")

    def __init__(self, out_dir, session=None):
        """Every run appends under its own session id; nothing earlier is ever erased."""
        self.session = session or datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_path = out_dir / "events.csv"
        new_file = not csv_path.is_file()
        if not new_file:
            with open(csv_path, newline="") as f:
                if tuple(next(csv.reader(f), ())) != self.COLUMNS:
                    raise SystemExit(f"{csv_path} is in an older format; use a new --out folder")
        self.db = sqlite3.connect(out_dir / "events.db")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events (session TEXT, source TEXT, person INTEGER, "
            "kind TEXT, label TEXT, start REAL, \"end\" REAL, duration REAL)")
        self.csv_file = open(csv_path, "a", newline="")
        self.csv = csv.writer(self.csv_file)
        if new_file:
            self.csv.writerow(self.COLUMNS)

    def write(self, events):
        rows = [(self.session, e.source, e.person, e.kind, e.label, round(e.start, 2),
                 round(e.end, 2), round(e.duration, 2)) for e in events]
        self.db.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self.db.commit()
        self.csv.writerows(rows)
        self.csv_file.flush()

    def close(self):
        self.db.close()
        self.csv_file.close()
