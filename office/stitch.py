"""
Re-join tracks the tracker broke: same place, a few seconds later, same person.

ByteTrack matches boxes frame to frame. Someone hidden behind a monitor for a few
seconds, or a seated person whose detection drops out, comes back under a new id, and
the timeline shows a departure and a stranger arriving. A new id that appears where a
missing one was last seen, shortly after, is mapped back onto the old id.

Positions are compared in box heights, so "the same place" scales with distance from
the camera. Everything runs online: an id is resolved the first time it is seen.
"""

from dataclasses import replace


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def touches_edge(box, size, margin=0.05):
    """Is this box within `margin` of the frame border? size is (width, height)."""
    width, height = size
    mx, my = width * margin, height * margin
    return box[0] < mx or box[1] < my or box[2] > width - mx or box[3] > height - my


class IdStitcher:
    def __init__(self, gap_s=5.0, reach=0.75):
        self.gap_s = gap_s
        self.reach = reach
        self.alias = {}     # raw tracker id -> stable id
        self.last = {}      # stable id -> (time last seen, box)

    def update(self, people, t):
        """The same people, with broken tracks mapped back to their original ids."""
        active = {self.alias[p.track_id] for p in people if p.track_id in self.alias}
        out = []
        # sorted, so two new ids competing for one lost id resolve the same way each run
        for person in sorted(people, key=lambda p: p.track_id):
            if person.track_id not in self.alias:
                found = self._lost_near(person, t, active)
                self.alias[person.track_id] = person.track_id if found is None else found
                active.add(self.alias[person.track_id])
            stable = self.alias[person.track_id]
            self.last[stable] = (t, person.box)
            out.append(replace(person, track_id=stable))
        return out

    def _lost_near(self, person, t, active):
        x, y = _centre(person.box)
        height = person.box[3] - person.box[1]
        best, best_distance = None, None
        for stable, (seen, box) in self.last.items():
            if stable in active or not 0 < t - seen <= self.gap_s:
                continue
            ox, oy = _centre(box)
            distance = ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5
            if distance <= self.reach * max(height, box[3] - box[1]) and (
                    best is None or distance < best_distance):
                best, best_distance = stable, distance
        return best
