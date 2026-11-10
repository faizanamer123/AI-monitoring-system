"""
A small model that reads a person's movement: sit down, stand up, or neither.

Input is 3 seconds of the 17 body points YOLO-pose already produces, so it adds almost no
cost and has no idea what the room or the clothes look like. The hand-written posture
rules judge one frame's geometry; this judges how the body moves over time.
"""

from collections import deque

import numpy as np
import torch
from torch import nn

from .perception import L_HIP, R_HIP
from .posture import midpoint, torso_length

WINDOW_S = 3.0
RATE = 15                       # samples per second
FRAMES = int(WINDOW_S * RATE)
CLASSES = ("none", "sit down", "stand up")
MAX_GAP_S = 0.25                # nearest sample must be this close, or the frame is blank


def window_features(times, boxes, keypoints, centre):
    """(FRAMES, 51) array for the window around `centre`, or None if too little is visible.

    Coordinates are relative to where the hips were on average in the window, in torso
    lengths: position and distance from the camera drop out, movement stays.
    """
    grid = centre + np.linspace(-WINDOW_S / 2, WINDOW_S / 2, FRAMES)
    index = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    nearest = np.where(np.abs(times[index - 1] - grid) <= np.abs(times[index] - grid),
                       index - 1, index)
    seen = np.abs(times[nearest] - grid) <= MAX_GAP_S
    if seen.sum() < FRAMES * 0.6:
        return None

    frames = keypoints[nearest]
    hips, torsos = [], []
    for used, kp, box in zip(seen, frames, boxes[nearest]):
        if not used:
            continue
        hip = midpoint(kp, L_HIP, R_HIP)
        hips.append(hip if hip is not None else
                    np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))
        torso = torso_length(kp)
        torsos.append(torso if torso else (box[3] - box[1]) / 3)
    origin, scale = np.mean(hips, axis=0), max(float(np.median(torsos)), 1.0)

    out = np.zeros((FRAMES, 51), np.float32)
    points = (frames[:, :, :2] - origin) / scale
    confidence = frames[:, :, 2]
    out[:, 0::3], out[:, 1::3] = np.clip(points[:, :, 0], -5, 5), np.clip(points[:, :, 1], -5, 5)
    out[:, 2::3] = confidence
    out[~seen] = 0.0
    out[:, 0::3][out[:, 2::3] < 0.1] = 0.0
    out[:, 1::3][out[:, 2::3] < 0.1] = 0.0
    return out


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        return x + self.layers(x)


class SkeletonTCN(nn.Module):
    """A temporal convolution network: dilated 1-D convolutions over time."""

    def __init__(self, channels=64):
        super().__init__()
        self.input = nn.Sequential(nn.Conv1d(51, channels, 1), nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*(Block(channels, d) for d in (1, 2, 4, 8)))
        self.output = nn.Linear(channels, len(CLASSES))

    def forward(self, x):           # x: (batch, FRAMES, 51)
        x = self.blocks(self.input(x.transpose(1, 2)))
        return self.output(x.mean(dim=2))


class SkeletonEvents:
    """Live sit down / stand up from the trained model, one person at a time.

    The model needs 3 s of movement centred on the event, so a detection arrives about
    1.5 s after it happened. It looks every `every_s` seconds and will not repeat the same
    event for a person within `spacing_s`. The 0.95 threshold is the one chosen on MEVA's
    validation day.
    """

    def __init__(self, path, threshold=0.95, every_s=0.5, spacing_s=3.0):
        self.model = SkeletonTCN()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        self.threshold = threshold
        self.every_s = every_s
        self.spacing_s = spacing_s
        self.history = {}
        self.last_check = {}
        self.last_event = {}

    def update(self, person, t):
        """Returns ("sit down" or "stand up", when it happened), or None."""
        track = person.track_id
        history = self.history.setdefault(track, deque())
        history.append((t, person.box, person.keypoints))
        while t - history[0][0] > WINDOW_S + 1.0:
            history.popleft()
        if (t - history[0][0] < WINDOW_S
                or t - self.last_check.get(track, -1e9) < self.every_s):
            return None
        self.last_check[track] = t

        centre = t - WINDOW_S / 2
        times, boxes, keypoints = zip(*history)
        window = window_features(np.array(times), np.array(boxes, np.float32),
                                 np.array(keypoints, np.float32), centre)
        if window is None:
            return None
        scores = probabilities(self.model, window[None])[0]
        best = int(np.argmax(scores[1:])) + 1
        name = CLASSES[best]
        if (scores[best] < self.threshold
                or centre - self.last_event.get((track, name), -1e9) <= self.spacing_s):
            return None
        self.last_event[(track, name)] = centre
        return name, float(centre)

    def forget(self, track_id):
        self.history.pop(track_id, None)
        self.last_check.pop(track_id, None)
        for key in [k for k in self.last_event if k[0] == track_id]:
            del self.last_event[key]


def probabilities(model, windows, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            out.append(torch.softmax(model(torch.from_numpy(windows[i:i + batch])), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))
