"""Original procedural fixtures, NOT examples of any natural sign language."""

import numpy as np
from .features import LABELS, POINTS, features


def fixture(label, seed=0, count=32):
    if label not in LABELS:
        raise ValueError("Unknown synthetic motion")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, count) ** rng.uniform(0.85, 1.15)
    x = np.zeros((count, POINTS, 3), dtype=np.float32)
    shift = rng.uniform(-0.08, 0.08, 2)
    width = rng.uniform(0.25, 0.38)
    x[:, 11, :2] = [0.5 - width / 2, 0.45] + shift
    x[:, 12, :2] = [0.5 + width / 2, 0.45] + shift
    if label == "wave":
        dx, dy = 0.65 + 0.28 * np.sin(4 * np.pi * t), -0.3 + t * 0
    elif label == "sweep":
        dx, dy = -0.05 + 1.4 * t, -0.3 + t * 0
    elif label == "raise":
        dx, dy = 0.65 + t * 0, 0.45 - 1.5 * t
    else:
        dx, dy = 0.65 + 0.35 * np.cos(2 * np.pi * t), -0.3 + 0.35 * np.sin(2 * np.pi * t)
    wrist = np.stack([0.5 + width * dx + shift[0], 0.45 + width * dy + shift[1]], axis=1)
    wrist += rng.normal(0, 0.006, wrist.shape)
    for j in range(21):
        x[:, 522 + j, :2] = wrist + np.array([j % 4, j // 4]) * 0.005
    return x


def dataset(seed, per_class):
    xs, ys = [], []
    for index, label in enumerate(LABELS):
        for k in range(per_class):
            xs.append(features(fixture(label, seed + index * 10000 + k)))
            ys.append(index)
    return np.asarray(xs), np.asarray(ys)
