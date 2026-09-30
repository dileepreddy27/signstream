import numpy as np

FRAMES = 32
POINTS = 543  # pose 33, face 468, left hand 21, right hand 21
LABELS = ["wave", "sweep", "raise", "circle"]


def features(frames):
    """Shoulder-relative wrist trajectories; zero hands remain missing, not translated."""
    x = np.asarray(frames, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (POINTS, 3) or not 8 <= len(x) <= 96:
        raise ValueError("Expected 8–96 frames of 543 xyz landmarks")
    if not np.isfinite(x).all() or np.abs(x).max() > 10:
        raise ValueError("Coordinates must be finite and within [-10, 10]")
    shoulders = x[:, [11, 12], :]
    if np.any(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1) < 0.03):
        raise ValueError("Both shoulders must be visible")
    center = shoulders.mean(axis=1)
    scale = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    hands = x[:, [501, 522], :]
    visible = np.any(hands != 0, axis=2)
    if visible[:, 1].mean() < 0.8:
        raise ValueError("Right hand must be detected in at least 80% of frames")
    relative = (hands - center[:, None]) / scale[:, None, None]
    relative[~visible] = 0
    raw = np.concatenate([relative.reshape(len(x), 6), visible.astype(np.float32)], axis=1)
    times = np.linspace(0, len(x) - 1, FRAMES)
    return np.stack([np.interp(times, np.arange(len(x)), raw[:, i]) for i in range(8)], axis=1).astype(
        np.float32
    )


class TokenBuffer:
    """Require a neutral interval before the same label can be emitted again."""

    def __init__(self):
        self.tokens = []
        self.last = None

    def accept(self, label):
        if label is None:
            self.last = None
        elif label != self.last:
            self.tokens.append(label)
            self.tokens = self.tokens[-24:]
            self.last = label
        return list(self.tokens)
