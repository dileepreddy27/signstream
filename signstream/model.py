import os
from pathlib import Path
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import keras  # noqa: E402
from .features import LABELS, features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Recognizer:
    def __init__(self, path=None):
        self.model = keras.models.load_model(path or ROOT / "models/motion.keras", compile=False)

    def predict(self, frames):
        x = features(frames)
        scores = np.asarray(self.model(x[None], training=False))[0]
        index = int(scores.argmax())
        confidence = float(scores[index])
        # Confidence is a model score, not calibrated real-world reliability.
        label = LABELS[index] if confidence >= 0.8 else None
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": dict(zip(LABELS, map(float, scores))),
            "mode": "synthetic-lstm",
        }
