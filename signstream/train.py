"""Reproducible train/validation/test partition and baseline evaluation."""

import json
import time
import numpy as np
from .model import ROOT, keras
from .data import dataset
from .features import LABELS


def main():
    keras.utils.set_random_seed(42)
    train_x, train_y = dataset(1000, 100)
    val_x, val_y = dataset(2000, 25)
    test_x, test_y = dataset(3000, 50)
    model = keras.Sequential(
        [
            keras.Input((32, 8)),
            keras.layers.LSTM(24),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(4, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(0.008), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=18, batch_size=32, verbose=2)
    predicted = np.asarray(model(test_x, training=False)).argmax(axis=1)
    # Nearest-centroid baseline sees time-averaged features, losing trajectory ordering.
    centroids = np.stack([train_x[train_y == i].mean(axis=(0, 1)) for i in range(4)])
    baseline = ((test_x.mean(axis=1)[:, None] - centroids) ** 2).sum(axis=2).argmin(axis=1)
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (test_y, predicted), 1)
    timings = []
    for _ in range(25):
        start = time.perf_counter()
        model(test_x[:1], training=False)
        timings.append((time.perf_counter() - start) * 1000)
    report = {
        "scope": "Synthetic motion only; no human sign-language accuracy measured",
        "labels": LABELS,
        "seed": 42,
        "train": 400,
        "validation": 100,
        "test": 200,
        "split_seeds": [1000, 2000, 3000],
        "epochs": len(history.history["loss"]),
        "test_accuracy": float((predicted == test_y).mean()),
        "mean_feature_centroid_accuracy": float((baseline == test_y).mean()),
        "confusion_matrix_true_rows": confusion.tolist(),
        "single_window_cpu_ms_median": round(float(np.median(timings)), 2),
        "parameters": model.count_params(),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    model.save(ROOT / "models/motion.keras")
    (ROOT / "models/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
