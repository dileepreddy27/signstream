# Verification evidence

## Local training — September 29, 2026

Python 3.11.9, Keras 3.9.2, TensorFlow CPU 2.19.0, Windows. The checked-in
model was trained for 18 epochs using seed 42 and disjoint procedural
training/validation/test seeds. All 200 held-out test clips were classified
correctly (50 per class); the time-averaged nearest-centroid baseline classified
41% correctly. This is a test of synthetic trajectories, not sign-language skill.

The model has 3,636 parameters. Median eager single-window forward time was
756.37 ms over 25 calls on this local machine during concurrent work. This
excludes startup, preprocessing, MediaPipe, capture, and network transport. It
is not a throughput benchmark or an end-to-end real-time performance claim.
See `models/evaluation.json` for the full confusion matrix and run settings.

Dependency consistency (`python -m pip check`), Ruff lint, and JavaScript syntax
checks passed. **18 pytest tests passed** (50.74 seconds), covering model inference
on unseen fixtures, held-out synthetic accuracy, normalization and resampling,
invalid/missing coordinates, HTTP errors, WebSocket recovery, cross-origin
rejection, session isolation, deduplication, and bounded tokens. One third-party
Starlette/AnyIO deprecation warning remains; no tests failed.

Chromium browser regression passed: all four synthetic motions returned their
expected labels through the running WebSocket API, sequence clearing worked,
390-pixel mobile layout had no horizontal overflow, and no JavaScript page errors
were observed. `docs/demo.png` is a screenshot of that actual run. Browser launch
required execution outside the Windows sandbox due to a subprocess-pipe permission
restriction; the application itself needed no code workaround.

Public-bound source was scanned for common credential patterns, private absolute
paths, and email addresses; no matches were found. Virtual environments, caches,
scratch output, and environment files are ignored. Remote GitHub Actions results
are visible in the repository's Actions tab; publication verification is reported
separately after the push.

## Not verified

- Physical camera capture and MediaPipe landmark quality on human signing.
- Accuracy on any natural sign language, unseen signers, or real-world video.
- Sentence translation, GPU execution, public deployment, or production load.
- Retraining equality across operating systems (CI tests the saved model).
