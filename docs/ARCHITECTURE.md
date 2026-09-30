# Pipeline decisions

## Why a small temporal network?
The original goal is continuous sign-to-sentence translation. That requires
language-specific, accurately annotated video and an evaluation protocol with
held-out signers. This repository implements a smaller reproducible prerequisite:
end-to-end temporal classification with an honest procedural baseline. Keras LSTM
is used instead of a transformer because the demonstration has 32 time steps and
only 8 features. There is no seq2seq model hidden behind the UI.

## Data contract
Each frame is a `543 × 3` array in normalized MediaPipe coordinates (z is relative
depth, not meters). Order: pose, face, left hand, right hand. Missing landmarks
are zero-filled. Requests carry 8–96 frames and are resampled to 32 steps. The
generator and browser use the same ordering and server preprocessing path.
Coordinates are validated before inference; missing shoulders or a mostly
missing right wrist yield an error. Pose visibility/confidence is not transported,
so detecting a shoulder is not equivalent to confirming its tracking quality.

## Transport and state
HTTP `POST /api/predict` provides stateless inference for integrations; `GET
/api/fixture/{label}` returns explicit procedural test data. WebSocket `/ws`
accepts `{ "frames": [...] }` or `{ "reset": true }`. Errors are recoverable
without closing a valid socket. Each connection owns its token buffer. A shared
async lock serializes CPU inference, which runs off the event loop. This avoids
simultaneous calls but does not bound the number of queued clients.

The browser submits at most one outstanding request and drops completed capture
windows while busy. It does not grow an unbounded queue. Capture runs sequentially
to avoid overlapping MediaPipe inference. Stop and page navigation release media
tracks. No transcript is persisted beyond the active connection.

## Known operational boundaries
Use the documented loopback binding. A same-origin WebSocket check prevents
ordinary cross-site browser sockets, but is not authentication. HTTP endpoints
are unauthenticated. The Uvicorn WebSocket maximum is specified in quick start;
the server also checks text size after receipt. Public hosting needs upstream
HTTP body limits and connection/rate limits. Model files are trusted local
artifacts; there is no arbitrary model upload or remote deserialization route.

## Extension path
Replace procedural tracks with a documented, licensed language-specific corpus;
split by signer before augmentation; add face and articulated-hand features;
evaluate neutral/out-of-distribution rejection and continuous segmentation;
then introduce a gloss/translation decoder. Keep the procedural fixtures for
transport and regression tests, not as a substitute for linguistic evaluation.
