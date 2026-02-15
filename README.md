# DUI-Vision: Field Sobriety Test (FST) Edge Assistant

  DUI-Vision is a Jetson-based edge system that runs computer vision on live video to assist
  Horizontal Gaze Nystagmus (HGN) field testing workflows in near real time.

  ## What it does
  - Captures face/eye video from a local camera
  - Tracks pupil motion and estimates HGN indicators
  - Computes:
    - Lack of smooth pursuit (L/R)
    - Nystagmus prior to 45° (L/R)
    - Distinct nystagmus at max deviation (L/R)
    - Vertical nystagmus estimate
  - Detects excessive head movement during tests
  - Publishes completed session results to a local API
  - Serves a web dashboard for latest result + session history

  ## Tech stack
  - **Edge/device**: NVIDIA Jetson Nano
  - **Languages**: Python 3, JavaScript, HTML, CSS
  - **CV/ML**: OpenCV, NumPy, PyTorch, NanoOWL (OWL-ViT), Pillow
  - **Backend/API**: Python `http.server` + `ThreadingHTTPServer` (REST)
  - **Storage**: SQLite + CSV export

  ## Run it locally
  ```bash
  # Terminal 1: start API + dashboard
  python3 api_server.py

  # Terminal 2: run HGN edge capture
  python3 hgn_tracker3.py

  Dashboard: http://localhost:8000
  API receives finalized sessions at POST /api/session/finish

  ## Why this matters for judges

  This demonstrates a practical edge-first safety workflow:

  - on-device inference (no mandatory cloud dependency),
  - lightweight persistence,
  - structured result delivery,
  - browser dashboard for review.

  ## Privacy note

  This is a field operations prototype; no public live law-enforcement endpoint is exposed in this
  release. Data handling is local-first, and sensitive operational data is intentionally not
  published publicly.

  ## Notes

  test_eyes.py is a validation utility used during development to test pupil extraction behavior
  before full pipeline integration.
