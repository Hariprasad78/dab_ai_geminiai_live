# Vertex Live DAB Agent

AI-driven Android TV / Google TV automated testing tool using **Vertex AI (Gemini Live)**, **LiveKit**, and the **DAB (Device Automation Bus)** protocol.

## Overview

The agent runs an **observe → plan → act → verify** loop:
1. **Observe** – Captures a screenshot from **HDMI capture card** (preferred when available) or DAB, and optionally runs OCR
2. **Plan** – Uses Vertex AI (Gemini) or a heuristic planner to decide the next action
3. **Act** – Sends the action to the TV device via DAB (key press, app launch, etc.)
4. **Verify** – Validates the outcome semantically or deterministically

Everything works locally with **mock mode on** — no TV device, no cloud credentials required.

## Project Structure

```
vertex_live_dab_agent/
├── __main__.py         # python -m vertex_live_dab_agent entrypoint
├── config.py           # Centralized env-var config
├── dab/
│   ├── client.py       # DAB client (Mock + MQTT stub with extension guide)
│   └── topics.py       # DAB topic templates, key codes, format_topic()
├── planner/
│   ├── planner.py      # Planning layer (Vertex AI + heuristic fallback)
│   └── schemas.py      # ActionType enum + PlannedAction Pydantic schema
├── capture/
│   ├── capture.py      # Screenshot capture + optional OCR (pytesseract)
│   └── validator.py    # Semantic (Vertex AI) + deterministic validation
├── session/
│   └── manager.py      # Session lifecycle manager
├── orchestrator/
│   ├── orchestrator.py # Main observe→plan→act→verify loop + artifact saving
│   └── run_state.py    # RunState Pydantic model
├── api/
│   ├── api.py          # FastAPI backend (11 endpoints)
│   └── models.py       # Request/response Pydantic models
├── artifacts/
│   └── logger.py       # ArtifactStore + setup_logging()
└── livekit_agent/
    └── agent.py        # LiveKit agent entrypoint (stub)
static/
└── index.html          # Web UI demo
tests/
├── test_schemas.py
├── test_planner.py
├── test_dab_client.py
├── test_run_state.py
├── test_api.py
└── test_artifacts.py
```

## Quick Start (mock mode — no device needed)

### 1. Clone and install

```bash
git clone <repo-url>
cd dab_ai_geminiai_live

# Minimum install (mock mode, no optional deps)
pip install -e ".[dev]"

# With all optional extras
pip install -e ".[dev,vertex,ocr]"
```

### 2. Configure environment (optional for mock mode)

```bash
cp .env.example .env
# For mock mode the defaults are fine — nothing needs to be changed
```

### 3. Start the API server

```bash
# Option A — package entrypoint (reads API_HOST / API_PORT from env)
python -m vertex_live_dab_agent

# Option B — uvicorn directly (with auto-reload for development)
uvicorn vertex_live_dab_agent.api.api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the web UI

Browse to **http://localhost:8000** — the browser demo loads automatically.

Interactive API docs: **http://localhost:8000/docs**

### 5. Run the tests

```bash
pytest tests/ -v
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DAB_MOCK_MODE` | `true` | Use mock DAB client (no real device needed) |
| `GOOGLE_CLOUD_PROJECT` | `` | GCP project ID for Vertex AI |
| `GOOGLE_CLOUD_LOCATION` | `asia-south1` | GCP region |
| `GOOGLE_APPLICATION_CREDENTIALS` | `` | Path to service account JSON |
| `VERTEX_PLANNER_MODEL` | `gemini-2.5-flash` | Gemini model for `/run/start` and `/planner/debug` |
| `VERTEX_LIVE_MODEL` | `gemini-2.0-flash-live-preview-04-09` | Gemini Live/LiveKit model |
| `LIVEKIT_URL` | `` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | `` | LiveKit API key |
| `LIVEKIT_API_SECRET` | `` | LiveKit API secret |
| `ENABLE_LIVEKIT_AGENT` | `false` | Start LiveKit worker in background with API process |
| `MAX_STEPS_PER_RUN` | `50` | Max steps before timeout |
| `ARTIFACTS_BASE_DIR` | `./artifacts` | Where to store run artifacts |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `DAB_MQTT_BROKER` | `localhost` | MQTT broker host (real-device mode) |
| `DAB_MQTT_PORT` | `1883` | MQTT broker port |
| `DAB_DEVICE_ID` | `mock-device` | DAB device identifier |
| `DAB_REQUEST_TIMEOUT` | `10.0` | DAB response timeout (seconds) |
| `IMAGE_SOURCE` | `auto` | Capture source: `auto`, `hdmi-capture`, `camera-capture`, or `dab` |
| `ENABLE_HDMI_CAPTURE` | `true` | Enable HDMI/capture-card video source probing and use |
| `ENABLE_CAMERA_CAPTURE` | `true` | Enable camera/webcam video source probing and use |
| `HDMI_CAPTURE_DEVICE` | `` | V4L2 path (e.g. `/dev/video2`); auto-detected when empty |
| `HDMI_CAPTURE_WIDTH` | `1920` | Requested HDMI capture width |
| `HDMI_CAPTURE_HEIGHT` | `1080` | Requested HDMI capture height |
| `HDMI_CAPTURE_FPS` | `30.0` | Requested HDMI capture FPS |
| `HDMI_CAPTURE_FOURCC` | `MJPG` | FOURCC codec (`MJPG` or `YUYV`) |
| `HDMI_STREAM_JPEG_QUALITY` | `80` | JPEG quality for browser live stream |
| `HDMI_AUDIO_ENABLED` | `false` | Enable HDMI audio streaming endpoint |
| `HDMI_AUDIO_INPUT_FORMAT` | `auto` | Audio input format: `auto`, `alsa`, or `pulse` |
| `HDMI_AUDIO_DEVICE` | `` | ALSA input device (e.g. `hw:1,0`); auto-select when empty |
| `HDMI_AUDIO_SAMPLE_RATE` | `48000` | Audio sample rate |
| `HDMI_AUDIO_CHANNELS` | `2` | Audio channels |
| `HDMI_AUDIO_BITRATE` | `128k` | MP3 bitrate for `/stream/audio` |
| `HDMI_AUDIO_CHUNK_BYTES` | `4096` | Chunk size for audio stream generator |
| `YOUTUBE_APP_ID` | `youtube` | Only allowed `app_id` for `LAUNCH_APP` |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI (static/index.html) |
| `GET` | `/health` | Health check |
| `GET` | `/config` | Configuration summary (non-sensitive) |
| `GET` | `/capture/source` | Capture source diagnostics (HDMI availability, active mode) |
| `GET` | `/capture/devices` | List `/dev/video*` devices with kind/readability diagnostics |
| `POST` | `/capture/select` | Select capture source and preferred/explicit video device |
| `GET` | `/audio/source` | HDMI audio diagnostics (ALSA devices, ffmpeg availability) |
| `POST` | `/run/start` | Start a new automation run |
| `GET` | `/runs` | List all runs (most-recent first) |
| `GET` | `/run/{id}/status` | Get run status |
| `GET` | `/run/{id}/history` | Get full action history |
| `GET` | `/run/{id}/ai-transcript` | Get full AI planner transcript |
| `GET` | `/run/{id}/screenshot` | Get latest screenshot (base64 PNG) |
| `POST` | `/run/{id}/stop` | Stop a running run |
| `POST` | `/action` | Execute a manual action |
| `POST` | `/actions/batch` | Execute multiple manual actions in one request |
| `POST` | `/task/macro` | Convert natural-language task to actions (optional execute) |
| `GET` | `/dab/operations` | List supported DAB operations from device |
| `GET` | `/dab/keys` | List supported DAB key codes from device |
| `GET` | `/dab/apps` | List launchable applications from device |
| `POST` | `/screenshot` | Capture a screenshot now |
| `GET` | `/stream/hdmi` | Live MJPEG stream from HDMI capture card |
| `GET` | `/stream/audio` | Live MP3 audio stream from HDMI/ALSA input |
| `POST` | `/planner/debug` | Debug the planner |

### Example: Start a run

```bash
curl -X POST http://localhost:8000/run/start \
  -H "Content-Type: application/json" \
  -d '{"goal": "Open YouTube and verify the home screen", "app_id": "youtube", "content": "lofi music", "max_steps": 20}'
```

### Example: Manual key press

```bash
curl -X POST http://localhost:8000/action \
  -H "Content-Type: application/json" \
  -d '{"action": "PRESS_OK"}'
```

### Example: Planner debug

```bash
curl -X POST http://localhost:8000/planner/debug \
  -H "Content-Type: application/json" \
  -d '{"goal": "Open Settings", "current_app": "launcher", "ocr_text": "Home"}'
```

## Architecture Notes

### LiveKit integration in API runtime

The FastAPI service can now launch the LiveKit agent in-process at startup.

- Set `ENABLE_LIVEKIT_AGENT=true`
- Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`
- Start API normally (`python -m vertex_live_dab_agent`)

When enabled, a background LiveKit task starts with the API and is cancelled cleanly on shutdown.

### Mock mode (default)

`MockDABClient` is used when `DAB_MOCK_MODE=true` (the default).  All DAB
operations succeed with 50 ms simulated latency.  Screenshots return a 1×1
white PNG placeholder.  This is sufficient for developing and testing the full
orchestration loop end-to-end.

### Real DAB transport (MQTT)

Set `DAB_MOCK_MODE=false` to switch to `MQTTDABClient`.  The stub is in
`vertex_live_dab_agent/dab/client.py` — follow the embedded extension guide to
wire up `aiomqtt` or `paho-mqtt`.  The `format_topic()` helper in
`dab/topics.py` resolves DAB topic templates with the configured device ID.

### Planner

- **No Vertex AI** → simple heuristic: capture screenshot → get state → NEED_BETTER_VIEW → FAILED
- **With Vertex AI** → sends structured prompt + context to Gemini; parses JSON action response

## Hybrid Planning Additions

The agent now has a local hybrid-planning foundation intended for guided YTS and device-settings work:

- `device_profiles/` stores one JSON capability profile per device.
- `experience/trajectories.jsonl` stores local step outcomes for retrieval.
- The orchestrator derives a `hybrid_policy_mode` per run:
  - `DIRECT_DAB_PREFERRED`
  - `LOCAL_MEMORY_ASSISTED`
  - `HYBRID_DIRECT_FIRST`
  - `UI_NAVIGATION_HEAVY`

The current implementation does not self-train a local model yet. It first builds the safer prerequisite layer:

- persistent device capability memory
- local trajectory retrieval for similar goals
- policy hints injected into planner execution state

This gives you the data pipeline needed before distilling a local navigation model.

## Local Training And Detection

The repo now includes a practical local-ML scaffold rather than a fake end-state:

- `vertex_live_dab_agent/hybrid/local_vision.py`
  - extracts local screenshot features from PNG metadata and OCR text
  - detects coarse screen labels such as `settings`, `timezone`, `language`, `video_player`, `network`
- `vertex_live_dab_agent/hybrid/dataset.py`
  - defines the normalized per-step training example schema
- `vertex_live_dab_agent/hybrid/local_ranker.py`
  - loads a lightweight distilled JSON model and ranks likely next actions
- `scripts/train_local_ranker.py`
  - trains that lightweight model offline from `trajectories.jsonl`

### Training Flow

1. Run guided or autonomous tests so the orchestrator keeps appending local step outcomes to `experience/trajectories.jsonl`.
2. Train the local ranker offline:

```bash
python3 scripts/train_local_ranker.py artifacts/experience/trajectories.jsonl artifacts/models/local_action_ranker.json
```

3. Restart the service or point `LOCAL_RANKER_MODEL_PATH` at the generated model.
4. New runs will expose:
   - `observation_features`
   - `local_action_suggestions`
   - `local_model_version`

### Important Limits

- This local ranker is not a deep vision model.
- It is a safer first stage that learns action priors from your own trajectories.
- If you later want a true local multimodal model, this scaffold gives you the dataset and interfaces needed to replace the ranker without rewriting the orchestrator.

### OCR (optional)

```bash
pip install -e ".[ocr]"
sudo apt install tesseract-ocr   # or: brew install tesseract
```

OCR is silently disabled if `pytesseract` is not installed.

### HDMI capture card mode (optional)

```bash
pip install -e ".[dev,hdmi]"

# Auto-select first working /dev/video* and prefer HDMI for screenshots
export IMAGE_SOURCE=auto

# Or force a specific device
export IMAGE_SOURCE=hdmi-capture
export HDMI_CAPTURE_DEVICE=/dev/video2
```

When HDMI is active, both AI step screenshots and manual `/screenshot` requests
use the HDMI frame source, and the UI can display the live stream from
`/stream/hdmi`.

To enable HDMI audio streaming in the UI:

```bash
export HDMI_AUDIO_ENABLED=true
export HDMI_AUDIO_DEVICE=hw:1,0   # optional; auto-select if omitted
# If ffmpeg says "Unknown input format: alsa", use pulse mode:
export HDMI_AUDIO_INPUT_FORMAT=pulse
export HDMI_AUDIO_DEVICE=default
```

The UI audio player uses `/stream/audio`.

Linux permission note:

If `sudo arecord ...` works but normal user capture fails, add your user to the
`audio` group and re-login:

```bash
sudo usermod -aG audio $USER
newgrp audio
```

For local desktop preview/testing of the capture card:

```bash
python capture_view.py --list
python capture_view.py --device /dev/video2

# Audio capture test
python capture_audio.py --list
python capture_audio.py --device hw:1,0 --seconds 10 --out hdmi_audio.wav
python capture_audio.py --format pulse --device default --seconds 10 --out hdmi_audio.wav
```

### Artifacts

Each run creates a directory under `./artifacts/<run_id>/`:

```
metadata.json          – run metadata
actions.jsonl          – one JSON record per step
screenshots/
    step_0000.png      – screenshot at each step
planner_traces/
    step_0000.json     – planner inputs + output per step
final_summary.json     – final status + stats
```

### LiveKit agent (stub)

```bash
python -m vertex_live_dab_agent.livekit_agent.agent
```

Runs in stub mode until the `livekit-agents` SDK is wired in
(`vertex_live_dab_agent/livekit_agent/agent.py` contains a commented-out
reference implementation).

## What Requires Real Infrastructure

The following items are intentionally stubbed and require user-specific
infrastructure to complete:

| Item | Location | What's needed |
|---|---|---|
| Real DAB transport | `dab/client.py` — `MQTTDABClient` | MQTT broker + Android TV device running DAB |
| Vertex AI planning | `planner/planner.py` — `_plan_with_vertex` | `GOOGLE_APPLICATION_CREDENTIALS` + Vertex AI API enabled |
| Semantic validation | `capture/validator.py` — `validate_semantic` | Same Vertex AI credentials |
| LiveKit agent | `livekit_agent/agent.py` | `LIVEKIT_URL/API_KEY/API_SECRET` + `livekit-agents` package |
| Persistent run storage | `api/api.py` — `_runs` dict | Replace with Redis / database for multi-process deployments |

## Development

```bash
# Run tests with short tracebacks
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_api.py -v

# Run with debug logging
LOG_LEVEL=DEBUG python -m vertex_live_dab_agent
```

### Vertex AI live smoke tests (real API)

```bash
# Install vertex extras if not already installed
pip install -e ".[dev,vertex]"

# Required auth/env
export GOOGLE_CLOUD_PROJECT="<your-project>"
export GOOGLE_CLOUD_LOCATION="asia-south1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Optional model override
export VERTEX_TEST_MODEL="gemini-2.5-flash"

# Run real Vertex tests
RUN_VERTEX_INTEGRATION_TESTS=1 pytest tests/test_vertex_integration.py -v -s
```

## Frontend on GCP, harness local (recommended)

If you want **only the webpage** on GCP and keep all automation/harness logic
on your local machine, use this pattern:

1. Deploy only `static/index.html` to GCP static hosting
2. Run harness API locally (`python -m vertex_live_dab_agent`)
3. Expose local API through a secure tunnel
4. Set **Harness API** in the webpage footer to the tunnel URL

### 1) Deploy frontend only to GCS static website

Use the included script:

```bash
./scripts/deploy_frontend_gcs.sh YOUR_GCP_PROJECT YOUR_BUCKET_NAME [API_BASE_URL]
```

Example:

```bash
# Auto-detects Cloud Run service URL for dab-live-api
./scripts/deploy_frontend_gcs.sh my-project dab-remote-ui-prod

# Or provide a fixed static API URL explicitly
./scripts/deploy_frontend_gcs.sh my-project dab-remote-ui-prod https://dab-live-api-xxxxx-uc.a.run.app
```

This deploys only [static/index.html](static/index.html) and [static/config.js](static/config.js).
`config.js` stores a static API base URL, so no ngrok is required when API is
publicly reachable (for example Cloud Run).

### 2) Start harness locally (optional)

```bash
export IMAGE_SOURCE=auto
export HDMI_CAPTURE_DEVICE=/dev/video0   # or /dev/video1
python -m vertex_live_dab_agent
```

Local harness API runs at `http://localhost:8000`.

### 3) Expose local harness with a tunnel (only if API is local)

Using Cloudflare Tunnel (recommended):

```bash
# one-time login:
cloudflared tunnel login

# quick temporary public URL -> localhost:8000
cloudflared tunnel --url http://localhost:8000
```

Or using ngrok:

```bash
ngrok http 8000
```

### 4) Connect GCP UI to local harness

In the webpage footer, set **Harness API** to your tunnel URL, for example:

```text
https://abc123.trycloudflare.com
```

Then click **Save**.

Tip: you can also open the UI with query param:

```text
http://<bucket>.storage.googleapis.com/index.html?api=https://abc123.trycloudflare.com
```
