# Vertex Live DAB Agent

AI-driven Android TV / Google TV automated testing tool using **Vertex AI (Gemini Live)**, **LiveKit**, and the **DAB (Device Automation Bus)** protocol.

## Overview

The agent runs an **observe → plan → act → verify** loop:
1. **Observe** – Captures a screenshot via DAB and optionally runs OCR
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
| `VERTEX_LIVE_MODEL` | `gemini-2.0-flash-live-preview-04-09` | Gemini model |
| `LIVEKIT_URL` | `` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | `` | LiveKit API key |
| `LIVEKIT_API_SECRET` | `` | LiveKit API secret |
| `MAX_STEPS_PER_RUN` | `50` | Max steps before timeout |
| `ARTIFACTS_BASE_DIR` | `./artifacts` | Where to store run artifacts |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `DAB_MQTT_BROKER` | `localhost` | MQTT broker host (real-device mode) |
| `DAB_MQTT_PORT` | `1883` | MQTT broker port |
| `DAB_DEVICE_ID` | `mock-device` | DAB device identifier |
| `DAB_REQUEST_TIMEOUT` | `10.0` | DAB response timeout (seconds) |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI (static/index.html) |
| `GET` | `/health` | Health check |
| `GET` | `/config` | Configuration summary (non-sensitive) |
| `POST` | `/run/start` | Start a new automation run |
| `GET` | `/runs` | List all runs (most-recent first) |
| `GET` | `/run/{id}/status` | Get run status |
| `GET` | `/run/{id}/history` | Get full action history |
| `GET` | `/run/{id}/screenshot` | Get latest screenshot (base64 PNG) |
| `POST` | `/run/{id}/stop` | Stop a running run |
| `POST` | `/action` | Execute a manual action |
| `POST` | `/screenshot` | Capture a screenshot now |
| `POST` | `/planner/debug` | Debug the planner |

### Example: Start a run

```bash
curl -X POST http://localhost:8000/run/start \
  -H "Content-Type: application/json" \
  -d '{"goal": "Launch Netflix and verify the home screen", "max_steps": 20}'
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

### OCR (optional)

```bash
pip install -e ".[ocr]"
sudo apt install tesseract-ocr   # or: brew install tesseract
```

OCR is silently disabled if `pytesseract` is not installed.

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

