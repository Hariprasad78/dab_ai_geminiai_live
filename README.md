# Vertex Live DAB Agent

AI-driven Android TV / Google TV automated testing tool using **Vertex AI (Gemini Live)**, **LiveKit**, and the **DAB (Device Automation Bus)** protocol.

## Overview

The agent runs an **observe → plan → act → verify** loop:
1. **Observe** – Captures a screenshot via DAB and optionally runs OCR
2. **Plan** – Uses Vertex AI (Gemini) or a heuristic planner to decide the next action
3. **Act** – Sends the action to the TV device via DAB (key press, app launch, etc.)
4. **Verify** – Validates the outcome semantically or deterministically

## Project Structure

```
vertex_live_dab_agent/
├── config.py           # Centralized env-var config
├── dab/
│   ├── client.py       # DAB client (Mock + MQTT stub)
│   └── topics.py       # DAB topic names and key codes
├── planner/
│   ├── planner.py      # Planning layer (Vertex AI + heuristic)
│   └── schemas.py      # Action schemas (Pydantic)
├── capture/
│   ├── capture.py      # Screenshot capture + OCR (optional)
│   └── validator.py    # Semantic + deterministic validation
├── session/
│   └── manager.py      # Session manager
├── orchestrator/
│   ├── orchestrator.py # Main observe→plan→act→verify loop
│   └── run_state.py    # RunState Pydantic model
├── api/
│   ├── api.py          # FastAPI backend
│   └── models.py       # Request/response models
├── artifacts/
│   └── logger.py       # Structured logging + artifact storage
└── livekit_agent/
    └── agent.py        # LiveKit agent entrypoint (stub)
static/
└── index.html          # Web UI demo
tests/
├── test_schemas.py
├── test_planner.py
├── test_dab_client.py
├── test_run_state.py
└── test_api.py
```

## Prerequisites

- Python 3.11+
- (Optional) Google Cloud project with Vertex AI enabled
- (Optional) LiveKit server
- (Optional) DAB-compatible MQTT broker + Android TV device

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd dab_ai_geminiai_live

# Install with dev dependencies
pip install -e ".[dev]"

# Install with all optional extras
pip install -e ".[dev,vertex,ocr]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Key environment variables:

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
| `API_PORT` | `8000` | FastAPI server port |

## Running

### API Server

```bash
uvicorn vertex_live_dab_agent.api.api:app --reload --host 0.0.0.0 --port 8000
```

Open the web UI at: **http://localhost:8000** (serves `static/index.html` via browser)

Or browse the interactive API docs at: **http://localhost:8000/docs**

### LiveKit Agent (stub)

```bash
python -m vertex_live_dab_agent.livekit_agent.agent
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/config` | Configuration summary |
| `POST` | `/run/start` | Start a new automation run |
| `GET` | `/run/{id}/status` | Get run status |
| `GET` | `/run/{id}/screenshot` | Get latest screenshot |
| `POST` | `/run/{id}/stop` | Stop a run |
| `GET` | `/runs` | List all runs |
| `POST` | `/action` | Execute a manual action |
| `POST` | `/screenshot` | Capture a screenshot |
| `POST` | `/planner/debug` | Debug the planner |

### Example: Start a run

```bash
curl -X POST http://localhost:8000/run/start \
  -H "Content-Type: application/json" \
  -d '{"goal": "Launch Netflix and verify the home screen"}'
```

### Example: Manual key press

```bash
curl -X POST http://localhost:8000/action \
  -H "Content-Type: application/json" \
  -d '{"action": "PRESS_OK"}'
```

## Running Tests

```bash
pytest tests/ -v
```

## Architecture Notes

### DAB Client

- **`MockDABClient`** – Used by default (`DAB_MOCK_MODE=true`). Returns simulated responses with minimal latency. Ideal for development and CI.
- **`MQTTDABClient`** – Stub adapter for a real MQTT-based DAB broker. Implement `aiomqtt` or `paho-mqtt` integration to connect to real devices.

### Planner

- Without a Vertex AI client: uses a simple heuristic (screenshot → get_state → need_better_view → failed)
- With a Vertex AI client: sends the current context and screenshot to Gemini, parses the structured JSON response

### OCR

Install optional OCR support:
```bash
pip install -e ".[ocr]"
sudo apt install tesseract-ocr  # or brew install tesseract on macOS
```

### Artifacts

Each run creates a directory under `./artifacts/<run_id>/` containing:
- `metadata.json` – Run metadata
- `actions.jsonl` – Action log (newline-delimited JSON)
- `screenshots/step_NNNN.png` – Per-step screenshots
- `planner_traces/step_NNNN.json` – Per-step planner inputs/outputs
- `final_summary.json` – Final run summary

## Development

```bash
# Run tests with coverage
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_api.py -v
```
