"""FastAPI backend for vertex_live_dab_agent."""
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from vertex_live_dab_agent.api.models import (
    ActionHistoryResponse,
    ActionRecordItem,
    ConfigSummaryResponse,
    HealthResponse,
    ManualActionRequest,
    ManualActionResponse,
    PlannerDebugRequest,
    PlannerDebugResponse,
    RunStatusResponse,
    RunSummaryItem,
    StartRunRequest,
    StartRunResponse,
)
from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.client import DABClientBase, create_dab_client
from vertex_live_dab_agent.dab.topics import KEY_MAP
from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState, RunStatus
from vertex_live_dab_agent.planner.planner import Planner

logger = logging.getLogger(__name__)

# Path to the bundled static frontend
_STATIC_DIR = Path(__file__).parent.parent.parent / "static"

app = FastAPI(
    title="Vertex Live DAB Agent",
    description="AI-driven Android TV testing tool using Vertex AI + LiveKit + DAB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared singleton clients (lazy-initialised)
# ---------------------------------------------------------------------------
_runs: Dict[str, RunState] = {}
_run_tasks: Dict[str, asyncio.Task] = {}
_dab_client: Optional[DABClientBase] = None
_planner: Optional[Planner] = None


def get_dab_client() -> DABClientBase:
    """Return (or lazily create) the singleton DAB client."""
    global _dab_client
    if _dab_client is None:
        _dab_client = create_dab_client()
    return _dab_client


def get_planner() -> Planner:
    """Return (or lazily create) the singleton Planner."""
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the bundled browser demo."""
    html_path = _STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Ensure static/index.html exists in the repo root.",
        )
    return FileResponse(str(html_path))


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    config = get_config()
    return HealthResponse(status="ok", mock_mode=config.dab_mock_mode)


@app.get("/config", response_model=ConfigSummaryResponse)
async def config_summary() -> ConfigSummaryResponse:
    """Configuration summary (non-sensitive values only)."""
    c = get_config()
    return ConfigSummaryResponse(
        google_cloud_project=c.google_cloud_project or "(not set)",
        google_cloud_location=c.google_cloud_location,
        vertex_live_model=c.vertex_live_model,
        dab_mock_mode=c.dab_mock_mode,
        dab_device_id=c.dab_device_id,
        max_steps_per_run=c.max_steps_per_run,
        artifacts_base_dir=c.artifacts_base_dir,
        log_level=c.log_level,
    )


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@app.post("/run/start", response_model=StartRunResponse)
async def start_run(request: StartRunRequest) -> StartRunResponse:
    """Start a new automation run."""
    run_id = str(uuid.uuid4())
    state = RunState(run_id=run_id, goal=request.goal)
    if request.app_id:
        state.current_app = request.app_id
    _runs[run_id] = state

    orchestrator = Orchestrator(
        dab_client=get_dab_client(),
        planner=get_planner(),
        max_steps=request.max_steps,  # None → uses config default
    )
    task = asyncio.create_task(orchestrator.run(state))
    _run_tasks[run_id] = task
    logger.info("Run started via API: run_id=%s goal=%r", run_id, request.goal)
    return StartRunResponse(run_id=run_id, status=state.status.value, goal=request.goal)


@app.get("/runs", response_model=List[RunSummaryItem])
async def list_runs() -> List[RunSummaryItem]:
    """List all runs (most-recent first)."""
    return [
        RunSummaryItem(
            run_id=s.run_id,
            goal=s.goal,
            status=s.status.value,
            step_count=s.step_count,
            started_at=s.started_at,
        )
        for s in reversed(list(_runs.values()))
    ]


@app.get("/run/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """Get status of a run."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return RunStatusResponse(
        run_id=state.run_id,
        status=state.status.value,
        goal=state.goal,
        step_count=state.step_count,
        retries=state.retries,
        current_app=state.current_app,
        current_screen=state.current_screen,
        last_actions=state.last_actions,
        started_at=state.started_at,
        finished_at=state.finished_at,
        error=state.error,
        has_screenshot=state.latest_screenshot_b64 is not None,
        artifacts_dir=state.artifacts_dir,
    )


@app.get("/run/{run_id}/history", response_model=ActionHistoryResponse)
async def get_run_history(run_id: str) -> ActionHistoryResponse:
    """Get full action history for a run."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return ActionHistoryResponse(
        run_id=state.run_id,
        goal=state.goal,
        action_count=len(state.action_history),
        actions=[ActionRecordItem(**r.model_dump()) for r in state.action_history],
    )


@app.get("/run/{run_id}/screenshot")
async def get_screenshot(run_id: str) -> JSONResponse:
    """Get latest screenshot for a run (base64 PNG)."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    if not state.latest_screenshot_b64:
        raise HTTPException(status_code=404, detail="No screenshot available yet")
    return JSONResponse({"run_id": run_id, "image_b64": state.latest_screenshot_b64})


@app.post("/run/{run_id}/stop")
async def stop_run(run_id: str) -> dict:
    """Stop a running run."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    task = _run_tasks.get(run_id)
    if task and not task.done():
        task.cancel()
        state.finish(RunStatus.STOPPED)
        logger.info("Run stopped via API: run_id=%s", run_id)
    return {"run_id": run_id, "status": state.status.value}


# ---------------------------------------------------------------------------
# Device / manual control endpoints
# ---------------------------------------------------------------------------

@app.post("/screenshot", response_model=dict)
async def capture_screenshot() -> dict:
    """Capture a screenshot from the device right now."""
    try:
        resp = await get_dab_client().capture_screenshot()
        return {"success": resp.success, "image_b64": resp.data.get("image")}
    except Exception as exc:
        logger.error("Screenshot capture failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/action", response_model=ManualActionResponse)
async def manual_action(request: ManualActionRequest) -> ManualActionResponse:
    """Execute a manual action directly against the DAB client."""
    try:
        dab = get_dab_client()
        action = request.action.upper()
        params = request.params or {}
        if action in KEY_MAP:
            resp = await dab.key_press(KEY_MAP[action])
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "LAUNCH_APP":
            app_id = params.get("app_id", "")
            if not app_id:
                raise HTTPException(status_code=400, detail="app_id is required for LAUNCH_APP")
            resp = await dab.launch_app(app_id)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "GET_STATE":
            app_id = params.get("app_id", "")
            resp = await dab.get_app_state(app_id)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action!r}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Manual action failed: action=%s error=%s", request.action, exc)
        return ManualActionResponse(success=False, action=request.action, error=str(exc))


# ---------------------------------------------------------------------------
# Planner debug
# ---------------------------------------------------------------------------

@app.post("/planner/debug", response_model=PlannerDebugResponse)
async def planner_debug(request: PlannerDebugRequest) -> PlannerDebugResponse:
    """Run the planner with the given inputs and return the planned action."""
    planned = await get_planner().plan(
        goal=request.goal,
        ocr_text=request.ocr_text,
        current_app=request.current_app,
        current_screen=request.current_screen,
        last_actions=request.last_actions or [],
    )
    return PlannerDebugResponse(
        action=planned.action,
        confidence=planned.confidence,
        reason=planned.reason,
        params=planned.params,
    )
