"""FastAPI backend for vertex_live_dab_agent."""
import asyncio
from contextlib import asynccontextmanager
import grp
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from vertex_live_dab_agent.api.models import (
    AITranscriptResponse,
        DABTranscriptResponse,
    ActionHistoryResponse,
    ActionRecordItem,
    CaptureSourceResponse,
    ConfigSummaryResponse,
    HealthResponse,
    FinalDiagnosis,
    FriendlyRunExplanationResponse,
    FriendlyStepItem,
    NarrationEventItem,
    NarrationResponse,
    ManualActionBatchRequest,
    ManualActionBatchResponse,
    ManualActionRequest,
    ManualActionResponse,
    PlannerDebugRequest,
    PlannerDebugResponse,
    RunStatusResponse,
    RunSummaryItem,
    StartRunRequest,
    StartRunResponse,
    TTSSpeakRequest,
    TTSSpeakResponse,
    TaskMacroRequest,
    TaskMacroResponse,
)
from vertex_live_dab_agent.api.tts_service import GoogleTTSService
from vertex_live_dab_agent.capture.capture import ScreenCapture
from vertex_live_dab_agent.capture.hdmi_audio import (
    HdmiAudioStreamSession,
    arecord_available,
    ffmpeg_available,
    ffmpeg_has_input_format,
    list_alsa_capture_devices,
    resolve_audio_input,
)
from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.client import DABClientBase, create_dab_client
from vertex_live_dab_agent.dab.topics import KEY_MAP
from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState, RunStatus
from vertex_live_dab_agent.planner.planner import Planner
from vertex_live_dab_agent.planner.vertex_client import VertexPlannerClient

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
# NOTE: These in-process singletons work correctly with a single uvicorn worker
# (the default).  For multi-process deployments (multiple uvicorn workers or
# gunicorn) use a shared backend (e.g. Redis, PostgreSQL) to persist run state
# and replace this in-memory dict with a store that supports concurrent access.
# ---------------------------------------------------------------------------
_runs: Dict[str, RunState] = {}
_run_tasks: Dict[str, asyncio.Task] = {}
_dab_client: Optional[DABClientBase] = None
_planner: Optional[Planner] = None
_screen_capture: Optional[ScreenCapture] = None
_livekit_task: Optional[asyncio.Task] = None
_tts_service: Optional[GoogleTTSService] = None

_APP_ID_ALIASES: Dict[str, str] = {
    "com.netflix.ninja": "netflix",
    "netflix": "netflix",
    "com.google.android.youtube": "youtube",
    "com.google.android.youtube.tv": "youtube",
    "youtube": "youtube",
    "com.android.settings": "settings",
    "com.android.tv.settings": "settings",
    "settings": "settings",
}


def _to_simple_action(action: str, params: Optional[dict] = None) -> str:
    action_u = str(action or "").upper()
    p = params or {}
    if action_u == "LAUNCH_APP":
        name = p.get("app_id") or p.get("app_name") or "the app"
        return f"Trying to open {name}"
    if action_u == "WAIT":
        return "Waiting briefly so the app or screen can load"
    if action_u == "GET_STATE":
        return "Checking whether the app is really open"
    if action_u == "CAPTURE_SCREENSHOT":
        return "Taking a screenshot to understand the current screen"
    if action_u == "NEED_PLAYER_CONTROLS_VISIBLE":
        return "The tool needs to clearly see the YouTube player controls"
    if action_u == "NEED_VIDEO_PLAYBACK_CONFIRMED":
        return "The tool needs to confirm a YouTube video is actually playing"
    if action_u == "NEED_SETTINGS_GEAR_LOCATION":
        return "The tool needs to find the settings gear icon in the player controls"
    if action_u == "NEED_PLAYER_MENU_CONFIRMATION":
        return "The tool needs to confirm the YouTube player settings menu is open"
    if action_u == "NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION":
        return "The tool needs to confirm the Stats for Nerds toggle"
    if action_u == "PRESS_BACK":
        return "Pressing the Back button"
    if action_u == "PRESS_HOME":
        return "Pressing the Home button"
    if action_u == "PRESS_OK":
        ok_intent = str((p or {}).get("ok_intent", "")).strip()
        if ok_intent:
            return f"Pressing OK for {ok_intent.lower().replace('_', ' ')}"
        return "Pressing OK (can toggle playback on video screens)"
    if action_u == "OPEN_CONTENT":
        return "Trying to open the requested content"
    if action_u == "SET_SETTING":
        return "Trying to change a system setting"
    if action_u == "GET_SETTING":
        return "Checking a system setting value"
    if action_u == "FAILED":
        return "The test has stopped because recovery did not work"
    return f"Running action {action_u or 'UNKNOWN'}"


def _to_step_title(action: str, params: Optional[dict] = None) -> str:
    action_u = str(action or "").upper()
    p = params or {}
    if action_u == "LAUNCH_APP":
        return f"Open {p.get('app_id') or p.get('app_name') or 'App'}"
    if action_u == "PRESS_BACK":
        return "Press Back"
    if action_u == "PRESS_HOME":
        return "Go to Home"
    if action_u == "GET_STATE":
        return "Verify App State"
    if action_u == "CAPTURE_SCREENSHOT":
        return "Capture Screen"
    if action_u == "WAIT":
        return "Wait"
    return action_u.replace("_", " ").title() or "Step"


def _friendly_timeline(state: RunState) -> List[FriendlyStepItem]:
    items: List[FriendlyStepItem] = []
    diagnosis_by_step: Dict[int, Dict[str, object]] = {}
    for ev in state.ai_transcript:
        if ev.get("type") == "stuck-diagnosis":
            try:
                diagnosis_by_step[int(ev.get("step", -1))] = ev
            except Exception:
                continue

    screen_seen_default = str(state.current_app_state or state.current_screen or "Unknown screen")
    for rec in state.action_history:
        result = str(rec.result or "").upper()
        simple_status = "PASS" if result == "PASS" else ("FAIL" if result == "FAIL" else "INFO")
        ev = diagnosis_by_step.get(int(rec.step), {})
        ctx = ev.get("context") if isinstance(ev.get("context"), dict) else {}
        screen_seen = str((ctx or {}).get("current_screenshot_summary") or screen_seen_default)
        recovery_tried = None
        if ev:
            recovery_tried = f"Recovery decision: {ev.get('decision', 'unknown')}"
        why_failed = None
        if simple_status == "FAIL":
            why_failed = str(rec.reason or "The action did not move to the expected screen.")
        next_hint = "The tool will continue with the next planned step."
        if ev:
            next_hint = f"Next, the tool will try: {ev.get('decision', 'recovery')}."
        items.append(
            FriendlyStepItem(
                step=int(rec.step),
                title=_to_step_title(rec.action, rec.params),
                simple_action=_to_simple_action(rec.action, rec.params),
                what_happened=(
                    f"The tool ran {str(rec.action).replace('_', ' ').lower()}"
                    + (f" with {rec.params}." if rec.params else ".")
                ),
                what_screen_was_seen=screen_seen,
                why_this_step=(rec.reason or "This step was needed to continue the test.").strip(),
                why_it_failed=why_failed,
                what_recovery_was_tried=recovery_tried,
                what_happens_next=next_hint,
                result=(
                    "Step completed successfully."
                    if result == "PASS"
                    else "Step did not work as expected."
                ),
                simple_status=simple_status,
            )
        )
    return items


def _build_final_diagnosis(state: RunState) -> FinalDiagnosis:
    status = state.status.value
    launch_attempted = any(str(a.action).upper() == "LAUNCH_APP" for a in state.action_history)
    parse_fail_events = [
        e for e in state.ai_transcript
        if str(e.get("intent", "")).startswith("parse_failure")
        or "parse_failure" in str(e.get("reason", ""))
    ]
    parse_fail_count = len(parse_fail_events)
    stuck_events = [e for e in state.ai_transcript if e.get("type") == "stuck-diagnosis"]
    ok_guard_events = [e for e in state.ai_transcript if e.get("type") == "commit-guard-blocked"]
    ok_effect_events = [e for e in state.ai_transcript if e.get("type") == "ok-effect-analysis"]
    repeated_ok_actions = sum(1 for a in state.action_history if str(a.action).upper() in {"PRESS_OK", "PRESS_ENTER"})

    background_seen = False
    if str(state.current_screen or "").upper() == "BACKGROUND":
        background_seen = True
    if str(getattr(state, "current_app_state", "") or "").upper() == "BACKGROUND":
        background_seen = True
    for ev in state.dab_transcript:
        if ev.get("op") == "applications/get-state":
            st = str(((ev.get("data") or {}).get("state", ""))).upper()
            if st == "BACKGROUND":
                background_seen = True
                break

    recovery_attempts = sum(
        1
        for a in state.action_history
        if str(a.action).upper() in {"GET_STATE", "CAPTURE_SCREENSHOT", "NEED_BETTER_VIEW", "PRESS_HOME", "PRESS_BACK"}
    )
    recovery_attempts += len(stuck_events)

    fail_markers: List[str] = []
    for a in state.action_history:
        if str(a.result).upper() == "FAIL":
            fail_markers.append(f"step {a.step}: {a.action}")
    for e in state.ai_transcript:
        if e.get("type") in {"launch-resolution-failed", "terminal-check-blocked"}:
            fail_markers.append(f"step {e.get('step', '?')}: {e.get('type')}")
        if str(e.get("intent", "")).startswith("parse_failure"):
            fail_markers.append(f"step {e.get('step', '?')}: {e.get('intent')}")

    what_failed_first = fail_markers[0] if fail_markers else None
    what_failed_last = fail_markers[-1] if fail_markers else None

    latest_stuck_ctx = {}
    if stuck_events:
        latest_ctx = stuck_events[-1].get("context")
        if isinstance(latest_ctx, dict):
            latest_stuck_ctx = latest_ctx
    seen_screen = str(
        latest_stuck_ctx.get("current_screenshot_summary")
        or state.latest_ocr_text
        or state.current_app_state
        or state.current_screen
        or "unknown screen"
    )

    if (
        "youtube" in (state.goal or "").lower()
        and "stats for nerds" in (state.goal or "").lower()
        and status == "TIMEOUT"
        and repeated_ok_actions >= 3
    ):
        short = "Repeated OK on YouTube playback screen"
        detailed = (
            "The tool was stuck repeating OK on the YouTube playback screen. "
            "That did not move focus to the settings gear, and the run timed out before opening player settings."
        )
        friendly = (
            "The tool kept pressing OK on the video screen. "
            "On this screen, OK was toggling playback instead of opening settings, so recovery switched to screenshot-based control navigation."
        )
        root = "Repeated commit action in playback context"
        technical = "Repeated PRESS_OK/PRESS_ENTER without phase advancement toward gear/menu target"
        summary = "The run timed out after repeated OK actions on playback screen without progressing to YouTube player settings."
        screen_based_reason = f"Latest observed screen: {seen_screen}."
        goal_based_reason = "Goal required reaching YouTube player settings and enabling Stats for Nerds, but repeated OK did not move focus to the gear icon."
        recovery_summary = "Guarded recovery switched to screenshot-based checks and directional focus correction, but timeout happened before the target menu was reached."
        evidence = [
            f"repeated_ok_actions={repeated_ok_actions}",
            f"ok_guard_events={len(ok_guard_events)}",
            f"ok_effect_events={len(ok_effect_events)}",
            f"stuck_diagnosis_events={len(stuck_events)}",
        ]
    elif launch_attempted and background_seen and parse_fail_count >= 2:
        short = "App launch could not be confirmed"
        detailed = (
            "The system tried to open the app, but it was not verified as active on screen. "
            "After that, the planner failed repeatedly while deciding the next step."
        )
        friendly = (
            "The app was started, but the tool could not confirm it fully opened. "
            "Then the system got stuck deciding the next action, so the test was stopped safely."
        )
        root = "Launch verification + planner parse failure loop"
        technical = "App state remained BACKGROUND and repeated planner parse failures blocked recovery"
        summary = "App launch was not clearly verified, and repeated planner parsing failures prevented recovery."
        screen_based_reason = f"Latest observed screen/state: {seen_screen}."
        goal_based_reason = f"Goal was '{state.goal}', but target app was not verified in foreground."
        recovery_summary = (
            "The tool retried state checks and screenshot-based recovery, but each attempt stayed away from the target screen."
        )
        evidence = [
            f"current_app={state.current_app or state.current_app_id}",
            f"current_state={state.current_app_state or state.current_screen}",
            f"parse_failures={parse_fail_count}",
            f"stuck_diagnosis_events={len(stuck_events)}",
        ]
    elif ("youtube" in (state.goal or "").lower() and "stats for nerds" in (state.goal or "").lower()
          and (status in {"FAILED", "ERROR", "TIMEOUT"} or parse_fail_count >= 1)):
        short = "Could not reach YouTube player settings"
        detailed = (
            "The tool could not stay on the exact YouTube playback-controls path needed to enable Stats for Nerds."
        )
        friendly = (
            "The tool could not clearly reach the video controls and settings gear in YouTube, so it could not enable Stats for Nerds."
        )
        root = "YouTube player-control navigation did not reach target menu"
        technical = "Playback/control-panel confirmation remained ambiguous during bounded recovery"
        summary = "The run stopped because it could not reliably reach YouTube player settings and enable Stats for Nerds."
        screen_based_reason = f"Latest observed screen: {seen_screen}."
        goal_based_reason = "Goal required enabling Stats for Nerds from YouTube player settings, but target UI was not confirmed."
        recovery_summary = "The tool tried focused YouTube recovery steps (playback/controls/gear checks) but could not confirm the target path."
        evidence = [
            f"goal={state.goal}",
            f"current_app={state.current_app or state.current_app_id}",
            f"stuck_diagnosis_events={len(stuck_events)}",
        ]
    elif parse_fail_count >= 2:
        short = "Planner could not build valid next steps"
        detailed = (
            "The planner kept returning invalid decisions, so the tool could not continue safely."
        )
        friendly = (
            "The system got stuck while deciding what to do next, so the run was stopped to avoid wrong actions."
        )
        root = "Planner parse failure loop"
        technical = "Repeated parse_failure / parse_failure_limit_reached events"
        summary = "The planner repeatedly failed to produce valid next actions."
        screen_based_reason = f"Latest observed screen/state: {seen_screen}."
        goal_based_reason = f"Goal was '{state.goal}', but the planner could not produce a grounded path to it."
        recovery_summary = "The tool performed recovery checks after failures, but decisions stayed invalid."
        evidence = [
            f"parse_failures={parse_fail_count}",
            f"recent_actions={state.last_actions[-5:]}",
        ]
    elif status in {"FAILED", "ERROR", "TIMEOUT"}:
        short = "Run stopped before completion"
        detailed = state.error or "The run could not finish after recovery attempts."
        friendly = "The test could not finish successfully, so it was stopped safely."
        root = "Recovery failed"
        technical = state.error or "Unknown failure"
        summary = "The run stopped after repeated problems."
        screen_based_reason = f"Latest observed screen/state: {seen_screen}."
        goal_based_reason = f"Goal was '{state.goal}', but the run could not reach the required result."
        recovery_summary = "The tool tried bounded recovery steps and then stopped safely."
        evidence = [
            f"status={status}",
            f"retries={state.retries}",
        ]
    else:
        short = "Run completed"
        detailed = "The run finished without critical failure."
        friendly = "The test completed successfully."
        root = "Completed"
        technical = "No terminal failures"
        summary = "The run finished successfully."
        screen_based_reason = f"Latest observed screen/state: {seen_screen}."
        goal_based_reason = f"Goal '{state.goal}' was completed."
        recovery_summary = "No major recovery was needed."
        evidence = ["status=DONE"]

    return FinalDiagnosis(
        status=status,
        final_summary=summary,
        root_cause=root,
        user_friendly_reason=friendly,
        technical_reason=technical,
        recovery_attempts=recovery_attempts,
        what_failed_first=what_failed_first,
        what_failed_last=what_failed_last,
        failure_reason_short=short,
        failure_reason_detailed=detailed,
        failure_reason_user_friendly=friendly,
        screen_based_reason=screen_based_reason,
        goal_based_reason=goal_based_reason,
        recovery_summary=recovery_summary,
        evidence_used=evidence,
    )


def _validate_app_id(app_id: str) -> str:
    """Validate and normalize app_id."""
    normalized = str(app_id or "").strip()
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="app_id must be a non-empty string",
        )
    return _APP_ID_ALIASES.get(normalized.lower(), normalized)


def _infer_launch_content_from_goal(goal: str) -> Optional[str]:
    """Infer desired content from short natural-language launch goals.

    Examples:
    - "open netflix" -> "netflix"
    - "open youtube lofi" -> "lofi"
    - "launch youtube" -> None
    """
    text = (goal or "").strip()
    if not text:
        return None

    m = re.search(r"\b(?:open|launch|start)\s+(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return None

    phrase = m.group(1).strip()
    if not phrase:
        return None

    lower_phrase = phrase.lower()
    direct_app_phrases = {
        "youtube",
        "netflix",
        "settings",
        "prime video",
        "disney",
    }
    if lower_phrase in direct_app_phrases:
        return None
    if lower_phrase.startswith("youtube "):
        phrase = phrase[len("youtube "):].strip()

    phrase = re.sub(r"\s+", " ", phrase).strip(" .,!?:;\"'")
    return phrase or None


_APP_ID_BY_NAME: Dict[str, str] = {
    "youtube": "youtube",
    "netflix": "netflix",
    "settings": "settings",
    "prime video": "com.amazon.amazonvideo.livingroom",
    "disney": "com.disney.disneyplus",
}


def _infer_app_id_from_text(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    for name, app_id in _APP_ID_BY_NAME.items():
        if name in t:
            return app_id
    return None


def _extract_repeat_count(text: str) -> int:
    m = re.search(r"\b(\d{1,2})\b", text or "")
    if not m:
        return 1
    count = int(m.group(1))
    return max(1, min(count, 20))


def _plan_task_macro_actions(instruction: str) -> List[ManualActionRequest]:
    actions: List[ManualActionRequest] = []
    chunks = [
        c.strip()
        for c in re.split(r"\s*(?:,|;|\band then\b|\bthen\b)\s*", instruction, flags=re.IGNORECASE)
        if c and c.strip()
    ]

    for chunk in chunks:
        lower = chunk.lower()

        # open / launch / start <app>
        if re.search(r"\b(open|launch|start)\b", lower):
            app_id = _infer_app_id_from_text(lower)
            if app_id:
                params: Dict[str, str] = {"app_id": app_id}
                if app_id == "youtube":
                    content = _infer_launch_content_from_goal(chunk)
                    if content:
                        params["content"] = content
                actions.append(ManualActionRequest(action="LAUNCH_APP", params=params))
                continue

        if re.search(r"\bwait\b", lower):
            m = re.search(r"(\d+(?:\.\d+)?)", lower)
            seconds = float(m.group(1)) if m else 1.0
            actions.append(ManualActionRequest(action="WAIT", params={"seconds": seconds}))
            continue

        key_map = [
            ("home", "PRESS_HOME"),
            ("back", "PRESS_BACK"),
            ("up", "PRESS_UP"),
            ("down", "PRESS_DOWN"),
            ("left", "PRESS_LEFT"),
            ("right", "PRESS_RIGHT"),
            ("ok", "PRESS_OK"),
            ("enter", "PRESS_OK"),
            ("select", "PRESS_OK"),
            ("menu", "PRESS_MENU"),
            ("play", "PRESS_PLAY"),
            ("pause", "PRESS_PAUSE"),
        ]
        matched = False
        for token, action_name in key_map:
            if re.search(rf"\b{re.escape(token)}\b", lower):
                repeat = _extract_repeat_count(lower)
                for _ in range(repeat):
                    actions.append(ManualActionRequest(action=action_name, params=None))
                matched = True
                break
        if matched:
            continue

    return actions


def _resolve_audio_input() -> tuple[Optional[str], Optional[str]]:
    """Resolve `(input_format, device)` for HDMI audio stream."""
    config = get_config()
    return resolve_audio_input(
        preferred_format=config.hdmi_audio_input_format,
        configured_device=config.hdmi_audio_device,
    )


def _resolve_vertex_project(explicit_project: str) -> str:
    """Resolve Vertex project from env first, then ADC project if available."""
    project = str(explicit_project or "").strip()
    if project:
        return project
    try:
        import google.auth

        _, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return str(adc_project or "").strip()
    except Exception:
        return ""


def _vertex_planner_requested() -> bool:
    """Return whether runtime should attempt Vertex planner.

    Behavior:
    - `ENABLE_VERTEX_PLANNER=false|0|no` => disabled
    - `ENABLE_VERTEX_PLANNER=true|1|yes` => enabled
    - unset/other => auto-enable when basic Vertex config is present
    """
    raw = os.environ.get("ENABLE_VERTEX_PLANNER")
    if raw is not None:
        value = raw.strip().lower()
        if value in {"0", "false", "no", "off"}:
            return False
        if value in {"1", "true", "yes", "on"}:
            return True

    config = get_config()
    project = _resolve_vertex_project(config.google_cloud_project)
    return bool(project and config.google_cloud_location and config.vertex_planner_model)


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
        c = get_config()
        vertex_client = None
        if _vertex_planner_requested():
            try:
                project = _resolve_vertex_project(c.google_cloud_project)
                vertex_client = VertexPlannerClient(
                    project=project,
                    location=c.google_cloud_location,
                    model=c.vertex_planner_model,
                )
                logger.info(
                    "Planner initialized with Vertex model=%s project=%s location=%s",
                    c.vertex_planner_model,
                    project,
                    c.google_cloud_location,
                )
            except Exception as exc:
                logger.warning("Vertex planner disabled, falling back to heuristic: %s", exc)
        else:
            logger.info("Vertex planner not requested/configured; using heuristic planner")
        _planner = Planner(vertex_client=vertex_client)
    return _planner


def get_screen_capture() -> ScreenCapture:
    """Return (or lazily create) the singleton capture service."""
    global _screen_capture
    if _screen_capture is None:
        _screen_capture = ScreenCapture(get_dab_client())
    return _screen_capture


def get_tts_service() -> GoogleTTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = GoogleTTSService()
    return _tts_service


async def _maybe_start_livekit_agent() -> None:
    """Start LiveKit agent loop in background when enabled by config."""
    global _livekit_task
    if _livekit_task is not None and not _livekit_task.done():
        return

    c = get_config()
    if not c.enable_livekit_agent:
        logger.info("LiveKit agent disabled (ENABLE_LIVEKIT_AGENT=false)")
        return

    try:
        from vertex_live_dab_agent.livekit_agent.agent import run_agent

        _livekit_task = asyncio.create_task(
            run_agent(skip_config_validation=False),
            name="livekit-agent",
        )
        logger.info("LiveKit agent background task started")
    except Exception as exc:
        logger.error("Failed to start LiveKit agent task: %s", exc, exc_info=True)


async def _stop_livekit_agent() -> None:
    """Stop background LiveKit agent task if running."""
    global _livekit_task
    task = _livekit_task
    _livekit_task = None
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("LiveKit agent task cancelled")
    except Exception as exc:
        logger.warning("LiveKit agent task ended with error during shutdown: %s", exc)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await _maybe_start_livekit_agent()
    try:
        yield
    finally:
        await _stop_livekit_agent()


app.router.lifespan_context = _lifespan


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
        vertex_planner_model=c.vertex_planner_model,
        vertex_live_model=c.vertex_live_model,
        enable_livekit_agent=c.enable_livekit_agent,
        dab_mock_mode=c.dab_mock_mode,
        image_source=c.image_source,
        youtube_app_id=c.youtube_app_id,
        dab_device_id=c.dab_device_id,
        max_steps_per_run=c.max_steps_per_run,
        artifacts_base_dir=c.artifacts_base_dir,
        log_level=c.log_level,
        tts_enabled=c.tts_enabled,
        tts_voice_provider=c.tts_voice_provider,
        tts_model=c.tts_model,
        tts_voice_name=c.tts_voice_name,
        tts_language_code=c.tts_language_code,
    )


@app.get("/capture/source", response_model=CaptureSourceResponse)
async def capture_source() -> CaptureSourceResponse:
    """Return capture source mode and HDMI availability diagnostics."""
    status = get_screen_capture().capture_source_status()
    return CaptureSourceResponse(**status)


@app.get("/audio/source", response_model=dict)
async def audio_source() -> dict:
    """Return HDMI audio stream diagnostics."""
    c = get_config()
    devices = list_alsa_capture_devices()
    input_format, device = _resolve_audio_input()
    try:
        audio_gid = grp.getgrnam("audio").gr_gid
        user_in_audio_group = audio_gid in os.getgroups()
    except Exception:
        user_in_audio_group = None

    return {
        "enabled": c.hdmi_audio_enabled,
        "ffmpeg_available": ffmpeg_available(),
        "arecord_available": arecord_available(),
        "user_in_audio_group": user_in_audio_group,
        "ffmpeg_alsa": ffmpeg_has_input_format("alsa"),
        "ffmpeg_pulse": ffmpeg_has_input_format("pulse"),
        "input_format": input_format,
        "device": device,
        "has_devices": len(devices) > 0,
        "devices": devices,
        "sample_rate": c.hdmi_audio_sample_rate,
        "channels": c.hdmi_audio_channels,
        "bitrate": c.hdmi_audio_bitrate,
    }


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@app.post("/run/start", response_model=StartRunResponse)
async def start_run(request: StartRunRequest) -> StartRunResponse:
    """Start a new automation run."""
    run_id = str(uuid.uuid4())
    state = RunState(run_id=run_id, goal=request.goal)
    if request.app_id:
        state.current_app = _validate_app_id(request.app_id)
    explicit_content = str(request.content or "").strip()
    inferred_content = _infer_launch_content_from_goal(request.goal)
    if explicit_content:
        state.launch_content = explicit_content
    elif inferred_content:
        state.launch_content = inferred_content
    _runs[run_id] = state

    orchestrator = Orchestrator(
        dab_client=get_dab_client(),
        planner=get_planner(),
        capture=get_screen_capture(),
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
        dab_log_count=len(state.dab_transcript),
        dab_logs_tail=state.dab_transcript[-25:],
        ai_log_count=len(state.ai_transcript),
        ai_logs_tail=state.ai_transcript[-25:],
        narration_count=len(state.narration_transcript),
        narration_tail=state.narration_transcript[-25:],
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


@app.get("/run/{run_id}/dab-transcript", response_model=DABTranscriptResponse)
async def get_run_dab_transcript(run_id: str) -> DABTranscriptResponse:
    """Get full DAB request/response transcript for a run."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return DABTranscriptResponse(
        run_id=state.run_id,
        goal=state.goal,
        count=len(state.dab_transcript),
        events=state.dab_transcript,
    )


@app.get("/run/{run_id}/ai-transcript", response_model=AITranscriptResponse)
async def get_run_ai_transcript(run_id: str) -> AITranscriptResponse:
    """Get full AI planning transcript for a run."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return AITranscriptResponse(
        run_id=state.run_id,
        goal=state.goal,
        count=len(state.ai_transcript),
        events=state.ai_transcript,
    )


@app.get("/run/{run_id}/explain", response_model=FriendlyRunExplanationResponse)
async def get_run_explain(run_id: str) -> FriendlyRunExplanationResponse:
    """Get student-friendly timeline and final diagnosis while preserving raw logs."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return FriendlyRunExplanationResponse(
        run_id=state.run_id,
        goal=state.goal,
        status=state.status.value,
        timeline=_friendly_timeline(state),
        diagnosis=_build_final_diagnosis(state),
    )


@app.get("/run/{run_id}/narration", response_model=NarrationResponse)
async def get_run_narration(run_id: str, since_idx: int = -1) -> NarrationResponse:
    """Get narration events for frontend voice/subtitle playback."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    events = []
    for i, e in enumerate(state.narration_transcript):
        if i <= since_idx:
            continue
        events.append(
            NarrationEventItem(
                idx=i,
                step=int(e.get("step", 0) or 0),
                tts_text=str(e.get("tts_text", "")),
                tts_priority=int(e.get("tts_priority", 20) or 20),
                tts_category=str(e.get("tts_category", "STEP_RESULT")),
                tts_should_play=bool(e.get("tts_should_play", True)),
                tts_interruptible=bool(e.get("tts_interruptible", True)),
            )
        )
    return NarrationResponse(
        run_id=state.run_id,
        goal=state.goal,
        count=len(state.narration_transcript),
        events=events,
    )


@app.post("/tts/speak", response_model=TTSSpeakResponse)
async def tts_speak(request: TTSSpeakRequest) -> TTSSpeakResponse:
    """Synthesize narration text to MP3 audio using Google Cloud TTS."""
    result = get_tts_service().synthesize_tts(request.text, use_ssml=request.use_ssml)
    return TTSSpeakResponse(**result)


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
        result = await get_screen_capture().capture()
        return {
            "success": result.image_b64 is not None,
            "image_b64": result.image_b64,
            "source": result.source,
            "ocr_text": result.ocr_text,
        }
    except Exception as exc:
        logger.error("Screenshot capture failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stream/hdmi")
async def stream_hdmi() -> StreamingResponse:
    """MJPEG stream from HDMI capture card for browser live preview."""
    config = get_config()
    capture = get_screen_capture()
    status = capture.capture_source_status()

    if not status.get("hdmi_configured"):
        raise HTTPException(
            status_code=400,
            detail="HDMI capture is disabled. Set IMAGE_SOURCE=auto or IMAGE_SOURCE=hdmi-capture.",
        )

    async def frame_generator():
        boundary = b"--frame\r\n"
        while True:
            frame = capture.get_hdmi_stream_frame_jpeg(quality=config.hdmi_stream_jpeg_quality)
            if frame is None:
                await asyncio.sleep(0.2)
                continue
            headers = (
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
            )
            yield boundary + headers + frame + b"\r\n"
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/audio")
async def stream_audio() -> StreamingResponse:
    """Live MP3 audio stream from HDMI capture card/ALSA input."""
    c = get_config()
    if not c.hdmi_audio_enabled:
        raise HTTPException(status_code=400, detail="HDMI audio streaming disabled. Set HDMI_AUDIO_ENABLED=true")
    if not ffmpeg_available() and not arecord_available():
        raise HTTPException(status_code=500, detail="ffmpeg not found on host")

    input_format, device = _resolve_audio_input()
    if not input_format or not device:
        raise HTTPException(status_code=404, detail="No supported audio input source (alsa/pulse) found")

    session = HdmiAudioStreamSession(
        device=device,
        input_format=input_format,
        sample_rate=c.hdmi_audio_sample_rate,
        channels=c.hdmi_audio_channels,
        bitrate=c.hdmi_audio_bitrate,
    )
    started = session.start()
    if not started:
        # If a fixed device is configured and failed, try one auto-resolve retry.
        if c.hdmi_audio_device:
            fallback_format, fallback_device = resolve_audio_input(
                preferred_format=c.hdmi_audio_input_format,
                configured_device="",
            )
            if fallback_format and fallback_device:
                session.close()
                session = HdmiAudioStreamSession(
                    device=fallback_device,
                    input_format=fallback_format,
                    sample_rate=c.hdmi_audio_sample_rate,
                    channels=c.hdmi_audio_channels,
                    bitrate=c.hdmi_audio_bitrate,
                )
                started = session.start()
                if started:
                    input_format = fallback_format
                    device = fallback_device

        if not started:
            detail = session.last_error or f"Unable to start HDMI audio stream for device {device}"
            raise HTTPException(status_code=500, detail=detail)

    async def audio_generator():
        try:
            while True:
                chunk = await asyncio.to_thread(session.read_chunk, c.hdmi_audio_chunk_bytes)
                if not chunk:
                    await asyncio.sleep(0.02)
                    continue
                yield chunk
        finally:
            session.close()

    return StreamingResponse(audio_generator(), media_type="audio/mpeg")


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
        elif action == "LONG_KEY_PRESS":
            key_action = str(params.get("key_action", "")).strip().upper()
            key_code = str(params.get("key_code", "")).strip().upper()
            duration_ms = int(params.get("duration_ms", 1500))
            resolved_key = KEY_MAP.get(key_action) if key_action else key_code
            if not resolved_key:
                raise HTTPException(
                    status_code=400,
                    detail="LONG_KEY_PRESS requires params.key_action or params.key_code",
                )
            resp = await dab.long_key_press(resolved_key, duration_ms=duration_ms)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "LAUNCH_APP":
            app_id = params.get("app_id", "")
            if not app_id:
                raise HTTPException(status_code=400, detail="app_id is required for LAUNCH_APP")
            app_id = _validate_app_id(app_id)
            launch_parameters = {}
            content = str(params.get("content", "")).strip()
            if content:
                launch_parameters["content"] = content
            resp = await dab.launch_app(app_id, parameters=launch_parameters or None)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "WAIT":
            seconds = float(params.get("seconds", 1.0))
            seconds = max(0.0, min(seconds, 30.0))
            await asyncio.sleep(seconds)
            return ManualActionResponse(success=True, action=action, result={"seconds": seconds})
        elif action == "GET_STATE":
            app_id = params.get("app_id") or get_config().youtube_app_id
            app_id = _validate_app_id(app_id)
            resp = await dab.get_app_state(app_id)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "EXIT_APP":
            app_id = params.get("app_id") or get_config().youtube_app_id
            app_id = _validate_app_id(app_id)
            resp = await dab.exit_app(app_id)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "OPERATIONS_LIST":
            resp = await dab.list_operations()
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "APPLICATIONS_LIST":
            resp = await dab.list_apps()
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "KEY_LIST":
            resp = await dab.list_keys()
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action!r}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Manual action failed: action=%s error=%s", request.action, exc)
        return ManualActionResponse(success=False, action=request.action, error=str(exc))


@app.post("/actions/batch", response_model=ManualActionBatchResponse)
async def manual_actions_batch(request: ManualActionBatchRequest) -> ManualActionBatchResponse:
    """Execute multiple manual actions sequentially in one request."""
    results: List[ManualActionResponse] = []
    all_success = True

    for item in request.actions:
        try:
            result = await manual_action(item)
        except HTTPException as exc:
            result = ManualActionResponse(
                success=False,
                action=item.action,
                error=str(exc.detail),
            )
        results.append(result)
        if not result.success:
            all_success = False
            if not request.continue_on_error:
                break

    return ManualActionBatchResponse(
        success=all_success,
        total=len(results),
        results=results,
    )


@app.post("/task/macro", response_model=TaskMacroResponse)
async def task_macro(request: TaskMacroRequest) -> TaskMacroResponse:
    """Expand plain-language task into actions, optionally execute them."""
    actions = _plan_task_macro_actions(request.instruction)
    if not actions:
        raise HTTPException(
            status_code=400,
            detail="Could not derive actions from instruction",
        )

    execution: Optional[ManualActionBatchResponse] = None
    if request.execute:
        execution = await manual_actions_batch(
            ManualActionBatchRequest(
                actions=actions,
                continue_on_error=request.continue_on_error,
            )
        )

    return TaskMacroResponse(
        success=(execution.success if execution is not None else True),
        instruction=request.instruction,
        planned_count=len(actions),
        planned_actions=actions,
        execution=execution,
    )


# ---------------------------------------------------------------------------
# Planner debug
# ---------------------------------------------------------------------------

@app.post("/planner/debug", response_model=PlannerDebugResponse)
async def planner_debug(request: PlannerDebugRequest) -> PlannerDebugResponse:
    """Run the planner with the given inputs and return the planned action."""
    screenshot_b64 = request.screenshot_b64
    ocr_text = request.ocr_text
    used_live_capture = False
    capture_source = None

    if request.use_live_capture and not screenshot_b64:
        live = await get_screen_capture().capture()
        screenshot_b64 = live.image_b64
        capture_source = live.source
        used_live_capture = live.image_b64 is not None
        if not ocr_text:
            ocr_text = live.ocr_text

    planned = await get_planner().plan(
        goal=request.goal,
        screenshot_b64=screenshot_b64,
        ocr_text=ocr_text,
        current_app=request.current_app,
        current_screen=request.current_screen,
        last_actions=request.last_actions or [],
    )
    return PlannerDebugResponse(
        action=planned.action,
        confidence=planned.confidence,
        reason=planned.reason,
        params=planned.params,
        used_live_capture=used_live_capture,
        capture_source=capture_source,
    )


# ---------------------------------------------------------------------------
# DAB operation discovery endpoints
# ---------------------------------------------------------------------------

@app.get("/dab/operations", response_model=dict)
async def dab_operations() -> dict:
    """Return supported DAB operations from device."""
    resp = await get_dab_client().list_operations()
    return {"success": resp.success, "result": resp.data}


@app.get("/dab/keys", response_model=dict)
async def dab_keys() -> dict:
    """Return supported key list from device."""
    resp = await get_dab_client().list_keys()
    return {"success": resp.success, "result": resp.data}


@app.get("/dab/apps", response_model=dict)
async def dab_apps() -> dict:
    """Return installed/launchable application list from device."""
    resp = await get_dab_client().list_apps()
    return {"success": resp.success, "result": resp.data}
