"""FastAPI backend for vertex_live_dab_agent."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import grp
import logging
import os
import re
import shlex
import subprocess
import contextlib
import uuid
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from vertex_live_dab_agent.api.models import (
    AITranscriptResponse,
    DABTranscriptResponse,
    ActionHistoryResponse,
    ActionRecordItem,
    CaptureSelectRequest,
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

_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# Path to the bundled static frontend
_STATIC_DIR = Path(__file__).parent.parent.parent / "static"
_YTS_TESTLIST_PATH = Path("/home/harry/youtube/ai_tool/testlist.json")
_YTS_GUIDED_TESTLIST_PATH = Path("/home/harry/youtube/ai_tool/testlist_guided.json")
_YTS_INTERACTIVE_CAPTURE_ATTEMPTS = 3
_YTS_INTERACTIVE_CAPTURE_DELAY_SECONDS = 0.9
_YTS_LIVE_VISUAL_MONITOR_INTERVAL_SECONDS = 1.0
_YTS_LIVE_VISUAL_MONITOR_STALE_SECONDS = 2.5
_YTS_LIVE_VISUAL_HISTORY_LIMIT = 60

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
_vertex_text_client: Optional[VertexPlannerClient] = None
_vertex_live_visual_client: Optional[VertexPlannerClient] = None
_yts_live_commands: Dict[str, Dict[str, Any]] = {}
_yts_live_tasks: Dict[str, asyncio.Task] = {}
_yts_live_visual_tasks: Dict[str, asyncio.Task] = {}
_yts_live_processes: Dict[str, asyncio.subprocess.Process] = {}
_yts_live_recording_processes: Dict[str, Dict[str, Any]] = {}
_yts_live_visual_cache: Dict[str, Dict[str, Any]] = {}
_yts_live_db_conn: Optional[sqlite3.Connection] = None
_yts_live_db_path: Optional[Path] = None
_yts_live_db_lock = threading.Lock()

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_yts_live_db_path() -> Path:
    base_dir = os.getenv("ARTIFACTS_BASE_DIR")
    root = Path(base_dir) if base_dir else Path("/home/harry/youtube/ai_tool/artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root / "yts_live_commands.sqlite3"


def _get_yts_live_artifacts_root() -> Path:
    base_dir = os.getenv("ARTIFACTS_BASE_DIR")
    root = Path(base_dir) if base_dir else Path("/home/harry/youtube/ai_tool/artifacts")
    path = root / "yts_live"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_yts_live_artifacts_dir(command_id: str) -> Path:
    path = _get_yts_live_artifacts_root() / command_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _close_yts_live_db() -> None:
    global _yts_live_db_conn, _yts_live_db_path
    with _yts_live_db_lock:
        if _yts_live_db_conn is not None:
            _yts_live_db_conn.close()
        _yts_live_db_conn = None
        _yts_live_db_path = None


def _ensure_yts_live_db() -> sqlite3.Connection:
    global _yts_live_db_conn, _yts_live_db_path

    db_path = _get_yts_live_db_path()
    with _yts_live_db_lock:
        if _yts_live_db_conn is None or _yts_live_db_path != db_path:
            if _yts_live_db_conn is not None:
                _yts_live_db_conn.close()
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS yts_live_commands (
                    command_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    command_text TEXT,
                    interactive_ai INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
            _yts_live_db_conn = conn
            _yts_live_db_path = db_path
        return _yts_live_db_conn


def _new_yts_live_state(command_id: str, interactive_ai: bool = False) -> Dict[str, Any]:
    timestamp = _utc_now_iso()
    artifacts_dir = str(_get_yts_live_artifacts_dir(command_id))
    return {
        "command_id": command_id,
        "command": "",
        "status": "running",
        "stdout": "",
        "stderr": "",
        "logs": [],
        "prompts": [],
        "pending_prompt": None,
        "awaiting_input": False,
        "responses": [],
        "setup_actions": [],
        "executed_setup_signatures": [],
        "interactive_ai": bool(interactive_ai),
        "ai_observing_tv": False,
        "ai_status_message": None,
        "visual_monitor_active": False,
        "latest_visual_analysis": {},
        "visual_monitor_history": [],
        "last_visual_analysis_at": None,
        "artifacts_dir": artifacts_dir,
        "record_video": False,
        "video_recording_status": "disabled",
        "video_file_name": None,
        "video_file_path": None,
        "terminal_log_name": f"yts-terminal-log-{command_id}.txt",
        "terminal_log_path": str(Path(artifacts_dir) / f"yts-terminal-log-{command_id}.txt"),
        "returncode": None,
        "result_file_content": None,
        "result_file_name": None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _normalize_yts_live_state(state: Dict[str, Any]) -> Dict[str, Any]:
    command_id = str(state.get("command_id") or uuid.uuid4())
    normalized = _new_yts_live_state(command_id, bool(state.get("interactive_ai")))
    normalized.update(state)
    normalized["command_id"] = command_id
    normalized["logs"] = list(normalized.get("logs") or [])
    normalized["prompts"] = list(normalized.get("prompts") or [])
    normalized["responses"] = list(normalized.get("responses") or [])
    normalized["setup_actions"] = list(normalized.get("setup_actions") or [])
    normalized["executed_setup_signatures"] = list(normalized.get("executed_setup_signatures") or [])
    normalized["awaiting_input"] = bool(normalized.get("awaiting_input"))
    normalized["interactive_ai"] = bool(normalized.get("interactive_ai"))
    normalized["ai_observing_tv"] = bool(normalized.get("ai_observing_tv"))
    normalized["visual_monitor_active"] = bool(normalized.get("visual_monitor_active"))
    normalized["latest_visual_analysis"] = dict(normalized.get("latest_visual_analysis") or {})
    normalized["visual_monitor_history"] = list(normalized.get("visual_monitor_history") or [])
    normalized["record_video"] = bool(normalized.get("record_video"))
    artifacts_dir = Path(str(normalized.get("artifacts_dir") or _get_yts_live_artifacts_dir(command_id)))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    normalized["artifacts_dir"] = str(artifacts_dir)
    normalized["terminal_log_name"] = str(normalized.get("terminal_log_name") or f"yts-terminal-log-{command_id}.txt")
    normalized["terminal_log_path"] = str(normalized.get("terminal_log_path") or (artifacts_dir / normalized["terminal_log_name"]))
    video_path_raw = normalized.get("video_file_path")
    video_path_text = str(video_path_raw or "").strip()
    if video_path_text.startswith("<coroutine object"):
        normalized["video_file_path"] = None
        normalized["video_file_name"] = None
        if normalized.get("record_video") and normalized.get("video_recording_status") in {None, "recording", "pending"}:
            normalized["video_recording_status"] = "failed"
    elif video_path_text:
        video_path = Path(video_path_text)
        if video_path.exists() and video_path.stat().st_size > 0:
            normalized["video_file_path"] = str(video_path)
            normalized["video_file_name"] = video_path.name
            if normalized.get("status") != "running":
                normalized["video_recording_status"] = "completed"
        elif normalized.get("status") != "running" and normalized.get("record_video"):
            normalized["video_recording_status"] = "failed"
    normalized["created_at"] = str(normalized.get("created_at") or _utc_now_iso())
    normalized["updated_at"] = str(normalized.get("updated_at") or normalized["created_at"])
    return normalized


def _render_yts_terminal_log(state: Dict[str, Any]) -> str:
    logs = state.get("logs") or []
    if not logs:
        return ""
    return "\n".join(
        f"[{entry.get('stream', 'log')}] {entry.get('raw_message', entry.get('message', ''))}"
        for entry in logs
    )


def _recent_yts_terminal_log_text(state: Dict[str, Any], limit: int = 40) -> str:
    recent_logs = list(state.get("logs") or [])[-max(1, int(limit or 40)) :]
    return "\n".join(
        f"[{entry.get('stream', 'log')}] {entry.get('raw_message', entry.get('message', ''))}"
        for entry in recent_logs
    )


def _build_yts_prompt_log_context(
    state: Dict[str, Any],
    prompt_text: str,
    *,
    leading_lines: int = 60,
    trailing_lines: int = 24,
    fallback_tail_lines: int = 140,
) -> str:
    logs = list(state.get("logs") or [])
    if not logs:
        return ""

    prompt_lines = [
        _strip_terminal_ansi(line).strip()
        for line in str(prompt_text or "").splitlines()
        if _strip_terminal_ansi(line).strip()
    ]
    anchor_line = prompt_lines[0] if prompt_lines else ""
    anchor_index: Optional[int] = None

    if anchor_line:
        for idx in range(len(logs) - 1, -1, -1):
            entry_text = _strip_terminal_ansi(str(logs[idx].get("raw_message", logs[idx].get("message", "")))).strip()
            if not entry_text:
                continue
            if entry_text == anchor_line or anchor_line in entry_text:
                anchor_index = idx
                break

    if anchor_index is None:
        excerpt = logs[-max(1, int(fallback_tail_lines or 140)) :]
    else:
        start = max(0, anchor_index - max(0, int(leading_lines or 60)))
        end = min(len(logs), anchor_index + max(len(prompt_lines) + 8, int(trailing_lines or 24)))
        excerpt = logs[start:end]

    return "\n".join(
        f"[{entry.get('stream', 'log')}] {entry.get('raw_message', entry.get('message', ''))}"
        for entry in excerpt
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if "```" in raw:
        for block in re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
            candidates.append(block.strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _parse_yts_live_visual_analysis(response_text: str) -> Dict[str, Any]:
    parsed = _extract_json_object(response_text)
    if parsed is None:
        summary = str(response_text or "").strip()
        return {
            "summary": summary or "Gemini did not return structured visual analysis.",
            "playback_visible": False,
            "player_controls_visible": False,
            "settings_gear_visible": False,
            "stats_for_nerds_visible": False,
            "focus_target": "unknown",
            "confidence": 0.0,
        }

    raw_confidence = parsed.get("confidence")
    confidence: float
    try:
        confidence = float(raw_confidence or 0.0)
    except (TypeError, ValueError):
        normalized_confidence = str(raw_confidence or "").strip().lower()
        confidence_scale = {
            "very high": 0.95,
            "high": 0.85,
            "medium": 0.6,
            "moderate": 0.6,
            "low": 0.3,
            "very low": 0.1,
            "unknown": 0.0,
        }
        confidence = confidence_scale.get(normalized_confidence, 0.0)

    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "playback_visible": bool(parsed.get("playback_visible")),
        "player_controls_visible": bool(parsed.get("player_controls_visible")),
        "settings_gear_visible": bool(parsed.get("settings_gear_visible")),
        "stats_for_nerds_visible": bool(parsed.get("stats_for_nerds_visible")),
        "focus_target": str(parsed.get("focus_target") or "unknown").strip() or "unknown",
        "confidence": confidence,
    }


def _get_cached_yts_visual_context(command_id: str, max_age_seconds: float = _YTS_LIVE_VISUAL_MONITOR_STALE_SECONDS) -> Optional[Dict[str, Any]]:
    cache = _yts_live_visual_cache.get(command_id)
    if not cache:
        return None
    captured_at = cache.get("captured_at")
    if captured_at:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(str(captured_at))).total_seconds()
            if age > max(0.5, float(max_age_seconds)):
                return None
        except Exception:
            return None
    return dict(cache)


def _write_yts_terminal_log_artifact(state: Dict[str, Any]) -> Optional[Path]:
    path_raw = state.get("terminal_log_path")
    if not path_raw:
        return None
    path = Path(str(path_raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_yts_terminal_log(state), encoding="utf-8")
    state["terminal_log_path"] = str(path)
    state["terminal_log_name"] = path.name
    return path


def _resolve_yts_recording_device() -> Optional[str]:
    status = get_screen_capture().capture_source_status()
    selected = str(status.get("selected_video_device") or "").strip()
    active = str(status.get("hdmi_device") or "").strip()
    return selected or active or None


async def _capture_yts_recording_stderr(command_id: str, stream) -> None:
    state = _get_yts_live_state(command_id)
    if not state or stream is None:
        return
    while True:
        chunk = await stream.readline()
        if not chunk:
            break
        message = chunk.decode(errors="replace").strip()
        if not message:
            continue
        state["logs"].append({"stream": "recording", "message": message, "raw_message": message})
        _persist_yts_live_state(state)


async def _pump_yts_video_recording_frames(command_id: str, process: asyncio.subprocess.Process, fps: float = 5.0) -> None:
    frame_interval = 1.0 / max(1.0, float(fps))
    try:
        while True:
            entry = _yts_live_recording_processes.get(command_id)
            if not entry or process.returncode is not None:
                break
            frame = await asyncio.to_thread(get_screen_capture().get_hdmi_stream_frame_jpeg, 85)
            if frame and process.stdin is not None and not process.stdin.is_closing():
                process.stdin.write(frame)
                await process.stdin.drain()
            await asyncio.sleep(frame_interval)
    except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
        raise
    except Exception as exc:
        state = _get_yts_live_state(command_id)
        if state:
            state["logs"].append({"stream": "stderr", "message": f"Video frame capture failed: {exc}"})
            _persist_yts_live_state(state)


async def _start_yts_video_recording(command_id: str) -> None:
    state = _get_yts_live_state(command_id)
    if not state or not state.get("record_video"):
        return
    if _yts_live_recording_processes.get(command_id):
        return

    if not ffmpeg_available():
        state["video_recording_status"] = "unavailable"
        state["logs"].append({"stream": "stderr", "message": "Video recording unavailable: ffmpeg not found"})
        _write_yts_terminal_log_artifact(state)
        _persist_yts_live_state(state)
        return

    capture_status = get_screen_capture().capture_source_status()
    if not capture_status.get("hdmi_available"):
        state["video_recording_status"] = "unavailable"
        state["logs"].append({"stream": "stderr", "message": "Video recording unavailable: no active capture session found"})
        _write_yts_terminal_log_artifact(state)
        _persist_yts_live_state(state)
        return

    artifacts_dir = Path(str(state.get("artifacts_dir")))
    output_path = artifacts_dir / f"yts-video-{command_id}.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-r",
        "5",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd="/home/harry/youtube/ai_tool",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_task = asyncio.create_task(_capture_yts_recording_stderr(command_id, process.stderr))
        pump_task = asyncio.create_task(_pump_yts_video_recording_frames(command_id, process, fps=5.0))
        _yts_live_recording_processes[command_id] = {
            "process": process,
            "stderr_task": stderr_task,
            "pump_task": pump_task,
        }
        state["video_recording_status"] = "recording"
        state["video_file_path"] = str(output_path)
        state["video_file_name"] = output_path.name
        state["logs"].append({
            "stream": "system",
            "message": f"Started video recording using active capture session: {shlex.join(command)}",
        })
    except Exception as exc:
        state["video_recording_status"] = "failed"
        state["logs"].append({"stream": "stderr", "message": f"Failed to start video recording: {exc}"})

    _write_yts_terminal_log_artifact(state)
    _persist_yts_live_state(state)


async def _stop_yts_video_recording(command_id: str) -> None:
    recording_entry = _yts_live_recording_processes.pop(command_id, None)
    state = _get_yts_live_state(command_id)
    if recording_entry is None or state is None:
        if state:
            _write_yts_terminal_log_artifact(state)
            _persist_yts_live_state(state)
        return

    process = recording_entry.get("process")
    pump_task = recording_entry.get("pump_task")
    stderr_task = recording_entry.get("stderr_task")

    try:
        if pump_task is not None and not pump_task.done():
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task
        if process is not None and process.returncode is None:
            if process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
                with contextlib.suppress(Exception):
                    await process.stdin.wait_closed()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        if stderr_task is not None and not stderr_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
    finally:
        output_path = Path(str(state.get("video_file_path") or "")) if state.get("video_file_path") else None
        if output_path and output_path.exists() and output_path.stat().st_size > 0:
            state["video_recording_status"] = "completed"
            state["video_file_name"] = output_path.name
            state["video_file_path"] = str(output_path)
        elif state.get("record_video"):
            state["video_recording_status"] = "failed"
        _write_yts_terminal_log_artifact(state)
        _persist_yts_live_state(state)


def _persist_yts_live_state(state: Dict[str, Any]) -> Dict[str, Any]:
    conn = _ensure_yts_live_db()
    normalized = _normalize_yts_live_state(state)
    normalized["updated_at"] = _utc_now_iso()
    state.clear()
    state.update(normalized)
    payload = json.dumps(normalized)
    with _yts_live_db_lock:
        conn.execute(
            """
            INSERT INTO yts_live_commands (
                command_id, status, command_text, interactive_ai, created_at, updated_at, state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(command_id) DO UPDATE SET
                status = excluded.status,
                command_text = excluded.command_text,
                interactive_ai = excluded.interactive_ai,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                state_json = excluded.state_json
            """,
            (
                normalized["command_id"],
                normalized["status"],
                normalized.get("command") or "",
                1 if normalized.get("interactive_ai") else 0,
                normalized["created_at"],
                normalized["updated_at"],
                payload,
            ),
        )
        conn.commit()
    return state


def _load_yts_live_state(command_id: str) -> Optional[Dict[str, Any]]:
    conn = _ensure_yts_live_db()
    with _yts_live_db_lock:
        row = conn.execute(
            "SELECT state_json FROM yts_live_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
    if not row:
        return None
    return _normalize_yts_live_state(json.loads(row["state_json"]))


def _get_yts_live_state(command_id: str) -> Optional[Dict[str, Any]]:
    state = _yts_live_commands.get(command_id)
    if state:
        return state
    loaded = _load_yts_live_state(command_id)
    if loaded:
        _yts_live_commands[command_id] = loaded
    return loaded


def _summarize_yts_live_state(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_yts_live_state(state)
    return {
        "command_id": normalized.get("command_id"),
        "command": normalized.get("command"),
        "status": normalized.get("status"),
        "interactive_ai": normalized.get("interactive_ai"),
        "record_video": normalized.get("record_video"),
        "video_recording_status": normalized.get("video_recording_status"),
        "video_file_name": normalized.get("video_file_name"),
        "result_file_name": normalized.get("result_file_name"),
        "awaiting_input": normalized.get("awaiting_input"),
        "updated_at": normalized.get("updated_at"),
        "created_at": normalized.get("created_at"),
        "returncode": normalized.get("returncode"),
        "artifacts_dir": normalized.get("artifacts_dir"),
        "log_count": len(normalized.get("logs") or []),
        "response_count": len(normalized.get("responses") or []),
    }


def _list_yts_live_states(limit: int = 10, active_only: bool = False) -> List[Dict[str, Any]]:
    conn = _ensure_yts_live_db()
    capped_limit = max(1, min(int(limit or 10), 100))
    query = "SELECT state_json FROM yts_live_commands"
    params: List[Any] = []
    if active_only:
        query += " WHERE status = ?"
        params.append("running")
    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(capped_limit)
    with _yts_live_db_lock:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_summarize_yts_live_state(json.loads(row["state_json"])) for row in rows]


def _mark_stale_yts_live_commands() -> None:
    stale_states = _list_yts_live_states(limit=100, active_only=True)
    if not stale_states:
        return
    for state_summary in stale_states:
        command_id = str(state_summary.get("command_id") or "")
        if not command_id or command_id in _yts_live_tasks:
            continue
        state = _get_yts_live_state(command_id) or _normalize_yts_live_state({"command_id": command_id})
        message = "Server restarted while this YTS command was running. Live terminal attachment is no longer available."
        if message not in str(state.get("stderr") or ""):
            separator = "\n" if state.get("stderr") else ""
            state["stderr"] = f"{state.get('stderr') or ''}{separator}{message}"
            state["logs"].append({"stream": "stderr", "message": message})
        state["status"] = "failed"
        state["awaiting_input"] = False
        state["pending_prompt"] = None
        _persist_yts_live_state(state)


def _read_yts_test_catalog(path: Path = _YTS_TESTLIST_PATH) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, dict):
        tests = content.get("tests", [])
        return tests if isinstance(tests, list) else []
    if isinstance(content, list):
        return content
    return []


def _catalog_path_for_mode(guided: bool = False) -> Path:
    return _YTS_GUIDED_TESTLIST_PATH if guided else _YTS_TESTLIST_PATH


def _refresh_yts_test_catalog(
    path: Path = _YTS_TESTLIST_PATH,
    guided: bool = False,
    raise_on_error: bool = False,
) -> List[Dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        discover_res = _run_yts_command(["yts", "discover", "--list"])
        if discover_res["returncode"] != 0:
            raise RuntimeError(f"YTS discover failed: {discover_res['stderr']}")

        list_cmd = ["yts", "list"]
        if guided:
            list_cmd.append("--guided")
        list_cmd.extend(["--json-output", str(path)])
        list_res = _run_yts_command(list_cmd)
        if list_res["returncode"] != 0:
            raise RuntimeError(f"YTS list failed: {list_res['stderr']}")

        return _read_yts_test_catalog(path)
    except Exception as exc:
        cached = _read_yts_test_catalog(path)
        if cached:
            logger.warning("Using cached YTS test catalog after refresh failure: %s", exc)
            return cached
        if raise_on_error:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        logger.warning("Unable to refresh YTS test catalog at startup: %s", exc)
        return []


def _build_yts_command(request: "YtsCommandRequest") -> List[str]:
    cmd = ["yts"]

    for option, value in request.global_options.items():
        if isinstance(value, bool) and value:
            cmd.append(option)
        elif isinstance(value, str) and value:
            cmd.extend([option, value])

    cmd.append(request.command)
    cmd.extend(request.params)
    return cmd


def _is_interactive_yts_prompt(text: str) -> bool:
    line = _strip_terminal_ansi(str(text or "")).strip().lower()
    if not line:
        return False
    prompt_markers = [
        "yes/no",
        "(y/n)",
        "enter choice",
        "enter selection",
        "please select",
        "select an option",
        "choose an option",
        "press 1",
        "press 2",
        "press 3",
        "press 4",
        "type yes",
        "type no",
    ]
    if any(marker in line for marker in prompt_markers):
        return True
    if re.search(r"\b[1-4]\)\b|\[[1-4]\]|\b1/2/3/4\b", line):
        return True
    return "?" in line


def _strip_terminal_ansi(text: str) -> str:
    cleaned = _ANSI_ESCAPE_RE.sub("", str(text or ""))
    return re.sub(r"^\s*\d{2}:\d{2}:\d{2}\.\d{3}\s+", "", cleaned)


def _parse_yts_prompt_option(text: str) -> Optional[str]:
    line = _strip_terminal_ansi(str(text or "")).strip()
    match = re.match(r"^([1-9])\s*[:.)-]\s+.+$", line)
    if match:
        return match.group(1)
    return None


def _extract_yts_prompt_option_labels(text: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = _strip_terminal_ansi(raw_line).strip()
        match = re.match(r"^([1-9])\s*[:.)-]\s+(.+)$", line)
        if not match:
            continue
        labels[match.group(1)] = re.sub(r"\s+", " ", match.group(2)).strip().lower()
    return labels


def _is_prompt_scaffolding_line(text: str) -> bool:
    line = _strip_terminal_ansi(str(text or "")).strip().lower()
    if not line:
        return False
    scaffolding_markers = [
        "please select from the following options",
        "select from the following options",
        "select an option",
        "choose an option",
        "enter choice",
        "enter selection",
        "available options",
    ]
    return any(marker in line for marker in scaffolding_markers)


def _prompt_decision_lines(prompt_entry: Dict[str, Any]) -> List[str]:
    decision_lines: List[str] = []
    for raw_line in str(prompt_entry.get("text") or "").splitlines():
        line = _strip_terminal_ansi(raw_line).strip()
        if not line:
            continue
        if _parse_yts_prompt_option(line):
            continue
        if _is_prompt_scaffolding_line(line):
            continue
        decision_lines.append(line)
    return decision_lines


def _extract_prompt_options(text: str) -> List[str]:
    line = _strip_terminal_ansi(str(text or ""))
    options: List[str] = []
    lowered = line.lower()
    if "yes/no" in lowered or "(y/n)" in lowered:
        options.extend(["yes", "no"])
    numbered_option = _parse_yts_prompt_option(line)
    if numbered_option:
        options.append(numbered_option)
    for digit in ["1", "2", "3", "4"]:
        if re.search(rf"(^|\D){digit}(\D|$)", line):
            options.append(digit)
    deduped: List[str] = []
    for option in options:
        if option not in deduped:
            deduped.append(option)
    return deduped


def _extract_setting_value(text: str, setting_name: str) -> Optional[str]:
    pattern = rf"(?:set|change|update)\s+(?:the\s+)?(?:{setting_name})(?:\s+(?:setting|value))?\s+(?:to|as)\s+([^\n.,;]+)"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip(" \t'\"()[]{}")
    return value or None


def _extract_locale_value(text: str, label: str) -> Optional[str]:
    pattern = rf"{label}\s*[:=]\s*([a-z]{{2,3}}(?:-[A-Za-z0-9]{{2,8}}){{0,3}})"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def _choose_alternate_language_value(current_value: Optional[str]) -> str:
    normalized = str(current_value or "").strip()
    lowered = normalized.lower()
    if lowered == "en-us":
        return "en-GB"
    if lowered == "en-gb":
        return "en-US"
    if lowered.startswith("en-"):
        return "en-US"
    return "en-US"


def _extract_language_setting_value(text: str) -> Optional[str]:
    combined = str(text or "")
    explicit_value = _extract_setting_value(combined, r"navigator\.language|device\s+language|language|locale")
    if explicit_value:
        return explicit_value
    initial_value = _extract_locale_value(combined, r"initial(?:\s+value(?:\s+of\s+navigator\.language)?)?")
    if initial_value:
        return _choose_alternate_language_value(initial_value)
    if re.search(r"navigator\.language|device\s+language|\blanguage\b|\blocale\b", combined, flags=re.IGNORECASE):
        return _choose_alternate_language_value(None)
    return None


def _normalize_setting_key(setting_key: str) -> str:
    normalized = str(setting_key or "").strip().lower()
    aliases = {
        "time zone": "timezone",
        "timezone": "timezone",
        "navigator.language": "language",
        "navigator_language": "language",
        "device language": "language",
        "language": "language",
        "locale": "language",
    }
    return aliases.get(normalized, str(setting_key or "").strip())


def _extract_yts_setup_instruction(prompt_text: str, log_text: str) -> Optional[str]:
    combined = "\n".join(part for part in [str(log_text or "").strip(), str(prompt_text or "").strip()] if part).strip()
    if not combined:
        return None

    timezone_value = _extract_setting_value(combined, r"time\s*zone|timezone")
    if timezone_value:
        return f"set time zone to {timezone_value}"

    language_value = _extract_language_setting_value(combined)
    if language_value:
        return f"set language to {language_value}"

    lowered = combined.lower()
    if "open settings" in lowered:
        return "open settings"

    return None


async def _maybe_execute_yts_setup_actions(command_id: str, prompt_text: str, log_text: str) -> List[Dict[str, Any]]:
    state = _get_yts_live_state(command_id)
    if not state:
        return []

    instruction = _extract_yts_setup_instruction(prompt_text, log_text)
    if not instruction:
        return []

    planned_actions = _plan_task_macro_actions(instruction)
    if not planned_actions:
        return []

    signature = json.dumps([action.model_dump() for action in planned_actions], sort_keys=True)
    executed_signatures = list(state.get("executed_setup_signatures") or [])
    if signature in executed_signatures:
        return []

    executed_records: List[Dict[str, Any]] = []
    for action in planned_actions:
        result = await manual_action(action)
        record = {
            "instruction": instruction,
            "action": action.action,
            "params": action.params or {},
            "success": bool(result.success),
            "result": result.result,
            "error": result.error,
            "executed_at": _utc_now_iso(),
        }
        executed_records.append(record)
        state["logs"].append(
            {
                "stream": "dab-setup",
                "message": f"{action.action} => {'ok' if result.success else 'failed'}",
                "raw_message": json.dumps(record, ensure_ascii=False),
            }
        )

    executed_signatures.append(signature)
    state["executed_setup_signatures"] = executed_signatures[-20:]
    state.setdefault("setup_actions", []).append(
        {
            "instruction": instruction,
            "actions": executed_records,
            "executed_at": _utc_now_iso(),
        }
    )
    _persist_yts_live_state(state)
    return executed_records


def _merge_yts_prompt_entry(prompt_entry: Dict[str, Any], line: str, stream_name: str) -> Dict[str, Any]:
    cleaned_line = _strip_terminal_ansi(str(line or "")).strip()
    if not cleaned_line:
        return prompt_entry

    original_text = str(prompt_entry.get("text") or "")
    original_options = list(prompt_entry.get("options") or [])
    existing_lines = [segment.strip() for segment in str(prompt_entry.get("text") or "").splitlines() if segment.strip()]
    if cleaned_line not in existing_lines:
        prompt_entry["text"] = f"{prompt_entry.get('text')}\n{cleaned_line}".strip() if prompt_entry.get("text") else cleaned_line

    options = list(prompt_entry.get("options") or [])
    for option in _extract_prompt_options(cleaned_line):
        if option not in options:
            options.append(option)
    prompt_entry["options"] = options
    prompt_entry["stream"] = stream_name
    if not prompt_entry.get("answered") and (
        prompt_entry.get("text") != original_text or prompt_entry.get("options") != original_options
    ):
        for key in ["ai_suggestion", "ai_source", "ai_visual_summary", "ai_visual_source", "ai_error"]:
            prompt_entry.pop(key, None)
    return prompt_entry


def _prompt_ready_for_ai_response(prompt_entry: Dict[str, Any]) -> bool:
    prompt_text = str(prompt_entry.get("text") or "")
    lowered_prompt = prompt_text.lower()
    options = list(prompt_entry.get("options") or [])
    decision_lines = _prompt_decision_lines(prompt_entry)
    option_labels = _extract_yts_prompt_option_labels(prompt_text)
    numeric_options = [option for option in options if str(option).isdigit() and option in option_labels]
    if not decision_lines:
        return False

    has_question_line = any("?" in line for line in decision_lines)
    has_decision_verb = any(
        re.search(
            r"^(do|does|did|is|are|can|could|should|would|will|continue|proceed|confirm|accept|allow|enable|disable|keep|use|select|choose)\b",
            line.lower(),
        )
        for line in decision_lines
    )
    has_yes_no_marker = any(marker in lowered_prompt for marker in ("yes/no", "(y/n)", "type yes", "type no"))
    has_selection_marker = any(
        marker in lowered_prompt for marker in ("enter choice", "enter selection", "please select", "select an option", "choose an option")
    )
    has_visible_selection_options = len(options) >= 2
    has_explicit_answer_format = has_yes_no_marker or has_visible_selection_options

    if numeric_options:
        return len(numeric_options) >= 2 and (has_question_line or has_decision_verb or has_selection_marker)
    if (has_question_line or has_decision_verb) and has_explicit_answer_format:
        return True
    if has_yes_no_marker and decision_lines:
        return True
    if has_selection_marker and has_visible_selection_options and decision_lines:
        return True
    return False


def _yts_prompt_requires_numeric_response(prompt_text: str, options: Optional[List[str]] = None) -> bool:
    prompt_text = str(prompt_text or "")
    provided_options = [str(option).strip() for option in (options or []) if str(option).strip()]
    option_labels = _extract_yts_prompt_option_labels(prompt_text)
    numeric_option_labels = [option for option in option_labels if option.isdigit()]
    numeric_options = [option for option in provided_options if option.isdigit()]
    if len(numeric_option_labels) >= 2:
        return True
    if len(numeric_options) >= 2:
        return True
    return False


def _normalize_yts_ai_suggestion(prompt_text: str, options: List[str], suggestion: str) -> str:
    normalized_suggestion = str(suggestion or "").strip()
    if not normalized_suggestion:
        return normalized_suggestion

    if _yts_prompt_requires_numeric_response(prompt_text, options):
        if normalized_suggestion in options and normalized_suggestion.isdigit():
            return normalized_suggestion
        match = re.search(r"\b(\d+)\b", normalized_suggestion)
        if match and match.group(1) in options:
            return match.group(1)

    option_labels = _extract_yts_prompt_option_labels(prompt_text)
    if option_labels:
        lowered = normalized_suggestion.lower()
        direct_match = option_labels.get(lowered)
        if direct_match:
            return lowered
        for option, label in option_labels.items():
            tokens = {label, label.replace("-", " ").strip()}
            if label.startswith("yes") and lowered in {"yes", "y", "true"}:
                return option
            if label.startswith("no") and lowered in {"no", "n", "false"}:
                return option
            if any(lowered == token or lowered in token or token in lowered for token in tokens if token):
                return option

    if options:
        lowered = normalized_suggestion.lower()
        for option in options:
            if lowered == option.lower() or option.lower() in lowered:
                return option
    return normalized_suggestion


def _update_yts_prompt_entry(command_id: str, prompt_id: int, **updates: Any) -> Optional[Dict[str, Any]]:
    state = _get_yts_live_state(command_id)
    if not state:
        return None

    target_prompt: Optional[Dict[str, Any]] = None
    for prompt in state.get("prompts") or []:
        if prompt.get("id") == prompt_id:
            target_prompt = prompt
            break
    if target_prompt is None:
        return None

    target_prompt.update(updates)
    pending_prompt = state.get("pending_prompt")
    if isinstance(pending_prompt, dict) and pending_prompt.get("id") == prompt_id:
        pending_prompt.update(updates)
        if pending_prompt.get("answered"):
            state["pending_prompt"] = None
            state["awaiting_input"] = False

    _persist_yts_live_state(state)
    return target_prompt


async def _send_yts_command_input(command_id: str, response_text: str, source: str = "manual") -> dict:
    state = _get_yts_live_state(command_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"YTS command {command_id!r} not found")

    if source != "manual":
        pending_prompt = state.get("pending_prompt") if isinstance(state.get("pending_prompt"), dict) else None
        if not state.get("awaiting_input") or pending_prompt is None or pending_prompt.get("answered"):
            raise HTTPException(status_code=409, detail="YTS prompt is no longer awaiting AI input")

    process = _yts_live_processes.get(command_id)
    if process is None or process.stdin is None or process.returncode is not None:
        raise HTTPException(status_code=409, detail="YTS command is not accepting input")

    payload = str(response_text or "").strip()
    if not payload:
        raise HTTPException(status_code=400, detail="Response text is required")

    process.stdin.write((payload + "\n").encode())
    await process.stdin.drain()
    pending_prompt = state.get("pending_prompt") if isinstance(state.get("pending_prompt"), dict) else None
    pending_prompt_id = pending_prompt.get("id") if pending_prompt else None
    state["responses"].append({"source": source, "message": payload})
    state["logs"].append({"stream": "stdin", "message": payload})
    if pending_prompt_id is not None:
        for prompt_entry in state.get("prompts") or []:
            if prompt_entry.get("id") == pending_prompt_id:
                prompt_entry["answered"] = True
                prompt_entry["response"] = payload
                prompt_entry["response_source"] = source
                break
    state["awaiting_input"] = False
    state["pending_prompt"] = None
    _persist_yts_live_state(state)
    return {"command_id": command_id, "response": payload, "source": source}


def _heuristic_yts_prompt_response(prompt_text: str, options: Optional[List[str]] = None) -> str:
    lowered = str(prompt_text or "").lower()
    option_labels = _extract_yts_prompt_option_labels(prompt_text)
    if _yts_prompt_requires_numeric_response(prompt_text, options):
        numeric_options = [str(option).strip() for option in (options or []) if str(option).strip().isdigit()]
        if numeric_options:
            return numeric_options[0]
        for option in option_labels:
            if str(option).isdigit():
                return option
    if option_labels:
        for option, label in option_labels.items():
            if label.startswith("yes"):
                return option
        return next(iter(option_labels))
    if "yes/no" in lowered or "(y/n)" in lowered or "type yes" in lowered:
        return "yes"
    for digit in ["1", "2", "3", "4"]:
        if re.search(rf"(^|\D){digit}(\D|$)", prompt_text):
            return digit
    return "yes"


async def _capture_yts_visual_context(command_id: str) -> Dict[str, Any]:
    state = _get_yts_live_state(command_id) or {}
    cached = _get_cached_yts_visual_context(command_id)
    if cached:
        visual_context = {
            "summary": cached.get("summary") or "Using latest Gemini live visual analysis.",
            "source": cached.get("source") or "unknown",
            "screenshot_b64": cached.get("screenshot_b64"),
            "observations": list(cached.get("observations") or []),
            "capture_status": dict(cached.get("capture_status") or {}),
            "analysis": dict(cached.get("analysis") or {}),
        }
        if state:
            state["last_visual_context"] = {
                "summary": visual_context["summary"],
                "source": visual_context["source"],
                "observations": visual_context["observations"],
                "capture_status": visual_context["capture_status"],
                "captured_at": cached.get("captured_at") or _utc_now_iso(),
                "analysis": visual_context["analysis"],
            }
            _persist_yts_live_state(state)
        return visual_context

    capture_status: Dict[str, Any] = {}
    summary = "No TV visual context captured yet."
    screenshot_b64: Optional[str] = None
    source = "unknown"
    observations: List[Dict[str, Any]] = []
    use_live_stream_only = False

    try:
        capture = get_screen_capture()
        capture_status = capture.capture_source_status()
        use_live_stream_only = bool(capture_status.get("hdmi_available")) and str(capture_status.get("configured_source") or "").lower() != "dab"
        for attempt in range(_YTS_INTERACTIVE_CAPTURE_ATTEMPTS):
            if use_live_stream_only and hasattr(capture, "capture_live_stream_frame"):
                result = await capture.capture_live_stream_frame()
            else:
                result = await capture.capture()
            source = str(result.source or capture_status.get("configured_source") or "unknown")
            image_b64 = result.image_b64
            if image_b64:
                screenshot_b64 = image_b64
            observations.append(
                {
                    "attempt": attempt + 1,
                    "source": source,
                    "has_screenshot": bool(image_b64),
                }
            )
            if attempt < (_YTS_INTERACTIVE_CAPTURE_ATTEMPTS - 1):
                await asyncio.sleep(_YTS_INTERACTIVE_CAPTURE_DELAY_SECONDS)

        capture_count = len(observations)
        screenshot_count = sum(1 for item in observations if item.get("has_screenshot"))

        if screenshot_b64:
            summary = (
                f"Captured {capture_count} TV frame(s) from {source}; "
                f"{screenshot_count} frame(s) included screenshots. "
                "Use the attached screenshot directly as the visual source of truth."
            )
        else:
            summary = (
                f"No screenshot could be captured over {capture_count} attempts from {source}. "
                "Use the terminal guide and choose the safest option."
            )
    except Exception as exc:
        logger.warning("Unable to capture TV visual context for YTS command %s: %s", command_id, exc)
        summary = f"Failed to capture TV visual context: {exc}"

    visual_context = {
        "summary": summary,
        "source": source,
        "screenshot_b64": screenshot_b64,
        "observations": observations,
        "analysis": {},
        "capture_status": {
            "configured_source": capture_status.get("configured_source"),
            "selected_video_device": capture_status.get("selected_video_device"),
            "hdmi_available": capture_status.get("hdmi_available"),
            "live_stream_only": use_live_stream_only,
        },
    }
    if state:
        state["last_visual_context"] = {
            "summary": summary,
            "source": source,
            "observations": observations,
            "capture_status": visual_context["capture_status"],
            "captured_at": _utc_now_iso(),
            "analysis": {},
        }
        _persist_yts_live_state(state)
    return visual_context


def get_vertex_text_client() -> Optional[VertexPlannerClient]:
    global _vertex_text_client
    if _vertex_text_client is not None:
        return _vertex_text_client
    if not _vertex_planner_requested():
        return None
    c = get_config()
    try:
        project = _resolve_vertex_project(c.google_cloud_project)
        _vertex_text_client = VertexPlannerClient(
            project=project,
            location=c.google_cloud_location,
            model=c.vertex_planner_model,
        )
    except Exception as exc:
        logger.warning("Vertex text client unavailable for YTS interactive help: %s", exc)
        _vertex_text_client = None
    return _vertex_text_client


def _is_live_preview_vertex_model(model_name: Optional[str]) -> bool:
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return False
    live_markers = (
        "-live",
        "live-preview",
        "native-audio",
        "audio-preview",
    )
    return any(marker in normalized for marker in live_markers)


def _select_vertex_visual_model() -> str:
    c = get_config()
    live_model = str(c.vertex_live_model or "").strip()
    planner_model = str(c.vertex_planner_model or "").strip()
    if live_model and not _is_live_preview_vertex_model(live_model):
        return live_model
    return planner_model or live_model


def get_vertex_live_visual_client() -> Optional[VertexPlannerClient]:
    global _vertex_live_visual_client
    if _vertex_live_visual_client is not None:
        return _vertex_live_visual_client
    if not _vertex_planner_requested():
        return None
    c = get_config()
    try:
        project = _resolve_vertex_project(c.google_cloud_project)
        _vertex_live_visual_client = VertexPlannerClient(
            project=project,
            location=c.google_cloud_location,
            model=_select_vertex_visual_model(),
        )
    except Exception as exc:
        logger.warning("Vertex live visual client unavailable for YTS monitoring: %s", exc)
        _vertex_live_visual_client = None
    return _vertex_live_visual_client


async def _capture_yts_live_monitor_frame(command_id: str) -> Dict[str, Any]:
    capture = get_screen_capture()
    capture_status = capture.capture_source_status()
    use_live_stream_only = bool(capture_status.get("hdmi_available")) and str(capture_status.get("configured_source") or "").lower() != "dab"
    if use_live_stream_only and hasattr(capture, "capture_live_stream_frame"):
        result = await capture.capture_live_stream_frame()
    else:
        result = await capture.capture()
    source = str(result.source or capture_status.get("configured_source") or "unknown")
    return {
        "source": source,
        "screenshot_b64": result.image_b64,
        "observations": [{"attempt": 1, "source": source, "has_screenshot": bool(result.image_b64)}],
        "capture_status": {
            "configured_source": capture_status.get("configured_source"),
            "selected_video_device": capture_status.get("selected_video_device"),
            "hdmi_available": capture_status.get("hdmi_available"),
            "live_stream_only": use_live_stream_only,
        },
    }


async def _refresh_yts_live_visual_monitor(command_id: str) -> Optional[Dict[str, Any]]:
    state = _get_yts_live_state(command_id)
    if not state or state.get("status") != "running" or not state.get("interactive_ai"):
        return None

    snapshot = await _capture_yts_live_monitor_frame(command_id)
    screenshot_b64 = snapshot.get("screenshot_b64")
    if not screenshot_b64:
        state["visual_monitor_active"] = True
        state["latest_visual_analysis"] = {
            "summary": "Live visual monitor could not capture a TV frame.",
            "source": snapshot.get("source") or "unknown",
            "capture_status": snapshot.get("capture_status") or {},
        }
        state["last_visual_analysis_at"] = _utc_now_iso()
        state.setdefault("visual_monitor_history", []).append(dict(state["latest_visual_analysis"]))
        state["visual_monitor_history"] = state["visual_monitor_history"][-_YTS_LIVE_VISUAL_HISTORY_LIMIT:]
        _persist_yts_live_state(state)
        return state["latest_visual_analysis"]

    client = get_vertex_live_visual_client() or get_vertex_text_client()
    log_text = _recent_yts_terminal_log_text(state, limit=30)
    analysis: Dict[str, Any]
    if client is None:
        analysis = {
            "summary": "Gemini live visual monitor unavailable; using raw HDMI frame only.",
            "playback_visible": False,
            "player_controls_visible": False,
            "settings_gear_visible": False,
            "stats_for_nerds_visible": False,
            "focus_target": "unknown",
            "confidence": 0.0,
        }
    else:
        prompt = (
            "You are continuously monitoring an Android TV guided validation run. "
            "Inspect the attached screenshot directly. Do not rely on OCR or local text extraction. "
            "Return strict JSON with keys: summary, playback_visible, player_controls_visible, "
            "settings_gear_visible, stats_for_nerds_visible, focus_target, confidence. "
            "Use short factual summary text. If numbered or menu options appear, mention only what is visibly selected.\n\n"
            f"YTS command: {state.get('command') or 'unknown'}\n"
            f"Recent terminal logs:\n{log_text or '(no recent logs)'}\n"
        )
        try:
            response = await client.generate_content(
                prompt,
                screenshot_b64=screenshot_b64,
                session_id=f"yts-live-visual-{command_id}",
            )
        except Exception as exc:
            fallback_client = get_vertex_text_client()
            live_model = str(get_config().vertex_live_model or "").strip()
            if fallback_client is None or fallback_client is client:
                raise
            logger.warning(
                "YTS live visual monitor falling back to planner model after visual model failure (%s): %s",
                live_model or "unknown-model",
                exc,
            )
            response = await fallback_client.generate_content(
                prompt,
                screenshot_b64=screenshot_b64,
                session_id=f"yts-live-visual-{command_id}",
            )
        analysis = _parse_yts_live_visual_analysis(response)

    captured_at = _utc_now_iso()
    cache_entry = {
        "summary": analysis.get("summary") or "Gemini live visual analysis available.",
        "source": snapshot.get("source") or "unknown",
        "screenshot_b64": screenshot_b64,
        "observations": snapshot.get("observations") or [],
        "capture_status": snapshot.get("capture_status") or {},
        "analysis": analysis,
        "captured_at": captured_at,
    }
    _yts_live_visual_cache[command_id] = cache_entry

    history_entry = {
        "captured_at": captured_at,
        "source": cache_entry["source"],
        "summary": cache_entry["summary"],
        "analysis": analysis,
    }
    state["visual_monitor_active"] = True
    state["latest_visual_analysis"] = history_entry
    state["last_visual_analysis_at"] = captured_at
    state.setdefault("visual_monitor_history", []).append(history_entry)
    state["visual_monitor_history"] = state["visual_monitor_history"][-_YTS_LIVE_VISUAL_HISTORY_LIMIT:]
    state["last_visual_context"] = {
        "summary": cache_entry["summary"],
        "source": cache_entry["source"],
        "observations": cache_entry["observations"],
        "capture_status": cache_entry["capture_status"],
        "captured_at": captured_at,
        "analysis": analysis,
    }
    _persist_yts_live_state(state)
    return history_entry


async def _run_yts_live_visual_monitor(command_id: str) -> None:
    try:
        while True:
            state = _get_yts_live_state(command_id)
            if not state or state.get("status") != "running" or not state.get("interactive_ai"):
                break
            try:
                await _refresh_yts_live_visual_monitor(command_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("YTS live visual monitor failed for %s: %s", command_id, exc)
                state = _get_yts_live_state(command_id)
                if state:
                    state["visual_monitor_active"] = True
                    state["latest_visual_analysis"] = {
                        "captured_at": _utc_now_iso(),
                        "summary": f"Live visual monitor error: {exc}",
                        "analysis": {},
                        "source": "error",
                    }
                    state.setdefault("visual_monitor_history", []).append(dict(state["latest_visual_analysis"]))
                    state["visual_monitor_history"] = state["visual_monitor_history"][-_YTS_LIVE_VISUAL_HISTORY_LIMIT:]
                    _persist_yts_live_state(state)
            await asyncio.sleep(_YTS_LIVE_VISUAL_MONITOR_INTERVAL_SECONDS)
    finally:
        state = _get_yts_live_state(command_id)
        if state:
            state["visual_monitor_active"] = False
            _persist_yts_live_state(state)
        _yts_live_visual_tasks.pop(command_id, None)
        _yts_live_visual_cache.pop(command_id, None)


async def _suggest_yts_prompt_response(command_id: str, prompt_text: str, options: Optional[List[str]] = None) -> dict:
    options = options or []
    state = _get_yts_live_state(command_id) or {}
    active_test_log_text = _build_yts_prompt_log_context(state, prompt_text)
    recent_log_text = _recent_yts_terminal_log_text(state, limit=140)
    log_text = active_test_log_text or recent_log_text
    numeric_response_required = _yts_prompt_requires_numeric_response(prompt_text, options)
    fallback = _heuristic_yts_prompt_response(prompt_text, options)
    if state:
        state["ai_observing_tv"] = True
        state["ai_status_message"] = "Gemini is watching the TV stream and reading the terminal guide..."
        _persist_yts_live_state(state)

    try:
        setup_actions = await _maybe_execute_yts_setup_actions(command_id, prompt_text, log_text)
        visual_context = await _capture_yts_visual_context(command_id)
        client = get_vertex_text_client()
        if client is None:
            return {
                "response": fallback,
                "source": "heuristic",
                "visual_summary": visual_context.get("summary"),
                "visual_source": visual_context.get("source"),
                "setup_actions": setup_actions,
            }

        response_instruction = (
            "The terminal expects a numbered choice. Return only one numeric option token such as 1, 2, 3, or 4. Do not return yes or no.\n\n"
            if numeric_response_required
            else "Prefer a single token like 1, 2, 3, 4, yes, or no. If the terminal shows numbered options, return the number only.\n\n"
        )
        prompt = "".join(
            [
                "You are helping answer an interactive YTS terminal prompt. ",
                "Read the terminal guide carefully and inspect the attached TV screenshot before answering. ",
                "Return only the exact response to send back to the terminal. ",
                response_instruction,
                f"Interactive prompt: {prompt_text}\n",
                f"Allowed options: {', '.join(options) if options else 'infer from prompt'}\n\n",
                "Use these rules:\n",
                "1. The active test context and recent terminal logs may contain the test guide, the current test name, and operator instructions.\n",
                "2. The attached screenshot is the primary visual context.\n",
                "3. Choose the option that matches both the guide and what is visible on TV.\n",
                "4. If the screen and logs conflict, prefer the screenshot.\n",
                "5. Treat setup notes, headings, and option lists as context only until there is a clear question or decision request.\n",
                "6. If uncertain, choose the safest non-destructive option.\n\n",
                f"Active test terminal context:\n{active_test_log_text or '(active test context unavailable)'}\n\n",
                f"Recent terminal logs:\n{recent_log_text or '(no recent logs)'}\n\n",
                f"Executed DAB setup actions before answering: {json.dumps(setup_actions, ensure_ascii=False)}\n\n",
                f"TV visual context source: {visual_context.get('source', 'unknown')}\n",
                f"TV observation sequence: {json.dumps(visual_context.get('observations') or [], ensure_ascii=False)}\n\n",
                f"Latest Gemini live visual analysis: {json.dumps(visual_context.get('analysis') or state.get('latest_visual_analysis') or {}, ensure_ascii=False)}\n\n",
                f"TV capture summary:\n{visual_context.get('summary', 'No TV screenshot available.')}\n\n",
                f"Capture status: {json.dumps(visual_context.get('capture_status') or {}, ensure_ascii=False)}\n",
            ]
        )
        try:
            response = await client.generate_content(
                prompt,
                screenshot_b64=visual_context.get("screenshot_b64"),
                session_id=f"yts-live-{command_id}",
            )
            suggestion = str(response or "").strip().splitlines()[0].strip()
            if not suggestion:
                suggestion = fallback
            suggestion = _normalize_yts_ai_suggestion(prompt_text, options, suggestion)
            if numeric_response_required and (suggestion not in options or not str(suggestion).isdigit()):
                suggestion = fallback
            return {
                "response": suggestion,
                "source": "gemini",
                "visual_summary": visual_context.get("summary"),
                "visual_source": visual_context.get("source"),
                "setup_actions": setup_actions,
            }
        except Exception as exc:
            logger.warning("Gemini prompt suggestion failed for YTS command %s: %s", command_id, exc)
            return {
                "response": fallback,
                "source": "heuristic",
                "visual_summary": visual_context.get("summary"),
                "visual_source": visual_context.get("source"),
                "setup_actions": setup_actions,
            }
    finally:
        refreshed_state = _get_yts_live_state(command_id)
        if refreshed_state:
            refreshed_state["ai_observing_tv"] = False
            refreshed_state["ai_status_message"] = None
            _persist_yts_live_state(refreshed_state)


async def _append_yts_stream_output(command_id: str, stream_name: str, stream) -> None:
    if stream is None:
        return

    while True:
        chunk = await stream.readline()
        if not chunk:
            break
        state = _get_yts_live_state(command_id)
        if not state:
            return
        text = chunk.decode(errors="replace")
        cleaned_text = _strip_terminal_ansi(text).rstrip("\n")
        state[stream_name] += text
        state["logs"].append({
            "stream": stream_name,
            "message": cleaned_text,
        })
        stripped = cleaned_text.strip()
        option_value = _parse_yts_prompt_option(stripped)
        if _is_interactive_yts_prompt(stripped) or option_value:
            prompt_entry = None
            pending_prompt = state.get("pending_prompt")
            if state.get("awaiting_input") and isinstance(pending_prompt, dict) and not pending_prompt.get("answered"):
                prompt_entry = pending_prompt
            if prompt_entry is None:
                prompt_entry = {
                    "id": len(state["prompts"]) + 1,
                    "text": "",
                    "options": [],
                    "stream": stream_name,
                    "answered": False,
                }
                state["prompts"].append(prompt_entry)
                state["pending_prompt"] = prompt_entry
                state["awaiting_input"] = True

            _merge_yts_prompt_entry(prompt_entry, stripped, stream_name)
            _persist_yts_live_state(state)

            if state.get("interactive_ai") and _prompt_ready_for_ai_response(prompt_entry) and not prompt_entry.get("answered") and not prompt_entry.get("ai_suggestion"):
                prompt_id = int(prompt_entry["id"])
                try:
                    suggestion = await _suggest_yts_prompt_response(command_id, prompt_entry.get("text", ""), prompt_entry.get("options") or [])
                    _update_yts_prompt_entry(
                        command_id,
                        prompt_id,
                        ai_suggestion=suggestion.get("response"),
                        ai_source=suggestion.get("source"),
                        ai_visual_summary=suggestion.get("visual_summary"),
                        ai_visual_source=suggestion.get("visual_source"),
                        ai_error=None,
                    )
                    if suggestion.get("response"):
                        current_state = _get_yts_live_state(command_id) or {}
                        current_pending = current_state.get("pending_prompt") if isinstance(current_state.get("pending_prompt"), dict) else None
                        if not current_state.get("awaiting_input") or current_pending is None or current_pending.get("id") != prompt_id or current_pending.get("answered"):
                            _update_yts_prompt_entry(command_id, prompt_id, ai_error="Prompt closed before AI response could be sent")
                            continue
                        await _send_yts_command_input(command_id, suggestion["response"], source=suggestion["source"])
                except HTTPException as exc:
                    if exc.status_code == 409:
                        _update_yts_prompt_entry(command_id, prompt_id, ai_error=exc.detail)
                    else:
                        logger.exception("Unable to auto-answer YTS prompt for %s", command_id)
                        _update_yts_prompt_entry(command_id, prompt_id, ai_error=str(exc))
                except Exception as exc:
                    logger.exception("Unable to auto-answer YTS prompt for %s", command_id)
                    _update_yts_prompt_entry(command_id, prompt_id, ai_error=str(exc))
            continue
        _persist_yts_live_state(state)


async def _run_yts_command_live(command_id: str, request: "YtsCommandRequest") -> None:
    state = _yts_live_commands[command_id]
    cmd = _build_yts_command(request)
    state["command"] = " ".join(cmd)
    if state.get("record_video"):
        state["video_recording_status"] = "pending"
    _persist_yts_live_state(state)

    recording_started = False
    if state.get("record_video"):
        await _start_yts_video_recording(command_id)
        state = _get_yts_live_state(command_id) or state
        recording_started = state.get("video_recording_status") == "recording"

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd="/home/harry/youtube/ai_tool",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        state["status"] = "failed"
        state["returncode"] = 127
        state["stderr"] = "yts binary not found in PATH"
        state["logs"].append({"stream": "stderr", "message": state["stderr"]})
        if state.get("record_video"):
            await _stop_yts_video_recording(command_id)
            state = _get_yts_live_state(command_id) or state
            if state.get("video_recording_status") == "recording":
                state["video_recording_status"] = "stopped"
        await asyncio.to_thread(_write_yts_terminal_log_artifact, state)
        _persist_yts_live_state(state)
        return

    _yts_live_processes[command_id] = process

    stdout_task = asyncio.create_task(_append_yts_stream_output(command_id, "stdout", process.stdout))
    stderr_task = asyncio.create_task(_append_yts_stream_output(command_id, "stderr", process.stderr))

    try:
        returncode = await process.wait()
        await asyncio.gather(stdout_task, stderr_task)
        state["returncode"] = returncode
        state["status"] = "completed" if returncode == 0 else "failed"

        if request.output_file and Path(request.output_file).exists():
            try:
                output_path = Path(request.output_file)
                state["result_file_content"] = output_path.read_text(encoding="utf-8")
                state["result_file_name"] = output_path.name
                output_path.unlink()
            except Exception as exc:
                msg = f"Error reading result file: {exc}"
                state["stderr"] += ("\n" if state["stderr"] else "") + msg
                state["logs"].append({"stream": "stderr", "message": msg})
        if state.get("record_video"):
            await _stop_yts_video_recording(command_id)
            state = _get_yts_live_state(command_id) or state
            video_path_raw = state.get("video_file_path")
            video_path = Path(str(video_path_raw)) if video_path_raw else None
            if recording_started and video_path and video_path.exists():
                state["video_recording_status"] = "completed"
            elif state.get("video_recording_status") == "recording":
                state["video_recording_status"] = "stopped"
        await asyncio.to_thread(_write_yts_terminal_log_artifact, state)
        _persist_yts_live_state(state)
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
            await process.wait()
        state["status"] = "stopped"
        state["returncode"] = process.returncode
        if state.get("record_video"):
            await _stop_yts_video_recording(command_id)
            state = _get_yts_live_state(command_id) or state
            video_path_raw = state.get("video_file_path")
            video_path = Path(str(video_path_raw)) if video_path_raw else None
            if recording_started and video_path and video_path.exists():
                state["video_recording_status"] = "completed"
            elif state.get("video_recording_status") == "recording":
                state["video_recording_status"] = "stopped"
        await asyncio.to_thread(_write_yts_terminal_log_artifact, state)
        _persist_yts_live_state(state)
        raise
    finally:
        _yts_live_processes.pop(command_id, None)
        visual_task = _yts_live_visual_tasks.get(command_id)
        if visual_task and not visual_task.done():
            visual_task.cancel()


def _find_active_run_id() -> Optional[str]:
    """Return a currently active run id, if any."""
    for run_id, state in _runs.items():
        if state.status in {RunStatus.PENDING, RunStatus.RUNNING}:
            return run_id
    return None


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

        timezone_value = _extract_setting_value(chunk, r"time\s*zone|timezone")
        if timezone_value:
            actions.append(
                ManualActionRequest(
                    action="SET_SETTING",
                    params={"key": "timezone", "value": timezone_value},
                )
            )
            continue

        language_value = _extract_language_setting_value(chunk)
        if language_value:
            actions.append(
                ManualActionRequest(
                    action="SET_SETTING",
                    params={"key": "language", "value": language_value},
                )
            )
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
    await asyncio.to_thread(_mark_stale_yts_live_commands)
    await asyncio.to_thread(_refresh_yts_test_catalog, _YTS_TESTLIST_PATH, False, False)
    await asyncio.to_thread(_refresh_yts_test_catalog, _YTS_GUIDED_TESTLIST_PATH, True, False)
    await _maybe_start_livekit_agent()
    try:
        yield
    finally:
        for command_id, task in list(_yts_live_visual_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            _yts_live_visual_tasks.pop(command_id, None)
        for command_id, task in list(_yts_live_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            _yts_live_tasks.pop(command_id, None)
        await _stop_livekit_agent()
        await asyncio.to_thread(_close_yts_live_db)


app.router.lifespan_context = _lifespan


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

def _frontend_no_cache_headers() -> dict:
    return {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }


def _frontend_file_response(filename: str, media_type: Optional[str] = None) -> FileResponse:
    path = _STATIC_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Frontend asset {filename!r} not found")
    return FileResponse(str(path), media_type=media_type, headers=_frontend_no_cache_headers())

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the bundled browser demo."""
    return _frontend_file_response("index.html")


@app.get("/app.js", include_in_schema=False)
async def serve_frontend_app_js() -> FileResponse:
    return _frontend_file_response("app.js", media_type="application/javascript")


@app.get("/styles.css", include_in_schema=False)
async def serve_frontend_styles() -> FileResponse:
    return _frontend_file_response("styles.css", media_type="text/css")


@app.get("/config.js", include_in_schema=False)
async def serve_frontend_config() -> FileResponse:
    """Serve the frontend config bootstrap script."""
    return _frontend_file_response("config.js", media_type="application/javascript")


@app.get("/testlist.json", include_in_schema=False)
async def serve_testlist(guided: bool = False) -> FileResponse:
    """Serve the testlist.json file generated by YTS CLI."""
    testlist_path = _catalog_path_for_mode(guided)
    if not testlist_path.exists():
        tests = await asyncio.to_thread(_refresh_yts_test_catalog, testlist_path, guided, False)
        if not tests:
            raise HTTPException(status_code=404, detail="testlist.json not found. Run 'yts list --json-output=testlist.json' to generate it.")
    return FileResponse(str(testlist_path))


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


@app.get("/capture/devices", response_model=dict)
async def capture_devices() -> dict:
    """List available /dev/video* devices with kind/readability diagnostics."""
    status = get_screen_capture().capture_source_status()
    return {
        "configured_source": status.get("configured_source"),
        "selected_video_device": status.get("selected_video_device"),
        "preferred_video_kind": status.get("preferred_video_kind"),
        "devices": status.get("video_device_details", []),
    }


@app.post("/capture/select", response_model=CaptureSourceResponse)
async def capture_select(request: CaptureSelectRequest) -> CaptureSourceResponse:
    """Select capture source and /dev/video device (HDMI card or camera)."""
    try:
        status = get_screen_capture().set_capture_preference(
            source=request.source,
            device=request.device,
            preferred_kind=request.preferred_kind,
            persist=bool(request.persist),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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


@app.get("/stream/status", response_model=dict)
async def stream_status() -> dict:
    """Return a consolidated status report for video and audio streaming."""
    video_status = get_screen_capture().capture_source_status()
    audio_status = await audio_source()
    return {
        "video": video_status,
        "audio": audio_status,
    }


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@app.post("/run/start", response_model=StartRunResponse)
async def start_run(request: StartRunRequest) -> StartRunResponse:
    """Start a new automation run."""
    active_run_id = _find_active_run_id()
    if active_run_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "Another run is already active "
                f"(run_id={active_run_id}). Wait for it to finish before starting a new run."
            ),
        )

    run_id = str(uuid.uuid4())
    state = RunState(run_id=run_id, goal=request.goal)
    if request.app_id:
        state.current_app = _validate_app_id(request.app_id)
    if request.device_profile_id:
        state.device_profile_id = str(request.device_profile_id).strip()
    if request.policy_mode:
        state.hybrid_policy_mode = str(request.policy_mode).strip()
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


@app.post("/yts/job", response_model=StartRunResponse)
async def yts_create_job(request: StartRunRequest) -> StartRunResponse:
    """Create a YTS job (alias for /run/start)."""
    return await start_run(request)


@app.get("/yts/jobs", response_model=List[RunSummaryItem])
async def yts_list_jobs() -> List[RunSummaryItem]:
    """List YTS jobs (alias for /runs)."""
    return await list_runs()


@app.get("/yts/job/{run_id}", response_model=RunStatusResponse)
async def yts_job_status(run_id: str) -> RunStatusResponse:
    """Get YTS job status (alias for /run/{run_id}/status)."""
    return await get_run_status(run_id)


@app.post("/yts/job/{run_id}/stop")
async def yts_job_stop(run_id: str) -> dict:
    """Stop a YTS job (alias for /run/{run_id}/stop)."""
    return await stop_run(run_id)


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


def _run_yts_command(args_list, cwd=None):
    try:
        result = subprocess.run(
            args_list,
            capture_output=True,
            text=True,
            cwd=cwd or '/home/harry/youtube/ai_tool',
            check=False,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail='yts binary not found in PATH')

    return {
        'returncode': result.returncode,
        'stdout': result.stdout.strip(),
        'stderr': result.stderr.strip(),
    }


@app.get("/yts/discover")
async def yts_discover() -> dict:
    """Run YTS discover and return discovered devices."""
    # --list gives previously discovered devices and new discovery.
    res = _run_yts_command(['yts', 'discover', '--list'])
    return res


@app.get("/yts/tests")
async def yts_list_tests(guided: bool = False) -> List[Dict[str, str]]:
    """Return the stored YTS test catalog, refreshing it if needed."""
    path = _catalog_path_for_mode(guided)
    tests = _read_yts_test_catalog(path)
    if tests:
        return tests
    return await asyncio.to_thread(_refresh_yts_test_catalog, path, guided, True)


@app.post("/yts/tests/refresh")
async def yts_refresh_tests(guided: bool = False) -> List[Dict[str, str]]:
    """Refresh the stored YTS test catalog and return the latest tests."""
    path = _catalog_path_for_mode(guided)
    return await asyncio.to_thread(_refresh_yts_test_catalog, path, guided, True)


class TestRequest(BaseModel):
    device: str
    test: str
    filters: Optional[List[str]] = None
    json_output: Optional[str] = None
    args: Optional[List[str]] = None

class YtsCommandRequest(BaseModel):
    command: str
    params: List[str] = []
    global_options: Dict[str, Union[str, bool]] = {}
    output_file: Optional[str] = None
    interactive_ai: bool = False
    record_video: bool = False


class YtsInteractiveResponseRequest(BaseModel):
    response: str


class YtsInteractiveSuggestRequest(BaseModel):
    send_response: bool = False


@app.post("/yts/command/live")
async def yts_live_command(request: YtsCommandRequest) -> dict:
    """Start a YTS command and capture live stdout/stderr for polling."""
    command_id = str(uuid.uuid4())
    state = _new_yts_live_state(command_id, bool(request.interactive_ai))
    state["record_video"] = bool(request.record_video)
    state["video_recording_status"] = "pending" if request.record_video else "disabled"
    _yts_live_commands[command_id] = state
    _write_yts_terminal_log_artifact(state)
    _persist_yts_live_state(state)
    if request.interactive_ai:
        _yts_live_visual_tasks[command_id] = asyncio.create_task(_run_yts_live_visual_monitor(command_id))
    _yts_live_tasks[command_id] = asyncio.create_task(_run_yts_command_live(command_id, request))
    return {
        "command_id": command_id,
        "status": state["status"],
    }


@app.get("/yts/command/live")
async def list_yts_live_commands(limit: int = 10, active_only: bool = False) -> List[dict]:
    return await asyncio.to_thread(_list_yts_live_states, limit, active_only)


@app.get("/yts/command/live/{command_id}/terminal-log")
async def download_yts_terminal_log(command_id: str) -> FileResponse:
    state = _get_yts_live_state(command_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"YTS command {command_id!r} not found")
    path = await asyncio.to_thread(_write_yts_terminal_log_artifact, state)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Terminal log not available")
    return FileResponse(str(path), media_type="text/plain", filename=path.name)


@app.get("/yts/command/live/{command_id}/video")
async def download_yts_video_recording(command_id: str) -> FileResponse:
    state = _get_yts_live_state(command_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"YTS command {command_id!r} not found")
    path_raw = state.get("video_file_path")
    path = Path(str(path_raw)) if path_raw else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Recorded video not available")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


@app.get("/yts/command/live/{command_id}/result")
async def download_yts_result_file(command_id: str) -> Response:
    state = _get_yts_live_state(command_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"YTS command {command_id!r} not found")
    result_content = state.get("result_file_content")
    if not result_content:
        raise HTTPException(status_code=404, detail="Saved result file not available")
    filename = str(state.get("result_file_name") or f"yts-result-{command_id}.json")
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=str(result_content), media_type="application/octet-stream", headers=headers)


@app.get("/yts/command/live/{command_id}")
async def get_yts_live_command(command_id: str) -> dict:
    state = _get_yts_live_state(command_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"YTS command {command_id!r} not found")
    return state


@app.post("/yts/command/live/{command_id}/stop")
async def stop_yts_live_command(command_id: str) -> dict:
    state = _get_yts_live_state(command_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"YTS command {command_id!r} not found")

    process = _yts_live_processes.get(command_id)
    if process and process.returncode is None:
        process.terminate()

    task = _yts_live_tasks.get(command_id)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    visual_task = _yts_live_visual_tasks.get(command_id)
    if visual_task and not visual_task.done():
        visual_task.cancel()
        try:
            await visual_task
        except asyncio.CancelledError:
            pass

    state["status"] = "stopped"
    state["visual_monitor_active"] = False
    _persist_yts_live_state(state)
    return {
        "command_id": command_id,
        "status": state["status"],
    }


@app.post("/yts/command/live/{command_id}/respond")
async def respond_yts_live_command(command_id: str, request: YtsInteractiveResponseRequest) -> dict:
    state = _get_yts_live_state(command_id)
    prompt_entry = None
    if state:
        prompt_entry = state.get("pending_prompt") or (state.get("prompts")[-1] if state.get("prompts") else None)
    result = await _send_yts_command_input(command_id, request.response, source="manual")
    if prompt_entry and prompt_entry.get("id") is not None:
        _update_yts_prompt_entry(
            command_id,
            int(prompt_entry["id"]),
            answered=True,
            response=request.response,
            response_source="manual",
            ai_error=None,
        )
    return result


@app.post("/yts/command/live/{command_id}/suggest")
async def suggest_yts_live_command_response(command_id: str, request: YtsInteractiveSuggestRequest) -> dict:
    state = _get_yts_live_state(command_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"YTS command {command_id!r} not found")
    prompt_entry = state.get("pending_prompt") or (state.get("prompts")[-1] if state.get("prompts") else None)
    if not prompt_entry:
        raise HTTPException(status_code=409, detail="No interactive prompt is waiting for input")
    if not _prompt_ready_for_ai_response(prompt_entry):
        raise HTTPException(status_code=409, detail="YTS prompt is still collecting the full question/options")
    prompt_id = int(prompt_entry["id"])

    suggestion = await _suggest_yts_prompt_response(command_id, prompt_entry.get("text", ""), prompt_entry.get("options") or [])
    _update_yts_prompt_entry(
        command_id,
        prompt_id,
        ai_suggestion=suggestion["response"],
        ai_source=suggestion["source"],
        ai_visual_summary=suggestion.get("visual_summary"),
        ai_visual_source=suggestion.get("visual_source"),
        ai_error=None,
    )
    if request.send_response:
        await _send_yts_command_input(command_id, suggestion["response"], source=suggestion["source"])
        _update_yts_prompt_entry(
            command_id,
            prompt_id,
            answered=True,
            response=suggestion["response"],
            response_source=suggestion["source"],
            ai_error=None,
        )
    return {
        "command_id": command_id,
        "suggestion": suggestion["response"],
        "source": suggestion["source"],
        "sent": bool(request.send_response),
    }

@app.post("/yts/command")
async def yts_generic_command(request: YtsCommandRequest) -> dict:
    """Execute a generic YTS command."""
    cmd = _build_yts_command(request)

    res = _run_yts_command(cmd)

    output = {
        'command': ' '.join(cmd),
        'returncode': res['returncode'],
        'stdout': res['stdout'],
        'stderr': res['stderr'],
        'result_file_content': None,
        'result_file_name': None,
    }

    if request.output_file and Path(request.output_file).exists():
        try:
            output['result_file_content'] = Path(request.output_file).read_text(encoding='utf-8')
            output['result_file_name'] = Path(request.output_file).name
            Path(request.output_file).unlink() # Clean up the file
        except Exception as e:
            output['stderr'] += f"\nError reading result file: {e}"

    return output


@app.post("/yts/test")
async def yts_test(request: TestRequest) -> dict:
    """Execute a YTS test using the official yts CLI."""
    # Ensure discover has happened
    discover_res = _run_yts_command(['yts', 'discover', '--list'])
    if discover_res['returncode'] != 0:
        raise HTTPException(status_code=500, detail=f"YTS discover failed: {discover_res['stderr']}")

    result_file = Path(request.json_output or '/tmp/yts_test_result.json')
    result_file.unlink(missing_ok=True)

    cmd = ['yts', 'test', request.device, request.test]
    if request.filters:
        cmd.extend(request.filters)
    if request.args:
        cmd.extend(request.args)
    if request.json_output:
        cmd.extend(['--json-output', str(result_file)])
    
    res = _run_yts_command(cmd)

    output = {
        'command': ' '.join(cmd),
        'returncode': res['returncode'],
        'stdout': res['stdout'],
        'stderr': res['stderr'],
    }

    if result_file.exists():
        try:
            output['details'] = json.loads(result_file.read_text(encoding='utf-8'))
        except Exception:
            output['details'] = result_file.read_text(encoding='utf-8')

    return output


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
        device_profile_id=state.device_profile_id,
        hybrid_policy_mode=state.hybrid_policy_mode,
        hybrid_policy_rationale=state.hybrid_policy_rationale,
        retrieved_experiences=state.retrieved_experiences[-5:],
        observation_features=state.observation_features,
        local_action_suggestions=state.local_action_suggestions,
        local_model_version=state.local_model_version,
    )


@app.get("/run/{run_id}/history", response_model=ActionHistoryResponse)
async def get_run_history(run_id: str) -> ActionHistoryResponse:
    """Get action history for a run."""
    state = _runs.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return ActionHistoryResponse(
        run_id=state.run_id,
        goal=state.goal,
        action_count=len(state.action_history),
        actions=[ActionRecordItem(**r.model_dump()) for r in state.action_history],
    )


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
        device_profile_id=state.device_profile_id,
        hybrid_policy_mode=state.hybrid_policy_mode,
        hybrid_policy_rationale=state.hybrid_policy_rationale,
        retrieved_experiences=state.retrieved_experiences[-5:],
        observation_features=state.observation_features,
        local_action_suggestions=state.local_action_suggestions,
        local_model_version=state.local_model_version,
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
            # Offload the blocking frame capture to a separate thread
            frame = await asyncio.to_thread(
                capture.get_hdmi_stream_frame_jpeg, quality=config.hdmi_stream_jpeg_quality
            )
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
    # MP3 output for browser playback always requires ffmpeg (including arecord fallback path).
    if not ffmpeg_available():
        raise HTTPException(status_code=500, detail="ffmpeg not found on host; required for /stream/audio")

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

    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
        elif action == "SETTINGS_LIST":
            resp = await dab.list_settings()
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "GET_SETTING":
            setting_key = _normalize_setting_key(str(params.get("key") or params.get("setting_key") or ""))
            if not setting_key:
                raise HTTPException(status_code=400, detail="key is required for GET_SETTING")
            resp = await dab.get_setting(setting_key)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "SET_SETTING":
            setting_key = _normalize_setting_key(str(params.get("key") or params.get("setting_key") or ""))
            if not setting_key:
                raise HTTPException(status_code=400, detail="key is required for SET_SETTING")
            if "value" not in params:
                raise HTTPException(status_code=400, detail="value is required for SET_SETTING")
            resp = await dab.set_setting(setting_key, params.get("value"))
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


class TestRequest(BaseModel):
    device: str
    test: str


@app.get("/yts/discover")
async def yts_discover():
    """Run YTS discover command."""
    result = subprocess.run(
        ['python3', 'yts.py', 'discover'],
        capture_output=True,
        text=True,
        cwd='/home/harry/youtube/ai_tool'
    )
    return {'output': result.stdout.strip(), 'error': result.stderr.strip()}


@app.get("/yts/list")
async def yts_list():
    """Run YTS list command."""
    result = subprocess.run(
        ['python3', 'yts.py', 'list'],
        capture_output=True,
        text=True,
        cwd='/home/harry/youtube/ai_tool'
    )
    return {'output': result.stdout.strip(), 'error': result.stderr.strip()}


@app.post("/yts/test")
async def yts_test(request: TestRequest) -> dict:
    """Run YTS test command."""
    cmd = ['python3', 'yts.py', 'test', request.device, request.test]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd='/home/harry/youtube/ai_tool'
    )
    return {'output': result.stdout.strip(), 'error': result.stderr.strip()}


@app.get("/dab/apps", response_model=dict)
async def dab_apps() -> dict:
    """Return installed/launchable application list from device."""
    resp = await get_dab_client().list_apps()
    return {"success": resp.success, "result": resp.data}
