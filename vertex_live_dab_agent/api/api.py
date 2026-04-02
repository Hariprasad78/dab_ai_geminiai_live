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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from vertex_live_dab_agent.android_timezone import (
    get_timezone_via_adb,
    is_adb_device_online,
    list_timezones_via_adb,
    resolve_timezone_from_supported,
    set_timezone_via_adb,
)
from vertex_live_dab_agent.api.models import (
    AITranscriptResponse,
    DABTranscriptResponse,
    ActionHistoryResponse,
    ActionRecordItem,
    CaptureSelectRequest,
    CaptureSourceResponse,
    ConfigSummaryResponse,
    RuntimeModelResponse,
    RuntimeModelUpdateRequest,
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
from vertex_live_dab_agent.dab.client import DABClientBase, DABError, create_dab_client
from vertex_live_dab_agent.dab.topics import KEY_MAP
from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState, RunStatus
from vertex_live_dab_agent.planner.planner import Planner
from vertex_live_dab_agent.planner.vertex_client import VertexPlannerClient
from vertex_live_dab_agent.system_ops.routing import (
    has_android_adb_fallback,
    operation_supported_by_dab,
    resolve_execution_method,
)
from vertex_live_dab_agent.system_ops.device_detection import get_device_platform_info

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
_runtime_vertex_planner_model_override: Optional[str] = None
_yts_live_commands: Dict[str, Dict[str, Any]] = {}
_yts_live_tasks: Dict[str, asyncio.Task] = {}
_yts_live_visual_tasks: Dict[str, asyncio.Task] = {}
_yts_live_processes: Dict[str, asyncio.subprocess.Process] = {}
_yts_live_recording_processes: Dict[str, Dict[str, Any]] = {}
_yts_live_visual_cache: Dict[str, Dict[str, Any]] = {}
_yts_live_db_conn: Optional[sqlite3.Connection] = None
_yts_live_db_path: Optional[Path] = None
_yts_live_db_lock = threading.Lock()
_yts_discover_cache: List[Dict[str, str]] = []
_yts_discover_cache_at: float = 0.0
_selected_device_id_override: Optional[str] = None
_discovered_devices_cache: List[Dict[str, Any]] = []
_discovered_devices_cache_at: float = 0.0
_discovery_warning_cache: Optional[str] = None
_discover_devices_in_flight: Optional[asyncio.Task] = None
_device_capabilities_cache: Dict[str, Any] = {}
_device_capabilities_cache_at: float = 0.0
_device_dab_catalog_cache: Dict[str, Dict[str, Any]] = {}
_device_dab_catalog_inflight: Dict[str, asyncio.Task] = {}
_device_settings_values_cache: Dict[str, Dict[str, Any]] = {}
_device_settings_values_inflight: Dict[str, asyncio.Task] = {}
_device_settings_values_last_request_at: Dict[str, float] = {}
_device_dab_catalog_ttl_seconds: float = 120.0
_device_settings_values_ttl_seconds: float = 15.0
_device_settings_values_min_interval_seconds: float = 15.0
_device_settings_get_max_concurrency: int = 4
_device_settings_get_semaphore = asyncio.Semaphore(_device_settings_get_max_concurrency)

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

_COMMON_VERTEX_MODELS: List[str] = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-live-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


def _shared_ai_prompt_preamble() -> str:
    """Return shared prompt style used across orchestrator planner and YTS flows."""
    try:
        planner = get_planner()
        text = str(getattr(planner, "_planner_system_prompt", "") or "").strip()
        if text:
            return text
    except Exception:
        pass
    return (
        "You are a TV UI navigation planner. "
        "Use screenshot-grounded evidence, stay concise, and avoid speculation. "
        "For uncertain UI navigation, request a UI checkpoint capture before any commit/select action. "
        "Return machine-readable output when requested."
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_context_path() -> Path:
    base_dir = os.getenv("ARTIFACTS_BASE_DIR")
    root = Path(base_dir) if base_dir else Path("/home/harry/youtube/ai_tool/artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root / "device_context.json"


def _device_capabilities_cache_path() -> Path:
    base_dir = os.getenv("ARTIFACTS_BASE_DIR")
    root = Path(base_dir) if base_dir else Path("/home/harry/youtube/ai_tool/artifacts")
    root.mkdir(parents=True, exist_ok=True)
    return root / "device_capabilities_cache.json"


def _device_system_state_path(device_id: str) -> Path:
    base_dir = os.getenv("ARTIFACTS_BASE_DIR")
    root = Path(base_dir) if base_dir else Path("/home/harry/youtube/ai_tool/artifacts")
    root.mkdir(parents=True, exist_ok=True)
    safe_device_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(device_id or "unknown").strip())
    safe_device_id = safe_device_id.strip("._-") or "unknown"
    return root / f"device_system_state_{safe_device_id}.json"


def _normalize_settings_entries_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize settings/list payloads that may be list-based or key-value objects."""
    if not isinstance(payload, dict):
        return []

    settings_entries: List[Dict[str, Any]] = []
    raw_settings = payload.get("settings")

    if isinstance(raw_settings, list):
        for item in raw_settings:
            if isinstance(item, dict):
                key = str(item.get("key") or item.get("name") or item.get("id") or "").strip()
                if not key:
                    continue
                row = dict(item)
                row["key"] = key
                settings_entries.append(row)
            elif isinstance(item, str):
                key = str(item).strip()
                if key:
                    settings_entries.append({"key": key, "friendlyName": key})
        return settings_entries

    ignored = {
        "status",
        "error",
        "warning",
        "degraded",
        "requestId",
        "request_id",
        "topic",
    }
    for key, value in payload.items():
        k = str(key or "").strip()
        if not k or k in ignored:
            continue
        row: Dict[str, Any] = {
            "key": k,
            "friendlyName": k.replace("_", " ").title(),
            "writable": True,
            "schema": value,
        }
        if isinstance(value, dict):
            if "min" in value:
                row["min"] = value.get("min")
            if "max" in value:
                row["max"] = value.get("max")
            if "values" in value and isinstance(value.get("values"), list):
                row["allowedValues"] = value.get("values")
        elif isinstance(value, list):
            row["allowedValues"] = value
        settings_entries.append(row)
    return settings_entries


def _extract_setting_value(data: Dict[str, Any], key: str) -> Any:
    value = data.get("value")
    if value is not None:
        return value
    if key in data:
        return data.get(key)
    if isinstance(data.get("result"), dict):
        return data.get("result", {}).get("value")
    return None


def _build_settings_read_error(resp_success: bool, data: Dict[str, Any]) -> Optional[str]:
    if resp_success:
        return None
    return str(data.get("error") or "settings/get failed")


def _extract_bulk_settings_map(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    direct_settings = data.get("settings")
    if isinstance(direct_settings, dict):
        return {str(k): v for k, v in direct_settings.items()}
    if isinstance(direct_settings, list):
        out: Dict[str, Any] = {}
        for item in direct_settings:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("name") or "").strip()
            if not key:
                continue
            if "value" in item:
                out[key] = item.get("value")
            elif key in item:
                out[key] = item.get(key)
        if out:
            return out

    values = data.get("values")
    if isinstance(values, dict):
        return {str(k): v for k, v in values.items()}
    if isinstance(values, list):
        out: Dict[str, Any] = {}
        for item in values:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("name") or "").strip()
            if not key:
                continue
            if "value" in item:
                out[key] = item.get("value")
        if out:
            return out

    ignored = {
        "success",
        "status",
        "error",
        "message",
        "request_id",
        "requestId",
        "device_id",
        "deviceId",
        "captured_at",
        "timestamp",
        "topic",
        "values",
        "settings",
    }
    fallback: Dict[str, Any] = {}
    for key, value in data.items():
        k = str(key or "").strip()
        if not k or k in ignored:
            continue
        fallback[k] = value
    return fallback


def _normalize_operation_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _operation_is_supported(supported_operations: List[str], candidates: List[str]) -> bool:
    normalized = {_normalize_operation_name(item) for item in (supported_operations or []) if str(item or "").strip()}
    if not normalized:
        return False
    return any(_normalize_operation_name(candidate) in normalized for candidate in candidates)


def _get_cached_payload(cache: Dict[str, Dict[str, Any]], device_id: str) -> Optional[Dict[str, Any]]:
    entry = cache.get(device_id)
    if not isinstance(entry, dict):
        return None
    payload = entry.get("payload")
    return dict(payload) if isinstance(payload, dict) else None


async def _safe_dab_call(label: str, callable_obj) -> Dict[str, Any]:
    try:
        resp = await callable_obj()
        data = dict(resp.data or {}) if isinstance(getattr(resp, "data", None), dict) else {}
        return {
            "success": bool(resp.success),
            "status": int(getattr(resp, "status", 0) or 0),
            "error": None if resp.success else str(data.get("error") or f"{label} failed"),
            "data": data,
        }
    except Exception as exc:
        return {
            "success": False,
            "status": 0,
            "error": str(exc),
            "data": {},
        }


async def _refresh_device_dab_catalog(device_id: str) -> Dict[str, Any]:
    logger.info("DAB catalog refresh start: device=%s", device_id)
    dab = get_dab_client()
    operations_result = await _safe_dab_call("operations/list", dab.list_operations)
    operations = operations_result["data"].get("operations")
    if not isinstance(operations, list):
        operations = []

    ops_truth_available = bool(operations_result["success"] and operations)

    async def _fetch_list_with_support(
        operation_name: str,
        candidates: List[str],
        fn,
    ) -> Dict[str, Any]:
        if ops_truth_available and not _operation_is_supported(operations, candidates):
            return {
                "success": False,
                "status": 0,
                "error": f"{operation_name} unsupported by operations/list",
                "data": {},
                "unsupported": True,
            }
        result = await _safe_dab_call(operation_name, fn)
        result["unsupported"] = False
        return result

    keys_result = await _fetch_list_with_support("input/key/list", ["input/key/list"], dab.list_keys)
    apps_result = await _fetch_list_with_support("applications/list", ["applications/list", "application/list"], dab.list_apps)
    settings_result = await _fetch_list_with_support("system/settings/list", ["system/settings/list"], dab.list_settings)
    voice_result = await _fetch_list_with_support("voice/list", ["voice/list", "voices/list"], dab.list_voices)

    keys_payload = dict(keys_result.get("data") or {})
    keys = keys_payload.get("keyCodes")
    if not isinstance(keys, list):
        keys = keys_payload.get("keys")
    if not isinstance(keys, list):
        keys = keys_payload.get("supportedKeys")
    if not isinstance(keys, list):
        keys = []
    apps_payload = dict(apps_result.get("data") or {})
    apps = apps_payload.get("apps")
    if not isinstance(apps, list):
        apps = apps_payload.get("applications")
    if not isinstance(apps, list):
        apps = apps_payload.get("list")
    if not isinstance(apps, list):
        apps = []
    voice_payload = dict(voice_result.get("data") or {})
    voices = voice_payload.get("voices")
    if not isinstance(voices, list):
        voices = voice_payload.get("voiceSystems")
    if not isinstance(voices, list):
        voices = voice_payload.get("list")
    if not isinstance(voices, list):
        voices = []

    raw_settings_payload = settings_result["data"] if isinstance(settings_result.get("data"), dict) else {}
    settings_list = _normalize_settings_entries_from_payload(raw_settings_payload)

    payload = {
        "device_id": device_id,
        "captured_at": _utc_now_iso(),
        "operations": {
            "status": operations_result["status"],
            "success": operations_result["success"],
            "error": operations_result["error"],
            "list": operations,
            "raw_payload": dict(operations_result.get("data") or {}),
        },
        "keys": {
            "status": keys_result["status"],
            "success": keys_result["success"],
            "error": keys_result["error"],
            "list": keys,
            "raw_payload": dict(keys_result.get("data") or {}),
            "unsupported": bool(keys_result.get("unsupported")),
        },
        "applications_list": {
            "status": apps_result["status"],
            "success": apps_result["success"],
            "error": apps_result["error"],
            "list": apps,
            "raw_payload": dict(apps_result.get("data") or {}),
            "unsupported": bool(apps_result.get("unsupported")),
        },
        "voice_list": {
            "status": voice_result["status"],
            "success": voice_result["success"],
            "error": voice_result["error"],
            "list": voices,
            "raw_payload": dict(voice_result.get("data") or {}),
            "unsupported": bool(voice_result.get("unsupported")),
        },
        "settings_list": {
            "status": settings_result["status"],
            "success": settings_result["success"],
            "error": settings_result["error"],
            "warning": raw_settings_payload.get("warning"),
            "degraded": bool(raw_settings_payload.get("degraded")),
            "list": settings_list,
            "raw_payload": dict(raw_settings_payload),
            "unsupported": bool(settings_result.get("unsupported")),
        },
    }

    _device_dab_catalog_cache[device_id] = {
        "cached_at": time.monotonic(),
        "payload": payload,
    }
    logger.info(
        "DAB catalog refresh end: device=%s ops=%s keys=%s settings=%s apps=%s voices=%s",
        device_id,
        len(operations),
        len(keys),
        len(settings_list),
        len(apps),
        len(voices),
    )
    return payload


async def _get_device_dab_catalog_cached(device_id: str, *, force: bool = False) -> Dict[str, Any]:
    now = time.monotonic()
    cached_entry = _device_dab_catalog_cache.get(device_id)
    if (not force) and isinstance(cached_entry, dict):
        cached_at = float(cached_entry.get("cached_at") or 0.0)
        if (now - cached_at) <= max(0.0, float(_device_dab_catalog_ttl_seconds)):
            logger.info("DAB catalog cache hit: device=%s age=%.2fs", device_id, now - cached_at)
            payload = _get_cached_payload(_device_dab_catalog_cache, device_id)
            if payload is not None:
                return payload

    inflight = _device_dab_catalog_inflight.get(device_id)
    if inflight is not None and not inflight.done():
        payload = _get_cached_payload(_device_dab_catalog_cache, device_id)
        if payload is not None:
            logger.info("DAB catalog dedup hit (stale-safe): device=%s", device_id)
            return payload
        logger.info("DAB catalog dedup wait: device=%s", device_id)
        return await inflight

    logger.info("DAB catalog cache miss: device=%s force=%s", device_id, force)
    task = asyncio.create_task(_refresh_device_dab_catalog(device_id))
    _device_dab_catalog_inflight[device_id] = task
    try:
        return await task
    finally:
        current = _device_dab_catalog_inflight.get(device_id)
        if current is task:
            _device_dab_catalog_inflight.pop(device_id, None)


async def _persist_settings_values_to_snapshot(device_id: str, payload: Dict[str, Any]) -> None:
    snapshot_path = _device_system_state_path(device_id)
    snapshot: Dict[str, Any] = {}
    if snapshot_path.exists():
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            snapshot = loaded
    current_values = list(payload.get("values") or [])
    refresh_status = dict(snapshot.get("refresh_status") or {})
    refresh_status["settings_get"] = {
        "success": True,
        "status": 200,
        "error": None,
        "count": int(payload.get("count") or 0),
        "failed": int(payload.get("failed") or 0),
        "updated_at": str(payload.get("captured_at") or _utc_now_iso()),
    }
    snapshot.update(
        {
            "success": True,
            "device_id": device_id,
            "captured_at": str(payload.get("captured_at") or _utc_now_iso()),
            "last_updated": str(payload.get("captured_at") or _utc_now_iso()),
            "json_file": str(snapshot_path),
            "current_setting_values": current_values,
            "refresh_status": refresh_status,
            "settings_get": {
                "count": int(payload.get("count") or 0),
                "failed": int(payload.get("failed") or 0),
                "values": current_values,
            },
        }
    )
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


_DAB_OPERATION_GRID_DEFINITIONS: List[Dict[str, Any]] = [
    {"operation": "operations/list", "candidates": ["operations/list"], "metadata_source": "operations/list", "related_source": "supported_operations", "default_action": "OPERATIONS_LIST"},
    {"operation": "applications/list", "candidates": ["applications/list", "application/list"], "metadata_source": "applications/list", "related_source": "installed_applications", "default_action": "APPLICATIONS_LIST"},
    {"operation": "applications/launch", "candidates": ["applications/launch", "application/launch"], "metadata_source": "applications/list", "related_source": "installed_applications", "default_action": "LAUNCH_APP"},
    {"operation": "applications/launch-with-content", "candidates": ["applications/launch-with-content", "applications/launch"], "metadata_source": "applications/list", "related_source": "installed_applications", "default_action": "LAUNCH_APP"},
    {"operation": "applications/get-state", "candidates": ["applications/get-state", "application/get-state"], "metadata_source": "applications/list", "related_source": "installed_applications", "default_action": "GET_STATE"},
    {"operation": "applications/exit", "candidates": ["applications/exit", "application/exit"], "metadata_source": "applications/list", "related_source": "installed_applications", "default_action": "EXIT_APP"},
    {"operation": "voice/list", "candidates": ["voice/list", "voices/list"], "metadata_source": "voice/list", "related_source": "supported_voice_systems", "default_action": "VOICE_LIST"},
    {"operation": "system/settings/list", "candidates": ["system/settings/list"], "metadata_source": "system/settings/list", "related_source": "supported_settings", "default_action": "SETTINGS_LIST"},
    {"operation": "system/settings/get", "candidates": ["system/settings/get"], "metadata_source": "system/settings/list", "related_source": "current_setting_values", "default_action": "GET_SETTING"},
    {"operation": "system/settings/set", "candidates": ["system/settings/set"], "metadata_source": "system/settings/list", "related_source": "supported_settings", "default_action": "SET_SETTING"},
    {"operation": "input/key/list", "candidates": ["input/key/list"], "metadata_source": "input/key/list", "related_source": "supported_keys", "default_action": "KEY_LIST"},
    {"operation": "input/key-press", "candidates": ["input/key-press", "input/key/press"], "metadata_source": "input/key/list", "related_source": "supported_keys", "default_action": "KEY_PRESS_CODE"},
    {"operation": "input/long-key-press", "candidates": ["input/long-key-press", "input/long-key/press"], "metadata_source": "input/key/list", "related_source": "supported_keys", "default_action": "LONG_KEY_PRESS"},
]


def _build_device_capability_status_snapshot(
    device_id: str,
    catalog: Dict[str, Any],
    values_payload: Dict[str, Any],
    snapshot_path: Path,
) -> Dict[str, Any]:
    ops_section = dict(catalog.get("operations") or {})
    keys_section = dict(catalog.get("keys") or {})
    apps_section = dict(catalog.get("applications_list") or {})
    voices_section = dict(catalog.get("voice_list") or {})
    settings_section = dict(catalog.get("settings_list") or {})

    supported_operations = list(ops_section.get("list") or [])
    supported_keys = list(keys_section.get("list") or [])
    installed_applications = list(apps_section.get("list") or [])
    supported_voice_systems = list(voices_section.get("list") or [])
    supported_settings = list(settings_section.get("list") or [])
    current_setting_values = list(values_payload.get("values") or [])

    unsupported_or_missing_sections: List[Dict[str, Any]] = []
    section_checks = [
        ("operations/list", ops_section),
        ("input/key/list", keys_section),
        ("applications/list", apps_section),
        ("voice/list", voices_section),
        ("system/settings/list", settings_section),
        ("system/settings/get", {"success": True, "status": 200, "error": None} if current_setting_values else {"success": False, "status": 0, "error": "No settings/get values yet"}),
    ]
    for section_name, section_data in section_checks:
        unsupported = bool(section_data.get("unsupported"))
        success = bool(section_data.get("success"))
        if unsupported or not success:
            unsupported_or_missing_sections.append(
                {
                    "section": section_name,
                    "status": "unsupported" if unsupported else "missing_or_failed",
                    "reason": str(section_data.get("error") or ("Unsupported by operations/list" if unsupported else "No data")),
                }
            )

    refresh_status = {
        "catalog": {
            "updated_at": str(catalog.get("captured_at") or _utc_now_iso()),
            "success": bool(ops_section.get("success")),
            "error": ops_section.get("error"),
        },
        "settings_get": {
            "updated_at": str(values_payload.get("captured_at") or _utc_now_iso()),
            "success": True,
            "count": int(values_payload.get("count") or 0),
            "failed": int(values_payload.get("failed") or 0),
        },
    }

    snapshot = {
        "success": True,
        "device_id": device_id,
        "last_updated": str(values_payload.get("captured_at") or catalog.get("captured_at") or _utc_now_iso()),
        "supported_operations": supported_operations,
        "supported_keys": supported_keys,
        "installed_applications": installed_applications,
        "supported_voice_systems": supported_voice_systems,
        "supported_settings": supported_settings,
        "current_setting_values": current_setting_values,
        "unsupported_or_missing_sections": unsupported_or_missing_sections,
        "raw_list_payloads": {
            "operations/list": dict(ops_section.get("raw_payload") or {}),
            "input/key/list": dict(keys_section.get("raw_payload") or {}),
            "applications/list": dict(apps_section.get("raw_payload") or {}),
            "voice/list": dict(voices_section.get("raw_payload") or {}),
            "system/settings/list": dict(settings_section.get("raw_payload") or {}),
        },
        "refresh_status": refresh_status,
        "json_file": str(snapshot_path),
        # backward-compatible fields for existing UI consumers
        "captured_at": str(values_payload.get("captured_at") or catalog.get("captured_at") or _utc_now_iso()),
        "operations": ops_section,
        "keys": keys_section,
        "applications_list": apps_section,
        "voice_list": voices_section,
        "settings_list": settings_section,
        "settings_get": {
            "count": int(values_payload.get("count") or 0),
            "failed": int(values_payload.get("failed") or 0),
            "values": current_setting_values,
        },
    }
    return snapshot


def _build_operations_grid_rows(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    supported_operations = [str(v or "").strip() for v in (snapshot.get("supported_operations") or []) if str(v or "").strip()]
    current_values = list(snapshot.get("current_setting_values") or [])
    rows: List[Dict[str, Any]] = []
    for item in _DAB_OPERATION_GRID_DEFINITIONS:
        operation_name = str(item.get("operation") or "")
        candidates = [str(v or "") for v in (item.get("candidates") or [operation_name])]
        supported = _operation_is_supported(supported_operations, candidates)
        rows.append(
            {
                "operation": operation_name,
                "supported": supported,
                "metadata_source": item.get("metadata_source"),
                "related_source": item.get("related_source"),
                "default_action": item.get("default_action"),
                "related_count": (
                    len(current_values)
                    if item.get("related_source") == "current_setting_values"
                    else len(snapshot.get(item.get("related_source") or "") or [])
                ),
            }
        )
    for op in supported_operations:
        if any(_normalize_operation_name(op) == _normalize_operation_name(str(row.get("operation") or "")) for row in rows):
            continue
        rows.append(
            {
                "operation": op,
                "supported": True,
                "metadata_source": "operations/list",
                "related_source": "n/a",
                "default_action": None,
                "related_count": 0,
            }
        )
    return rows


async def _refresh_device_settings_values(device_id: str, *, settings_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    logger.info("DAB settings/get refresh start: device=%s", device_id)
    dab = get_dab_client()

    if settings_entries is None:
        cached_catalog_payload = _get_cached_payload(_device_dab_catalog_cache, device_id)
        settings_entries = list(((cached_catalog_payload or {}).get("settings_list") or {}).get("list") or [])

    if settings_entries is None or not settings_entries:
        try:
            snapshot_path = _device_system_state_path(device_id)
            if snapshot_path.exists():
                loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    settings_entries = list(loaded.get("supported_settings") or [])
                    if not settings_entries:
                        settings_entries = list((loaded.get("settings_list") or {}).get("list") or [])
        except Exception as exc:
            logger.warning("Failed to reuse stored settings/list for settings/get refresh: device=%s error=%s", device_id, exc)

    if settings_entries is None or not settings_entries:
        # Last resort: fetch catalog (this is expected only on first load / explicit refresh).
        catalog = await _get_device_dab_catalog_cached(device_id, force=False)
        settings_entries = list((catalog.get("settings_list") or {}).get("list") or [])

    async def _read_setting(entry: Dict[str, Any]) -> Dict[str, Any]:
        key = str(entry.get("key") or "").strip()
        row = dict(entry)
        row["key"] = key
        if not key:
            row.update({"read_success": False, "read_status": 0, "read_error": "invalid setting key", "current_value": None, "read_raw": {}})
            return row

        async with _device_settings_get_semaphore:
            try:
                resp = await dab.get_setting(key)
                data = dict(resp.data or {}) if isinstance(getattr(resp, "data", None), dict) else {}
                row.update(
                    {
                        "read_success": bool(resp.success),
                        "read_status": int(getattr(resp, "status", 0) or 0),
                        "read_error": _build_settings_read_error(bool(resp.success), data),
                        "current_value": _extract_setting_value(data, key),
                        "read_raw": data,
                    }
                )
                return row
            except Exception as exc:
                row.update({"read_success": False, "read_status": 0, "read_error": str(exc), "current_value": None, "read_raw": {}})
                return row

    values: List[Dict[str, Any]] = []

    bulk_reader = getattr(dab, "get_all_settings_values", None)
    if callable(bulk_reader):
        try:
            bulk_resp = await bulk_reader()
            bulk_data = dict(bulk_resp.data or {}) if isinstance(getattr(bulk_resp, "data", None), dict) else {}
            bulk_map = _extract_bulk_settings_map(bulk_data)
            if bool(getattr(bulk_resp, "success", False)) and bulk_map:
                for entry in (settings_entries or []):
                    key = str((entry or {}).get("key") or "").strip()
                    row = dict(entry or {})
                    row["key"] = key
                    has_value = key in bulk_map
                    row.update(
                        {
                            "read_success": has_value,
                            "read_status": int(getattr(bulk_resp, "status", 0) or 0),
                            "read_error": None if has_value else "missing key in bulk settings/get response",
                            "current_value": bulk_map.get(key),
                            "read_raw": bulk_data,
                        }
                    )
                    values.append(row)
                logger.info("DAB settings/get refresh used bulk request: device=%s keys=%s", device_id, len(values))
            elif bool(getattr(bulk_resp, "success", False)) and not settings_entries and bulk_map:
                for key, current_value in bulk_map.items():
                    values.append(
                        {
                            "key": str(key),
                            "friendlyName": str(key).replace("_", " ").title(),
                            "read_success": True,
                            "read_status": int(getattr(bulk_resp, "status", 0) or 0),
                            "read_error": None,
                            "current_value": current_value,
                            "read_raw": bulk_data,
                        }
                    )
                logger.info("DAB settings/get refresh used bulk request without settings/list: device=%s keys=%s", device_id, len(values))
        except Exception as exc:
            logger.warning("Bulk settings/get request failed for %s: %s", device_id, exc)

    if not values and settings_entries:
        values = list(await asyncio.gather(*[_read_setting(item) for item in settings_entries]))

    payload = {
        "success": True,
        "device_id": device_id,
        "captured_at": _utc_now_iso(),
        "count": len(values),
        "failed": sum(1 for item in values if not bool(item.get("read_success"))),
        "values": values,
    }

    _device_settings_values_cache[device_id] = {
        "cached_at": time.monotonic(),
        "payload": payload,
    }
    _device_settings_values_last_request_at[device_id] = time.monotonic()

    try:
        await _persist_settings_values_to_snapshot(device_id, payload)
    except Exception as exc:
        logger.warning("Failed to persist settings/get value snapshot for %s: %s", device_id, exc)

    logger.info("DAB settings/get refresh end: device=%s keys=%s failed=%s", device_id, len(values), payload["failed"])
    return payload


async def _get_device_settings_values_cached(device_id: str, *, force: bool = False, throttle: bool = True) -> Dict[str, Any]:
    now = time.monotonic()
    cached_entry = _device_settings_values_cache.get(device_id)
    cached_payload = _get_cached_payload(_device_settings_values_cache, device_id)

    if (not force) and throttle and cached_payload is not None:
        last_req = float(_device_settings_values_last_request_at.get(device_id, 0.0) or 0.0)
        delta = now - last_req
        if delta < max(0.0, float(_device_settings_values_min_interval_seconds)):
            logger.info("DAB settings/get throttled: device=%s age=%.2fs", device_id, delta)
            _device_settings_values_last_request_at[device_id] = now
            return cached_payload

    if (not force) and isinstance(cached_entry, dict) and cached_payload is not None:
        cached_at = float(cached_entry.get("cached_at") or 0.0)
        if (now - cached_at) <= max(0.0, float(_device_settings_values_ttl_seconds)):
            logger.info("DAB settings/get cache hit: device=%s age=%.2fs", device_id, now - cached_at)
            _device_settings_values_last_request_at[device_id] = now
            return cached_payload

    inflight = _device_settings_values_inflight.get(device_id)
    if inflight is not None and not inflight.done():
        if cached_payload is not None:
            logger.info("DAB settings/get dedup hit (stale-safe): device=%s", device_id)
            _device_settings_values_last_request_at[device_id] = now
            return cached_payload
        logger.info("DAB settings/get dedup wait: device=%s", device_id)
        _device_settings_values_last_request_at[device_id] = now
        return await inflight

    logger.info("DAB settings/get cache miss: device=%s force=%s", device_id, force)
    task = asyncio.create_task(_refresh_device_settings_values(device_id))
    _device_settings_values_inflight[device_id] = task
    _device_settings_values_last_request_at[device_id] = now
    try:
        return await task
    finally:
        current = _device_settings_values_inflight.get(device_id)
        if current is task:
            _device_settings_values_inflight.pop(device_id, None)


def _load_device_capabilities_cache() -> Dict[str, Any]:
    path = _device_capabilities_cache_path()
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load device capabilities cache from %s: %s", path, exc)
        return {}


def _save_device_capabilities_cache(payload: Dict[str, Any]) -> None:
    try:
        path = _device_capabilities_cache_path()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save device capabilities cache: %s", exc)


def _is_valid_discovered_device_id(value: Any) -> bool:
    v = str(value or "").strip()
    if not v:
        return False
    if v.endswith(":") or v.endswith("/"):
        return False
    if v.lower() in {"adb", "adb:", "device", "unknown", "none", "null", "n/a", "na"}:
        return False
    return True


def _load_device_context_state() -> Dict[str, Any]:
    state = {
        "selected_device_id": str(get_config().dab_device_id or "").strip(),
    }
    path = _device_context_path()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                selected = str(loaded.get("selected_device_id") or "").strip()
                if _is_valid_discovered_device_id(selected):
                    state["selected_device_id"] = selected
        except Exception as exc:
            logger.warning("Failed to load device context from %s: %s", path, exc)
    return state


def _save_device_context_state(state: Dict[str, Any]) -> None:
    try:
        path = _device_context_path()
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to persist device context: %s", exc)


def _resolve_selected_device_id(device_id: Optional[str] = None) -> str:
    explicit = str(device_id or "").strip()
    if _is_valid_discovered_device_id(explicit):
        return explicit
    if _is_valid_discovered_device_id(_selected_device_id_override):
        return str(_selected_device_id_override or "").strip()
    loaded = _load_device_context_state()
    selected = str(loaded.get("selected_device_id") or "").strip()
    if _is_valid_discovered_device_id(selected):
        return selected
    return str(get_config().dab_device_id or "").strip()


async def _apply_selected_device_context(device_id: Optional[str], persist: bool = True) -> Dict[str, Any]:
    global _selected_device_id_override, _dab_client, _screen_capture

    resolved = _resolve_selected_device_id(device_id)
    _selected_device_id_override = resolved or None

    config = get_config()
    if resolved:
        config.dab_device_id = resolved

    old_dab = _dab_client
    _dab_client = None

    old_capture = _screen_capture
    _screen_capture = None

    if old_capture is not None:
        try:
            old_capture.close()
        except Exception:
            pass

    if old_dab is not None:
        try:
            await old_dab.close()
        except Exception:
            pass

    state = {"selected_device_id": resolved}
    if persist:
        _save_device_context_state(state)
    return state


def _normalize_discovered_devices(payload: Any) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(device_id: Any, label: Optional[str] = None, raw: Optional[Dict[str, Any]] = None) -> None:
        did = str(device_id or "").strip()
        if not _is_valid_discovered_device_id(did) or did in seen:
            return
        seen.add(did)
        devices.append(
            {
                "device_id": did,
                "label": str(label or did).strip() or did,
                "raw": raw or {"deviceId": did},
            }
        )

    if isinstance(payload, dict):
        candidates = payload.get("devices")
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, dict):
                    add(item.get("deviceId") or item.get("device_id") or item.get("id"), item.get("name") or item.get("label"), item)
                else:
                    add(item)
        elif _is_valid_discovered_device_id(payload.get("deviceId") or payload.get("device_id") or payload.get("id")):
            add(payload.get("deviceId") or payload.get("device_id") or payload.get("id"), payload.get("name") or payload.get("label"), payload)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                add(item.get("deviceId") or item.get("device_id") or item.get("id"), item.get("name") or item.get("label"), item)
            else:
                add(item)

    if not devices:
        configured = str(get_config().dab_device_id or "").strip()
        if _is_valid_discovered_device_id(configured):
            add(configured, "Configured device", {"deviceId": configured})
    return devices


async def _discover_dab_devices(max_age_seconds: float = 30.0) -> tuple[List[Dict[str, Any]], Optional[str]]:
    global _discovered_devices_cache, _discovered_devices_cache_at, _discovery_warning_cache, _discover_devices_in_flight

    now = time.monotonic()
    if _discovered_devices_cache and (now - _discovered_devices_cache_at) <= max(0.0, float(max_age_seconds)):
        return list(_discovered_devices_cache), _discovery_warning_cache

    if _discover_devices_in_flight is not None and not _discover_devices_in_flight.done():
        return await _discover_devices_in_flight

    async def _run_discovery() -> tuple[List[Dict[str, Any]], Optional[str]]:
        global _discovered_devices_cache, _discovered_devices_cache_at, _discovery_warning_cache
        try:
            resp = await get_dab_client().discover_devices()
            if not resp.success:
                raise HTTPException(status_code=502, detail=resp.data.get("error") or "DAB discovery failed")
            devices = _normalize_discovered_devices(resp.data)
            _discovered_devices_cache = list(devices)
            _discovered_devices_cache_at = time.monotonic()
            _discovery_warning_cache = None
            return devices, None
        except Exception as exc:
            message = str(getattr(exc, "detail", "") or exc).strip() or "DAB discovery failed"
            logger.warning("DAB discovery degraded to fallback devices: %s", message)
            fallback = _normalize_discovered_devices(_discovered_devices_cache)
            if fallback:
                _discovered_devices_cache = list(fallback)
                _discovered_devices_cache_at = time.monotonic()
            _discovery_warning_cache = message
            return fallback, message

    _discover_devices_in_flight = asyncio.create_task(_run_discovery())
    try:
        return await _discover_devices_in_flight
    finally:
        if _discover_devices_in_flight is not None and _discover_devices_in_flight.done():
            _discover_devices_in_flight = None


async def _ensure_selected_device_context(device_id: Optional[str], persist: bool = False) -> str:
    explicit = str(device_id or "").strip()
    if explicit:
        current = _resolve_selected_device_id()
        if explicit != current:
            state = await _apply_selected_device_context(explicit, persist=persist)
            return str(state.get("selected_device_id") or "").strip()
        if persist:
            _save_device_context_state({"selected_device_id": explicit})
        return explicit
    return _resolve_selected_device_id()


async def _collect_selected_device_capabilities(device_id: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "device_id": str(device_id or "").strip(),
        "captured_at": _utc_now_iso(),
        "operations": [],
        "keys": [],
        "settings": [],
        "settings_schema": {},
        "counts": {"operations": 0, "keys": 0, "settings": 0},
        "errors": [],
    }

    dab = get_dab_client()

    try:
        ops_resp = await dab.list_operations()
        ops = (ops_resp.data or {}).get("operations", []) if isinstance(getattr(ops_resp, "data", None), dict) else []
        if isinstance(ops, list):
            result["operations"] = [str(item).strip() for item in ops if str(item).strip()]
        elif not bool(getattr(ops_resp, "success", False)):
            result["errors"].append(str((ops_resp.data or {}).get("error") or "operations/list failed"))
    except Exception as exc:
        result["errors"].append(f"operations/list exception: {exc}")

    if any("input/key/list" in op.lower() for op in result["operations"]):
        try:
            keys_resp = await dab.list_keys()
            keys = (keys_resp.data or {}).get("keys", []) if isinstance(getattr(keys_resp, "data", None), dict) else []
            if isinstance(keys, list):
                result["keys"] = [str(item).strip() for item in keys if str(item).strip()]
            elif not bool(getattr(keys_resp, "success", False)):
                result["errors"].append(str((keys_resp.data or {}).get("error") or "input/key/list failed"))
        except Exception as exc:
            result["errors"].append(f"input/key/list exception: {exc}")

    if any("system/settings/list" in op.lower() for op in result["operations"]):
        try:
            settings_resp = await dab.list_settings()
            settings = (settings_resp.data or {}).get("settings", []) if isinstance(getattr(settings_resp, "data", None), dict) else []
            if isinstance(settings, list):
                cleaned = [item for item in settings if isinstance(item, dict)]
                result["settings"] = cleaned
                result["settings_schema"] = {
                    str(item.get("key") or "").strip(): item
                    for item in cleaned
                    if str(item.get("key") or "").strip()
                }
            if not bool(getattr(settings_resp, "success", False)):
                result["errors"].append(str((settings_resp.data or {}).get("error") or "system/settings/list failed"))
            warning = str(((settings_resp.data or {}).get("warning") if isinstance(getattr(settings_resp, "data", None), dict) else "") or "").strip()
            if warning:
                result["errors"].append(warning)
        except Exception as exc:
            result["errors"].append(f"system/settings/list exception: {exc}")

    result["counts"] = {
        "operations": len(result.get("operations") or []),
        "keys": len(result.get("keys") or []),
        "settings": len(result.get("settings") or []),
    }
    return result


async def _refresh_discovered_device_capabilities_cache(
    *,
    force: bool = False,
    max_age_seconds: float = 120.0,
) -> Dict[str, Any]:
    global _device_capabilities_cache, _device_capabilities_cache_at

    now = time.monotonic()
    if (not force) and _device_capabilities_cache and (now - _device_capabilities_cache_at) <= max(0.0, float(max_age_seconds)):
        return dict(_device_capabilities_cache)

    devices, warning = await _discover_dab_devices(max_age_seconds=0.0 if force else 30.0)
    selected_before = _resolve_selected_device_id()

    by_device: Dict[str, Any] = {}
    for item in devices:
        device_id = str(item.get("device_id") or "").strip()
        if not _is_valid_discovered_device_id(device_id):
            continue
        try:
            await _apply_selected_device_context(device_id, persist=False)
            by_device[device_id] = await _collect_selected_device_capabilities(device_id)
        except Exception as exc:
            by_device[device_id] = {
                "device_id": device_id,
                "captured_at": _utc_now_iso(),
                "operations": [],
                "keys": [],
                "settings": [],
                "settings_schema": {},
                "counts": {"operations": 0, "keys": 0, "settings": 0},
                "errors": [str(exc)],
            }

    if _is_valid_discovered_device_id(selected_before):
        try:
            await _apply_selected_device_context(selected_before, persist=False)
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "captured_at": _utc_now_iso(),
        "warning": warning,
        "devices": by_device,
        "device_ids": sorted(by_device.keys()),
        "cache_path": str(_device_capabilities_cache_path()),
    }
    _device_capabilities_cache = payload
    _device_capabilities_cache_at = time.monotonic()
    _save_device_capabilities_cache(payload)
    return dict(payload)


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


def _parse_yts_discover_output(stdout_text: str, stderr_text: str = "") -> List[Dict[str, str]]:
    text = "\n".join([str(stdout_text or ""), str(stderr_text or "")]).strip()
    entries: List[Dict[str, str]] = []
    seen: set[str] = set()

    pattern = re.compile(
        r"^\s*\((?P<short>[A-Za-z0-9]+)\)\s+(?P<label>.*?)\s+\((?P<kind>adb|dab)\s*:\s*(?P<value>[^)]+)\)\s*$",
        flags=re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue

        short_id = str(match.group("short") or "").strip()
        label = str(match.group("label") or "").strip()
        kind = str(match.group("kind") or "").strip().lower()
        value = str(match.group("value") or "").strip()
        if not short_id:
            continue

        key = f"{short_id}:{kind}:{value}".lower()
        if key in seen:
            continue
        seen.add(key)

        item: Dict[str, str] = {
            "short_id": short_id,
            "label": label,
        }
        if kind == "adb":
            item["adb"] = value
        elif kind == "dab":
            item["dab"] = value
        entries.append(item)

    return entries


async def _get_yts_discovered_devices(max_age_seconds: float = 30.0) -> List[Dict[str, str]]:
    global _yts_discover_cache, _yts_discover_cache_at

    now = time.monotonic()
    if _yts_discover_cache and (now - _yts_discover_cache_at) <= max(0.0, float(max_age_seconds)):
        return list(_yts_discover_cache)

    discover_res = await asyncio.to_thread(_run_yts_command, ["yts", "discover", "--list"])
    if discover_res.get("returncode") != 0:
        raise HTTPException(status_code=500, detail=f"YTS discover failed: {discover_res.get('stderr')}")

    parsed = _parse_yts_discover_output(
        str(discover_res.get("stdout") or ""),
        str(discover_res.get("stderr") or ""),
    )
    _yts_discover_cache = list(parsed)
    _yts_discover_cache_at = time.monotonic()
    return parsed


def _is_ip_port(value: str) -> bool:
    v = str(value or "").strip()
    if not v:
        return False
    if ":" not in v:
        return False
    host, _, port = v.rpartition(":")
    return bool(host and port.isdigit())


async def _resolve_yts_runner_device_id(device_id: Optional[str]) -> str:
    selected_device_id = await _ensure_selected_device_context(device_id, persist=bool(device_id))
    candidate = str(selected_device_id or "").strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="No selected device available")

    lower = candidate.lower()
    if lower.startswith("dab:"):
        return candidate

    if lower.startswith("adb:"):
        adb_tail = candidate[4:].strip()
        if _is_ip_port(adb_tail):
            return f"adb:{adb_tail}"

        discovered = await _get_yts_discovered_devices()
        for item in discovered:
            adb_val = str(item.get("adb") or "").strip()
            if not adb_val:
                continue
            if adb_val == adb_tail and _is_ip_port(adb_val):
                return f"adb:{adb_val}"
        return candidate

    if _is_ip_port(candidate):
        return f"adb:{candidate}"

    discovered = await _get_yts_discovered_devices()
    token = lower
    for item in discovered:
        short_id = str(item.get("short_id") or "").strip()
        adb_val = str(item.get("adb") or "").strip()
        dab_val = str(item.get("dab") or "").strip()
        label = str(item.get("label") or "").strip().lower()

        if token and token in {short_id.lower(), adb_val.lower(), f"adb:{adb_val}".lower(), dab_val.lower(), f"dab:{dab_val}".lower()}:
            if adb_val and _is_ip_port(adb_val):
                return f"adb:{adb_val}"
            if dab_val:
                return f"dab:{dab_val}"
            if short_id:
                return short_id
        if token and label and token in label:
            if adb_val and _is_ip_port(adb_val):
                return f"adb:{adb_val}"
            if short_id:
                return short_id

    return candidate


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


def _is_timezone_setting_key(setting_key: str) -> bool:
    return str(setting_key or "").strip().lower() in {"timezone", "time zone", "time_zone", "time-zone"}


def _is_dab_setting_operation_unavailable(resp: Any) -> bool:
    status = int(getattr(resp, "status", 0) or 0)
    if status in {404, 405, 501}:
        return True
    data = getattr(resp, "data", {}) or {}
    error_text = ""
    if isinstance(data, dict):
        error_text = str(data.get("error") or data.get("message") or "")
    lowered = error_text.lower()
    return any(
        marker in lowered
        for marker in (
            "not supported",
            "unsupported",
            "not implemented",
            "unavailable",
            "no shell command implementation",
            "operation not found",
        )
    )


async def _infer_android_and_adb_device_id(selected_device_id: str, device_info: Optional[Dict[str, Any]]) -> tuple[bool, Optional[str], str, bool, str, Optional[str]]:
    info = device_info or {}
    platform = str(
        info.get("platform")
        or info.get("osFamily")
        or info.get("os")
        or info.get("deviceType")
        or ""
    ).strip()
    lower_platform = platform.lower()
    selected = str(selected_device_id or "").strip()
    selected_lower = selected.lower()

    is_android = any(token in lower_platform for token in ("android", "android-tv", "google-tv")) or selected_lower.startswith("adb:")

    adb_candidates = [
        info.get("adbDeviceId"),
        info.get("adb_device_id"),
        info.get("adbSerial"),
        info.get("adb_serial"),
        info.get("adb"),
        info.get("serial"),
        selected,
    ]
    adb_device_id: Optional[str] = None
    for candidate in adb_candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        if value.lower().startswith("adb:"):
            tail = value[4:].strip()
            if tail:
                adb_device_id = tail
                break
            continue
        if value.upper().startswith("DAB/"):
            continue
        adb_device_id = value
        break

    is_android_tv = any(token in lower_platform for token in ("android-tv", "google-tv", "television", "leanback"))
    connection_type = "unknown"
    detection_error: Optional[str] = None
    if adb_device_id:
        detected = await get_device_platform_info(adb_device_id)
        connection_type = str(detected.get("connection_type") or "unknown")
        detection_error = str(detected.get("error") or "").strip() or None
        if bool(detected.get("reachable")):
            is_android = bool(detected.get("is_android"))
            is_android_tv = bool(detected.get("is_android_tv"))
            if not platform:
                platform = "android-tv" if is_android_tv else ("android" if is_android else "unknown")

    return is_android, adb_device_id, platform or "unknown", is_android_tv, connection_type, detection_error


def _resolve_api_setting_execution_method(
    *,
    supported_operations: List[str],
    operation: str,
    setting_key: str,
    is_android: bool,
    detection_error: Optional[str] = None,
) -> tuple[str, str]:
    decision = resolve_execution_method(
        is_android=bool(is_android),
        dab_supported=operation_supported_by_dab(supported_operations or [], operation),
        adb_fallback_available=has_android_adb_fallback(operation, setting_key),
    )
    if decision.method == "unsupported" and detection_error and has_android_adb_fallback(operation, setting_key):
        return decision.method, f"{decision.reason}; adb detection error: {detection_error}"
    return decision.method, decision.reason


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
    active_model = _get_active_vertex_planner_model()
    try:
        project = _resolve_vertex_project(c.google_cloud_project)
        try:
            _vertex_text_client = VertexPlannerClient(
                project=project,
                location=c.google_cloud_location,
                model=active_model,
                api_key=str(getattr(c, "google_api_key", "") or "").strip() or None,
            )
        except TypeError:
            _vertex_text_client = VertexPlannerClient(
                project=project,
                location=c.google_cloud_location,
                model=active_model,
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
    planner_model = _get_active_vertex_planner_model()
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
        try:
            _vertex_live_visual_client = VertexPlannerClient(
                project=project,
                location=c.google_cloud_location,
                model=_select_vertex_visual_model(),
                api_key=str(getattr(c, "google_api_key", "") or "").strip() or None,
            )
        except TypeError:
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
        prompt = "\n\n".join(
            [
                _shared_ai_prompt_preamble(),
                "Task: monitor Android TV guided validation state from the attached screenshot.",
                "Return strict JSON with keys: summary, playback_visible, player_controls_visible, settings_gear_visible, stats_for_nerds_visible, focus_target, confidence.",
                "Rules: use screenshot as source of truth; Do not rely on OCR or local text extraction; keep summary short and factual; mention selected menu item only when clearly visible; request a UI navigation checkpoint when commit path is uncertain.",
                f"YTS command: {state.get('command') or 'unknown'}",
                f"Recent terminal logs:\n{log_text or '(no recent logs)'}",
            ]
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
        prompt = "\n\n".join(
            [
                _shared_ai_prompt_preamble(),
                "Task: answer the interactive YTS terminal prompt with a single safe operator response.",
                response_instruction.strip(),
                f"Interactive prompt: {prompt_text}",
                f"Allowed options: {', '.join(options) if options else 'infer from prompt'}",
                "Rules: attached screenshot is the primary visual context; prefer non-destructive option when uncertain; if UI path is unclear, favor response that allows a UI navigation checkpoint first; return only the response token.",
                f"Active test terminal context:\n{active_test_log_text or '(active test context unavailable)'}",
                f"Recent terminal logs:\n{recent_log_text or '(no recent logs)'}",
                f"Executed DAB setup actions before answering: {json.dumps(setup_actions, ensure_ascii=False)}",
                f"TV visual context source: {visual_context.get('source', 'unknown')}",
                f"TV observation sequence: {json.dumps(visual_context.get('observations') or [], ensure_ascii=False)}",
                f"Latest Gemini live visual analysis: {json.dumps(visual_context.get('analysis') or state.get('latest_visual_analysis') or {}, ensure_ascii=False)}",
                f"TV capture summary:\n{visual_context.get('summary', 'No TV screenshot available.')}",
                f"Capture status: {json.dumps(visual_context.get('capture_status') or {}, ensure_ascii=False)}",
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
        key = str((p or {}).get("key", "")).strip()
        return f"Trying direct DAB write for {key or 'the setting'}"
    if action_u == "GET_SETTING":
        key = str((p or {}).get("key", "")).strip()
        return f"Trying direct DAB read for {key or 'the setting'}"
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
    status_obj = getattr(state, "status", None)
    status = getattr(status_obj, "value", str(status_obj or "UNKNOWN"))
    goal_text = str(getattr(state, "goal", "") or "")
    state_error = getattr(state, "error", None)
    recent_actions = list(getattr(state, "last_actions", []) or [])
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
    strategy_transitions = [e for e in state.ai_transcript if e.get("type") == "strategy-transition"]
    direct_unsupported_events = [e for e in state.ai_transcript if e.get("type") == "direct-op-unsupported"]
    used_ui_fallback = any(str(e.get("to", "")).upper() == "UI_NAVIGATION_FALLBACK" for e in strategy_transitions)
    settings_goal = any(k in goal_text.lower() for k in ("setting", "timezone", "time zone", "language"))
    has_direct_setting_success = any(
        str(a.action).upper() in {"GET_SETTING", "SET_SETTING"} and str(a.result).upper() == "PASS"
        for a in state.action_history
    )
    has_ui_nav_attempt = any(
        str(a.action).upper() in {"PRESS_UP", "PRESS_DOWN", "PRESS_LEFT", "PRESS_RIGHT", "PRESS_OK", "PRESS_BACK", "PRESS_HOME"}
        for a in state.action_history
    )
    verification_confidence = (
        "high" if has_direct_setting_success else ("medium" if has_ui_nav_attempt else "low")
    )

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
        or getattr(state, "latest_ocr_text", None)
        or getattr(state, "latest_visual_summary", None)
        or getattr(state, "current_app_state", None)
        or getattr(state, "current_screen", None)
        or "unknown screen"
    )

    if settings_goal and used_ui_fallback:
        latest_unsupported = direct_unsupported_events[-1] if direct_unsupported_events else {}
        unsupported_op = str(latest_unsupported.get("operation") or "system/settings/get")
        unsupported_key = str(latest_unsupported.get("setting_key") or "target setting")
        unsupported_reason = str(latest_unsupported.get("reason") or "backend does not expose this operation")

        if status == "DONE" and has_direct_setting_success:
            short = "Completed via direct and fallback checks"
            detailed = (
                f"Direct DAB access for {unsupported_key} had limitations, so the agent switched to UI navigation and still verified the result."
            )
            friendly = (
                f"Direct DAB read for {unsupported_key} was not fully supported, so the agent switched to UI navigation through Settings and verified the outcome."
            )
            summary = "The run finished successfully using UI navigation fallback after direct DAB limits."
        elif status == "DONE":
            short = "Completed with limited verification"
            detailed = (
                f"Direct DAB operation {unsupported_op} for {unsupported_key} was unavailable. The agent continued through UI navigation, but final value confirmation remained limited."
            )
            friendly = (
                f"Direct DAB read for {unsupported_key} is not supported on this device, so the agent switched to UI navigation through Settings. "
                "It reached the expected path, but the final value could not be confirmed with high confidence."
            )
            summary = "The run completed with fallback, but final verification confidence is limited."
        else:
            short = "Direct settings access unsupported"
            detailed = (
                f"Direct DAB operation {unsupported_op} for {unsupported_key} failed repeatedly with backend limitation: {unsupported_reason}."
            )
            friendly = (
                f"The device does not expose {unsupported_key} through the current DAB backend. "
                "The agent switched to UI navigation fallback, but the goal was not fully completed."
            )
            summary = "Direct settings API was unsupported, fallback was used, and the run did not fully complete."

        root = "Direct DAB settings operation unsupported on this device/build"
        technical = unsupported_reason
        screen_based_reason = f"Latest observed screen/state: {seen_screen}."
        goal_based_reason = (
            f"Goal was '{goal_text}'. The agent attempted direct DAB first, then switched to remote-style UI navigation fallback."
        )
        recovery_summary = (
            f"Fallback strategy: UI_NAVIGATION_FALLBACK using input keys with a {int(get_config().session_timeout_seconds)}s timeout ceiling. "
            f"Verification confidence: {verification_confidence}."
        )
        evidence = [
            f"strategy_transitions={len(strategy_transitions)}",
            f"direct_unsupported_events={len(direct_unsupported_events)}",
            f"verification_confidence={verification_confidence}",
        ]
    elif (
        "youtube" in goal_text.lower()
        and "stats for nerds" in goal_text.lower()
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
        goal_based_reason = f"Goal was '{goal_text}', but target app was not verified in foreground."
        recovery_summary = (
            "The tool retried state checks and screenshot-based recovery, but each attempt stayed away from the target screen."
        )
        evidence = [
            f"current_app={state.current_app or state.current_app_id}",
            f"current_state={state.current_app_state or state.current_screen}",
            f"parse_failures={parse_fail_count}",
            f"stuck_diagnosis_events={len(stuck_events)}",
        ]
    elif ("youtube" in goal_text.lower() and "stats for nerds" in goal_text.lower()
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
            f"goal={goal_text}",
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
        goal_based_reason = f"Goal was '{goal_text}', but the planner could not produce a grounded path to it."
        recovery_summary = "The tool performed recovery checks after failures, but decisions stayed invalid."
        evidence = [
            f"parse_failures={parse_fail_count}",
            f"recent_actions={recent_actions[-5:]}",
        ]
    elif status in {"FAILED", "ERROR", "TIMEOUT"}:
        short = "Run stopped before completion"
        detailed = state_error or "The run could not finish after recovery attempts."
        friendly = "The test could not finish successfully, so it was stopped safely."
        root = "Recovery failed"
        technical = state_error or "Unknown failure"
        summary = "The run stopped after repeated problems."
        screen_based_reason = f"Latest observed screen/state: {seen_screen}."
        goal_based_reason = f"Goal was '{goal_text}', but the run could not reach the required result."
        recovery_summary = "The tool tried bounded recovery steps and then stopped safely."
        evidence = [
            f"status={status}",
            f"retries={getattr(state, 'retries', 0)}",
        ]
    else:
        short = "Run completed"
        detailed = "The run finished without critical failure."
        friendly = "The test completed successfully."
        root = "Completed"
        technical = "No terminal failures"
        summary = "The run finished successfully."
        screen_based_reason = f"Latest observed screen/state: {seen_screen}."
        goal_based_reason = f"Goal '{goal_text}' was completed."
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
    api_key = str(getattr(config, "google_api_key", "") or "").strip()
    if api_key and config.vertex_planner_model:
        return True
    project = _resolve_vertex_project(config.google_cloud_project)
    return bool(project and config.google_cloud_location and config.vertex_planner_model)


def _get_active_vertex_planner_model() -> str:
    override = str(_runtime_vertex_planner_model_override or "").strip()
    if override:
        return override
    return str(get_config().vertex_planner_model or "").strip()


def _get_available_vertex_models() -> List[str]:
    c = get_config()
    models = {
        str(m).strip()
        for m in [
            *_COMMON_VERTEX_MODELS,
            str(c.vertex_planner_model or "").strip(),
            str(c.vertex_live_model or "").strip(),
        ]
        if str(m).strip()
    }
    return sorted(models)


def _reset_runtime_model_clients() -> None:
    global _planner, _vertex_text_client, _vertex_live_visual_client
    _planner = None
    _vertex_text_client = None
    _vertex_live_visual_client = None


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
        active_model = _get_active_vertex_planner_model()
        vertex_client = None
        if _vertex_planner_requested():
            try:
                project = _resolve_vertex_project(c.google_cloud_project)
                try:
                    vertex_client = VertexPlannerClient(
                        project=project,
                        location=c.google_cloud_location,
                        model=active_model,
                        api_key=str(getattr(c, "google_api_key", "") or "").strip() or None,
                    )
                except TypeError:
                    vertex_client = VertexPlannerClient(
                        project=project,
                        location=c.google_cloud_location,
                        model=active_model,
                    )
                logger.info(
                    "Planner initialized with Vertex model=%s project=%s location=%s",
                    active_model,
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
    global _selected_device_id_override, _device_capabilities_cache, _device_capabilities_cache_at

    loaded_device = str(_load_device_context_state().get("selected_device_id") or "").strip()
    if _is_valid_discovered_device_id(loaded_device):
        _selected_device_id_override = loaded_device
        get_config().dab_device_id = loaded_device

    await asyncio.to_thread(_mark_stale_yts_live_commands)
    await asyncio.to_thread(_refresh_yts_test_catalog, _YTS_TESTLIST_PATH, False, False)
    await asyncio.to_thread(_refresh_yts_test_catalog, _YTS_GUIDED_TESTLIST_PATH, True, False)
    try:
        _loaded_cap = _load_device_capabilities_cache()
        if _loaded_cap:
            _device_capabilities_cache = _loaded_cap
            _device_capabilities_cache_at = time.monotonic()
    except Exception:
        pass
    try:
        await _refresh_discovered_device_capabilities_cache(force=True)
    except Exception as exc:
        logger.warning("Startup capability cache warmup failed: %s", exc)
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


def _legacy_status_payload() -> dict:
    config = get_config()
    capture_status: dict[str, Any] = {}
    screen_capture = _screen_capture
    if screen_capture is not None:
        try:
            capture_status = screen_capture.capture_source_status()
        except Exception as exc:
            logger.warning("Failed to build websocket capture status snapshot: %s", exc)
            capture_status = {}
    raw_devices = capture_status.get("video_device_details") or []
    devices = []
    for index, item in enumerate(raw_devices, start=1):
        device_path = item.get("path") or item.get("device") or item.get("device_path")
        readable = bool(item.get("readable", item.get("exists", True)))
        state = "AVAILABLE" if readable else "FAILED"
        devices.append(
            {
                "device_id": item.get("device_id") or item.get("id") or f"device{index}",
                "kind": item.get("kind") or item.get("label") or "video",
                "locator": device_path,
                "required": False,
                "device_path": device_path,
                "state": state,
                "is_open": readable,
                "first_frame_received": readable,
                "frame_available": readable,
                "frames_captured": 0,
                "dropped_frames": 0,
                "reconnect_attempts": 0,
                "fps": 0.0,
                "startup_duration_ms": None,
                "last_frame_at": None,
                "last_frame_age_seconds": None,
                "last_error": item.get("error"),
                "last_warning": item.get("warning"),
                "initialization_complete": True,
            }
        )

    available_devices = [item for item in devices if item["state"] == "AVAILABLE"]
    return {
        "status": "ok",
        "service": "vertex_live_dab_agent",
        "mock_mode": config.dab_mock_mode,
        "run_count": len(_runs),
        "active_run_count": len([run for run in _runs.values() if run.status == RunStatus.RUNNING]),
        "configured_source": capture_status.get("configured_source"),
        "selected_video_device": capture_status.get("selected_video_device"),
        "preferred_video_kind": capture_status.get("preferred_video_kind"),
        "device_count": len(devices),
        "available_device_count": len(available_devices),
        "failed_required_count": 0,
        "devices": devices,
    }

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


@app.get("/stream-status")
async def stream_status() -> dict:
    """Compatibility status endpoint for preview clients."""
    return _legacy_status_payload()


@app.get("/stream/compat-status")
async def stream_status_alias() -> dict:
    """Compatibility alias for preview websocket/status clients."""
    return _legacy_status_payload()


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(_legacy_status_payload())
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


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


@app.get("/config/runtime-model", response_model=RuntimeModelResponse)
async def runtime_model_summary() -> RuntimeModelResponse:
    c = get_config()
    active = _get_active_vertex_planner_model()
    configured = str(c.vertex_planner_model or "").strip()
    return RuntimeModelResponse(
        success=True,
        active_vertex_planner_model=active,
        configured_vertex_planner_model=configured,
        available_models=_get_available_vertex_models(),
        message=("runtime override active" if active and active != configured else "using configured model"),
    )


@app.post("/config/runtime-model", response_model=RuntimeModelResponse)
async def runtime_model_update(request: RuntimeModelUpdateRequest) -> RuntimeModelResponse:
    global _runtime_vertex_planner_model_override
    requested = str(request.model or "").strip()
    c = get_config()
    configured = str(c.vertex_planner_model or "").strip()

    if not requested:
        raise HTTPException(status_code=400, detail="model is required")

    if requested.lower() in {"default", "configured", "reset"}:
        _runtime_vertex_planner_model_override = None
        _reset_runtime_model_clients()
        active = _get_active_vertex_planner_model()
        return RuntimeModelResponse(
            success=True,
            active_vertex_planner_model=active,
            configured_vertex_planner_model=configured,
            available_models=_get_available_vertex_models(),
            message="runtime override cleared; using configured model",
        )

    _runtime_vertex_planner_model_override = requested
    _reset_runtime_model_clients()
    active = _get_active_vertex_planner_model()
    return RuntimeModelResponse(
        success=True,
        active_vertex_planner_model=active,
        configured_vertex_planner_model=configured,
        available_models=_get_available_vertex_models(),
        message="runtime model updated",
    )


class DeviceContextSelectRequest(BaseModel):
    device_id: str
    persist: bool = True


@app.get("/dab/devices", response_model=dict)
async def dab_devices() -> dict:
    """Discover DAB devices and return normalized device list."""
    devices, warning = await _discover_dab_devices()
    selected_device_id = _resolve_selected_device_id()
    discovered_ids = {str(item.get("device_id") or "").strip() for item in devices}
    if devices and selected_device_id not in discovered_ids:
        selected_device_id = str(devices[0].get("device_id") or "").strip()
        if selected_device_id:
            await _apply_selected_device_context(selected_device_id, persist=True)
    caps = await _refresh_discovered_device_capabilities_cache(force=False)
    return {
        "success": True,
        "devices": devices,
        "selected_device_id": selected_device_id,
        "warning": warning,
        "capabilities_cache_path": str(_device_capabilities_cache_path()),
        "capabilities_cached_devices": list(caps.get("device_ids") or []),
    }


@app.get("/dab/capabilities/cache", response_model=dict)
async def dab_capabilities_cache(refresh: bool = False) -> dict:
    """Return per-device capability cache built from operations/list, key/list and settings/list."""
    global _device_capabilities_cache, _device_capabilities_cache_at
    if refresh:
        payload = await _refresh_discovered_device_capabilities_cache(force=True)
    else:
        if not _device_capabilities_cache:
            loaded = _load_device_capabilities_cache()
            if loaded:
                _device_capabilities_cache = loaded
                _device_capabilities_cache_at = time.monotonic()
        payload = _device_capabilities_cache or await _refresh_discovered_device_capabilities_cache(force=False)
    return {
        "success": True,
        "cache_path": str(_device_capabilities_cache_path()),
        "captured_at": payload.get("captured_at"),
        "warning": payload.get("warning"),
        "device_ids": payload.get("device_ids") or [],
        "devices": payload.get("devices") or {},
    }


@app.get("/device/context", response_model=dict)
async def device_context() -> dict:
    """Return current selected DAB device context."""
    devices = list(_discovered_devices_cache)
    warning: Optional[str] = None
    if not devices:
        devices, warning = await _discover_dab_devices()
    selected_device_id = _resolve_selected_device_id()
    discovered_ids = {str(item.get("device_id") or "").strip() for item in devices}
    if devices and selected_device_id not in discovered_ids:
        selected_device_id = str(devices[0].get("device_id") or "").strip()
        if selected_device_id:
            await _apply_selected_device_context(selected_device_id, persist=True)
    return {
        "selected_device_id": selected_device_id,
        "configured_device_id": str(get_config().dab_device_id or "").strip(),
        "devices": devices,
        "warning": warning,
    }


@app.post("/device/context/select", response_model=dict)
async def select_device_context(request: DeviceContextSelectRequest) -> dict:
    """Set selected DAB device context and persist it."""
    requested_device_id = str(request.device_id or "").strip()
    if not _is_valid_discovered_device_id(requested_device_id):
        raise HTTPException(status_code=400, detail="device_id is required")

    devices, warning = await _discover_dab_devices()
    discovered_ids = {str(item.get("device_id") or "").strip() for item in devices}
    if discovered_ids and requested_device_id not in discovered_ids:
        raise HTTPException(status_code=400, detail=f"Unknown device_id: {requested_device_id}")

    state = await _apply_selected_device_context(requested_device_id, persist=bool(request.persist))
    return {
        "selected_device_id": str(state.get("selected_device_id") or "").strip(),
        "configured_device_id": str(get_config().dab_device_id or "").strip(),
        "devices": devices,
        "warning": warning,
    }


@app.get("/dab/device-info", response_model=dict)
async def dab_device_info(device_id: Optional[str] = None) -> dict:
    """Return DAB device metadata for selected or requested device."""
    current_selected_device_id = str(_resolve_selected_device_id() or "").strip()
    resolved_device_id = str(_resolve_selected_device_id(device_id) or "").strip()
    if not _is_valid_discovered_device_id(resolved_device_id):
        raise HTTPException(status_code=400, detail="No selected device available")

    if str(device_id or "").strip() and resolved_device_id != current_selected_device_id:
        await _apply_selected_device_context(resolved_device_id, persist=True)

    try:
        resp = await get_dab_client().get_device_info()
    except DABError as exc:
        return {
            "success": False,
            "device_id": resolved_device_id,
            "result": {},
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "success": False,
            "device_id": resolved_device_id,
            "result": {},
            "error": str(exc),
        }
    if not resp.success:
        return {
            "success": False,
            "device_id": resolved_device_id,
            "result": resp.data,
            "error": str(resp.data.get("error") or "Failed to load device info"),
        }
    return {
        "success": True,
        "device_id": resolved_device_id,
        "result": resp.data,
    }


@app.get("/dab/device-settings", response_model=dict)
async def dab_device_settings(device_id: Optional[str] = None) -> dict:
    """Return settings/list plus current values using cached catalog/value snapshots."""
    current_selected_device_id = str(_resolve_selected_device_id() or "").strip()
    resolved_device_id = str(_resolve_selected_device_id(device_id) or "").strip()
    if not _is_valid_discovered_device_id(resolved_device_id):
        raise HTTPException(status_code=400, detail="No selected device available")

    if str(device_id or "").strip() and resolved_device_id != current_selected_device_id:
        await _apply_selected_device_context(resolved_device_id, persist=True)

    catalog = await _get_device_dab_catalog_cached(resolved_device_id, force=False)
    values_payload = await _get_device_settings_values_cached(resolved_device_id, force=False, throttle=True)
    settings_entries = list((catalog.get("settings_list") or {}).get("list") or [])
    by_key = {
        str(item.get("key") or "").strip(): item
        for item in (values_payload.get("values") or [])
        if isinstance(item, dict)
    }
    settings_with_values: List[Dict[str, Any]] = []
    for entry in settings_entries:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "").strip()
        merged = dict(entry)
        if key and key in by_key:
            merged.update(by_key[key])
        settings_with_values.append(merged)

    failed_reads = int(values_payload.get("failed") or 0)
    settings_meta = catalog.get("settings_list") or {}
    return {
        "success": bool(settings_meta.get("success", True)),
        "device_id": resolved_device_id,
        "captured_at": str(values_payload.get("captured_at") or _utc_now_iso()),
        "warning": settings_meta.get("warning"),
        "degraded": bool(settings_meta.get("degraded")),
        "list_status": int(settings_meta.get("status", 0) or 0),
        "list_error": settings_meta.get("error"),
        "settings_count": len(settings_with_values),
        "failed_reads": failed_reads,
        "settings": settings_with_values,
    }


@app.get("/dab/device-settings/values", response_model=dict)
async def dab_device_setting_values(device_id: Optional[str] = None, force: bool = False) -> dict:
    """Refresh setting values only (system/settings/get) using cached catalog when possible."""
    resolved_device_id = str(_resolve_selected_device_id(device_id) or "").strip()
    if not _is_valid_discovered_device_id(resolved_device_id):
        raise HTTPException(status_code=400, detail="No selected device available")

    current_selected_device_id = str(_resolve_selected_device_id() or "").strip()
    if str(device_id or "").strip() and resolved_device_id != current_selected_device_id:
        await _apply_selected_device_context(resolved_device_id, persist=True)

    return await _get_device_settings_values_cached(
        resolved_device_id,
        force=bool(force),
        throttle=not bool(force),
    )


@app.get("/dab/device-system-state", response_model=dict)
async def dab_device_system_state(device_id: Optional[str] = None, refresh: bool = False) -> dict:
    """Read persisted per-device capability/status JSON; refresh only when explicitly requested."""
    current_selected_device_id = str(_resolve_selected_device_id() or "").strip()
    resolved_device_id = str(_resolve_selected_device_id(device_id) or "").strip()
    if not _is_valid_discovered_device_id(resolved_device_id):
        raise HTTPException(status_code=400, detail="No selected device available")

    if str(device_id or "").strip() and resolved_device_id != current_selected_device_id:
        await _apply_selected_device_context(resolved_device_id, persist=True)

    snapshot_path = _device_system_state_path(resolved_device_id)
    if not refresh and snapshot_path.exists():
        try:
            cached = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                return cached
        except Exception:
            pass

    catalog = await _get_device_dab_catalog_cached(resolved_device_id, force=bool(refresh))
    values_payload = await _get_device_settings_values_cached(
        resolved_device_id,
        force=bool(refresh),
        throttle=not bool(refresh),
    )
    snapshot = _build_device_capability_status_snapshot(
        device_id=resolved_device_id,
        catalog=catalog,
        values_payload=values_payload,
        snapshot_path=snapshot_path,
    )

    try:
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to persist device system state snapshot for %s: %s", resolved_device_id, exc)

    return snapshot


async def _refresh_device_system_state_snapshot_safe(device_id: Optional[str] = None) -> None:
    """Best-effort refresh of persisted device system snapshot JSON."""
    try:
        await dab_device_system_state(device_id=device_id, refresh=True)
    except Exception as exc:
        logger.warning("Skipping system-state snapshot refresh for device=%s: %s", device_id, exc)


async def _refresh_device_setting_values_snapshot_safe(device_id: Optional[str] = None) -> None:
    """Best-effort refresh for settings/get values only (lightweight)."""
    try:
        await dab_device_setting_values(device_id=device_id, force=True)
    except Exception as exc:
        logger.warning("Skipping settings-values snapshot refresh for device=%s: %s", device_id, exc)


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
    await _ensure_selected_device_context(request.device_id, persist=bool(request.device_id))

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
    state.ui_navigation_allowed = bool(request.ui_navigation_allowed)
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
    device: Optional[str] = None
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
    device_id: Optional[str] = None


class YtsInteractiveResponseRequest(BaseModel):
    response: str


class YtsInteractiveSuggestRequest(BaseModel):
    send_response: bool = False


@app.post("/yts/command/live")
async def yts_live_command(request: YtsCommandRequest) -> dict:
    """Start a YTS command and capture live stdout/stderr for polling."""
    selected_device_id = await _ensure_selected_device_context(request.device_id, persist=bool(request.device_id))

    if str(request.command or "").strip().lower() == "test":
        params = list(request.params or [])
        raw_test_device = params[0] if params else selected_device_id
        resolved_test_device = await _resolve_yts_runner_device_id(raw_test_device)
        if params:
            params[0] = resolved_test_device
        else:
            params = [resolved_test_device]
        request.params = params

    command_id = str(uuid.uuid4())
    state = _new_yts_live_state(command_id, bool(request.interactive_ai))
    state["device_id"] = selected_device_id
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
        "device_id": selected_device_id,
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
    if str(request.command or "").strip().lower() == "test":
        params = list(request.params or [])
        raw_test_device = params[0] if params else request.device_id
        resolved_test_device = await _resolve_yts_runner_device_id(raw_test_device)
        if params:
            params[0] = resolved_test_device
        else:
            params = [resolved_test_device]
        request.params = params

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
    resolved_test_device = await _resolve_yts_runner_device_id(request.device)

    result_file = Path(request.json_output or '/tmp/yts_test_result.json')
    result_file.unlink(missing_ok=True)

    cmd = ['yts', 'test', resolved_test_device, request.test]
    if request.filters:
        cmd.extend(request.filters)
    if request.args:
        cmd.extend(request.args)
    if request.json_output:
        cmd.extend(['--json-output', str(result_file)])
    
    res = _run_yts_command(cmd)

    output = {
        'command': ' '.join(cmd),
        'device_id': resolved_test_device,
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
        ai_request_count=state.ai_request_count,
        ui_navigation_allowed=bool(state.ui_navigation_allowed),
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
        ai_request_count=state.ai_request_count,
        ui_navigation_allowed=bool(state.ui_navigation_allowed),
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

    try:
        diagnosis = _build_final_diagnosis(state)
    except Exception as exc:
        logger.warning("Explain diagnosis fallback for run_id=%s: %s", run_id, exc)
        diagnosis = FinalDiagnosis(
            status=getattr(getattr(state, "status", None), "value", str(getattr(state, "status", "UNKNOWN"))),
            final_summary="Diagnosis fallback: run data is incomplete, but timeline is still available.",
            root_cause="Insufficient diagnosis data",
            user_friendly_reason="The run summary is available, but some optional diagnostics were missing.",
            technical_reason=f"Diagnosis build failed: {exc}",
            recovery_attempts=0,
            what_failed_first=None,
            what_failed_last=None,
            failure_reason_short="Incomplete run metadata",
            failure_reason_detailed="Optional fields were unavailable while building diagnosis.",
            failure_reason_user_friendly="Some advanced diagnostics were unavailable for this run.",
            screen_based_reason="No stable screen evidence available.",
            goal_based_reason=f"Goal: {getattr(state, 'goal', '')}",
            recovery_summary="Use action/AI/DAB transcripts for manual inspection.",
            evidence_used=["fallback_diagnosis"],
        )

    return FriendlyRunExplanationResponse(
        run_id=state.run_id,
        goal=state.goal,
        status=state.status.value,
        timeline=_friendly_timeline(state),
        diagnosis=diagnosis,
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
        await _ensure_selected_device_context(None, persist=False)
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
        await _ensure_selected_device_context(request.device_id, persist=bool(request.device_id))
        dab = get_dab_client()
        action = request.action.upper()
        params = request.params or {}
        if action in KEY_MAP:
            resp = await dab.key_press(KEY_MAP[action])
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "KEY_PRESS_CODE":
            key_code = str(params.get("key_code") or params.get("keyCode") or "").strip().upper()
            if not key_code:
                raise HTTPException(status_code=400, detail="key_code is required for KEY_PRESS_CODE")
            resp = await dab.key_press(key_code)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "LONG_KEY_PRESS":
            key_action = str(params.get("key_action", "")).strip().upper()
            key_code = str(params.get("key_code") or params.get("keyCode") or "").strip().upper()
            duration_ms = int(params.get("duration_ms") or params.get("durationMs") or 1500)
            resolved_key = KEY_MAP.get(key_action) if key_action else key_code
            if not resolved_key:
                raise HTTPException(
                    status_code=400,
                    detail="LONG_KEY_PRESS requires params.key_action or params.key_code",
                )
            resp = await dab.long_key_press(resolved_key, duration_ms=duration_ms)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "LAUNCH_APP":
            app_id = params.get("app_id") or params.get("appId") or ""
            if not app_id:
                raise HTTPException(status_code=400, detail="app_id is required for LAUNCH_APP")
            app_id = _validate_app_id(app_id)
            launch_parameters: Dict[str, Any] = {}
            content = str(params.get("content") or params.get("content_id") or params.get("contentId") or "").strip()
            if content:
                launch_parameters["content"] = content
            raw_parameters = params.get("parameters")
            if isinstance(raw_parameters, list):
                launch_parameters["parameters"] = raw_parameters
            elif isinstance(raw_parameters, dict):
                launch_parameters.update(raw_parameters)
            resp = await dab.launch_app(app_id, parameters=launch_parameters or None)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "WAIT":
            seconds = float(params.get("seconds", 1.0))
            seconds = max(0.0, min(seconds, 30.0))
            await asyncio.sleep(seconds)
            return ManualActionResponse(success=True, action=action, result={"seconds": seconds})
        elif action == "GET_STATE":
            app_id = params.get("app_id") or params.get("appId") or get_config().youtube_app_id
            app_id = _validate_app_id(app_id)
            resp = await dab.get_app_state(app_id)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "EXIT_APP":
            app_id = params.get("app_id") or params.get("appId") or get_config().youtube_app_id
            app_id = _validate_app_id(app_id)
            resp = await dab.exit_app(app_id)
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "OPERATIONS_LIST":
            resp = await dab.list_operations()
            await _refresh_device_system_state_snapshot_safe(_resolve_selected_device_id(request.device_id))
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "APPLICATIONS_LIST":
            resp = await dab.list_apps()
            await _refresh_device_system_state_snapshot_safe(_resolve_selected_device_id(request.device_id))
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "KEY_LIST":
            resp = await dab.list_keys()
            await _refresh_device_system_state_snapshot_safe(_resolve_selected_device_id(request.device_id))
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "SETTINGS_LIST":
            resp = await dab.list_settings()
            await _refresh_device_system_state_snapshot_safe(_resolve_selected_device_id(request.device_id))
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "VOICE_LIST":
            resp = await dab.list_voices()
            await _refresh_device_system_state_snapshot_safe(_resolve_selected_device_id(request.device_id))
            return ManualActionResponse(success=resp.success, action=action, result=resp.data)
        elif action == "GET_SETTING":
            setting_key = _normalize_setting_key(str(params.get("key") or params.get("setting_key") or ""))
            if not setting_key:
                raise HTTPException(status_code=400, detail="key is required for GET_SETTING")

            selected_device_id = _resolve_selected_device_id(request.device_id)
            ops_resp = await dab.list_operations()
            supported_ops = [str(o) for o in ((ops_resp.data or {}).get("operations", []) if isinstance(getattr(ops_resp, "data", None), dict) else [])]

            device_info_payload: Dict[str, Any] = {}
            try:
                info_resp = await dab.get_device_info()
                if bool(getattr(info_resp, "success", False)) and isinstance(getattr(info_resp, "data", None), dict):
                    device_info_payload = dict(info_resp.data)
            except Exception as exc:
                logger.warning("GET_SETTING device/info probe failed: %s", exc)

            is_android, adb_device_id, platform, is_android_tv, connection_type, detection_error = await _infer_android_and_adb_device_id(
                selected_device_id,
                device_info_payload,
            )
            method, reason = _resolve_api_setting_execution_method(
                supported_operations=supported_ops,
                operation="system/settings/get",
                setting_key=setting_key,
                is_android=is_android,
                detection_error=detection_error,
            )
            logger.info(
                "Manual GET_SETTING routing: key=%s platform=%s is_android=%s is_android_tv=%s connection=%s method=%s reason=%s detection_error=%s",
                setting_key,
                platform,
                is_android,
                is_android_tv,
                connection_type,
                method,
                reason,
                detection_error,
            )

            if method == "dab":
                resp = await dab.get_setting(setting_key)
                if resp.success:
                    await _refresh_device_setting_values_snapshot_safe(selected_device_id)
                    return ManualActionResponse(success=True, action=action, result=resp.data)
                if not _is_dab_setting_operation_unavailable(resp):
                    return ManualActionResponse(success=False, action=action, result=resp.data)
                if not is_android:
                    return ManualActionResponse(success=False, action=action, result=resp.data, error="DAB setting read failed and Android fallback is not allowed")
                method = "adb"

            if method == "adb":
                if not is_android:
                    return ManualActionResponse(success=False, action=action, error="ADB fallback is Android-only")
                if not adb_device_id:
                    return ManualActionResponse(success=False, action=action, error="Unable to resolve adb device id for fallback")
                if not _is_timezone_setting_key(setting_key):
                    return ManualActionResponse(success=False, action=action, error=f"No Android ADB fallback mapping for setting '{setting_key}'")
                online, online_detail = await is_adb_device_online(adb_device_id)
                if not online:
                    return ManualActionResponse(success=False, action=action, error=f"ADB device unavailable: {online_detail}")
                fallback_read = await get_timezone_via_adb(adb_device_id)
                if not bool(fallback_read.get("success")):
                    return ManualActionResponse(
                        success=False,
                        action=action,
                        result={"path": "ADB_FALLBACK", "fallback": fallback_read},
                        error=str(fallback_read.get("error") or "ADB timezone read failed"),
                    )
                await _refresh_device_setting_values_snapshot_safe(selected_device_id)
                return ManualActionResponse(
                    success=True,
                    action=action,
                    result={
                        "key": "timezone",
                        "value": fallback_read.get("timezone"),
                        "path": "ADB_FALLBACK",
                    },
                )

            return ManualActionResponse(success=False, action=action, error=reason)
        elif action == "SET_SETTING":
            setting_key = _normalize_setting_key(str(params.get("key") or params.get("setting_key") or ""))
            requested_value = params.get("value") if "value" in params else None
            if not setting_key:
                reserved = {
                    "app_id", "appId", "content", "content_id", "contentId", "parameters",
                    "key", "setting_key", "settingKey", "value", "duration_ms", "durationMs",
                    "key_code", "keyCode", "key_action", "requestId", "request_id", "status",
                    "success", "error", "message", "topic", "timestamp", "captured_at",
                    "device_id", "deviceId",
                }
                dynamic_keys = [k for k in params.keys() if str(k or "").strip() and str(k) not in reserved]
                if len(dynamic_keys) == 1:
                    setting_key = _normalize_setting_key(str(dynamic_keys[0]))
                    requested_value = params.get(dynamic_keys[0])
            if not setting_key:
                raise HTTPException(status_code=400, detail="key is required for SET_SETTING")
            if requested_value is None:
                raise HTTPException(status_code=400, detail="value is required for SET_SETTING")
            selected_device_id = _resolve_selected_device_id(request.device_id)
            ops_resp = await dab.list_operations()
            supported_ops = [str(o) for o in ((ops_resp.data or {}).get("operations", []) if isinstance(getattr(ops_resp, "data", None), dict) else [])]

            device_info_payload: Dict[str, Any] = {}
            try:
                info_resp = await dab.get_device_info()
                if bool(getattr(info_resp, "success", False)) and isinstance(getattr(info_resp, "data", None), dict):
                    device_info_payload = dict(info_resp.data)
            except Exception as exc:
                logger.warning("Timezone fallback device/info probe failed: %s", exc)

            is_android, adb_device_id, platform, is_android_tv, connection_type, detection_error = await _infer_android_and_adb_device_id(
                selected_device_id,
                device_info_payload,
            )
            method, reason = _resolve_api_setting_execution_method(
                supported_operations=supported_ops,
                operation="system/settings/set",
                setting_key=setting_key,
                is_android=is_android,
                detection_error=detection_error,
            )
            logger.info(
                "Manual SET_SETTING routing: key=%s selected_device_id=%s platform=%s is_android=%s is_android_tv=%s connection=%s method=%s reason=%s detection_error=%s",
                setting_key,
                selected_device_id,
                platform,
                is_android,
                is_android_tv,
                connection_type,
                method,
                reason,
                detection_error,
            )

            if method == "dab":
                resp = await dab.set_setting(setting_key, requested_value)
                if resp.success:
                    if _is_timezone_setting_key(setting_key):
                        logger.info("Timezone setting path used: DAB direct operation succeeded key=%s", setting_key)
                    await _refresh_device_setting_values_snapshot_safe(selected_device_id)
                    return ManualActionResponse(success=True, action=action, result=resp.data)
                if not _is_dab_setting_operation_unavailable(resp):
                    return ManualActionResponse(success=False, action=action, result=resp.data)
                if not is_android:
                    return ManualActionResponse(success=False, action=action, result=resp.data, error="DAB setting write failed and Android fallback is not allowed")
                method = "adb"

            if method != "adb":
                return ManualActionResponse(success=False, action=action, error=reason)

            tz_value = str(requested_value or "").strip()
            if not tz_value:
                return ManualActionResponse(success=False, action=action, error="timezone value is required")
            if not is_android:
                return ManualActionResponse(success=False, action=action, error="Timezone ADB fallback is Android-only")
            if not adb_device_id:
                return ManualActionResponse(success=False, action=action, error="Unable to resolve adb device id for timezone fallback")
            if not _is_timezone_setting_key(setting_key):
                return ManualActionResponse(success=False, action=action, error=f"No Android ADB fallback mapping for setting '{setting_key}'")

            online, online_detail = await is_adb_device_online(adb_device_id)
            if not online:
                return ManualActionResponse(
                    success=False,
                    action=action,
                    error=f"ADB device unavailable for timezone fallback: {online_detail}",
                )

            timezone_listing = await list_timezones_via_adb(adb_device_id)
            timezone_to_apply = tz_value
            if not bool(timezone_listing.get("success")):
                logger.warning(
                    "ADB timezone listing failed but direct set will still be attempted: adb_device_id=%s error=%s",
                    adb_device_id,
                    timezone_listing.get("error"),
                )
            else:
                ai_client = get_vertex_text_client()
                resolved = await resolve_timezone_from_supported(
                    tz_value,
                    list(timezone_listing.get("timezones") or []),
                    ai_client=ai_client,
                )
                if not bool(resolved.get("success")):
                    return ManualActionResponse(
                        success=False,
                        action=action,
                        result={
                            "path": "ADB_FALLBACK",
                            "supported_count": len(list(timezone_listing.get("timezones") or [])),
                        },
                        error=str(resolved.get("reason") or "Requested timezone is not supported by this device"),
                    )
                timezone_to_apply = str(resolved.get("resolved_timezone") or tz_value)

            fallback_result = await set_timezone_via_adb(adb_device_id, timezone_to_apply)
            verified = bool(fallback_result.get("success"))
            logger.info(
                "Timezone fallback verification: adb_device_id=%s requested=%s observed=%s verified=%s",
                adb_device_id,
                timezone_to_apply,
                fallback_result.get("observed_timezone"),
                verified,
            )
            if not verified:
                return ManualActionResponse(
                    success=False,
                    action=action,
                    result={
                        "fallback": fallback_result,
                        "path": "ADB_FALLBACK",
                    },
                    error=str(fallback_result.get("error") or "Timezone verification failed").strip() or "Timezone verification failed",
                )

            await _refresh_device_setting_values_snapshot_safe(selected_device_id)
            return ManualActionResponse(
                success=True,
                action=action,
                result={
                    "key": "timezone",
                    "value": timezone_to_apply,
                    "updated": True,
                    "path": "ADB_FALLBACK",
                    "verification": {
                        "requested": fallback_result.get("requested_timezone") or timezone_to_apply,
                        "observed": fallback_result.get("observed_timezone"),
                        "matched": True,
                    },
                },
            )
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
    await _ensure_selected_device_context(request.device_id, persist=bool(request.device_id))

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
    device: Optional[str] = None
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
    resolved_test_device = await _resolve_yts_runner_device_id(request.device)
    cmd = ['python3', 'yts.py', 'test', resolved_test_device, request.test]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd='/home/harry/youtube/ai_tool'
    )
    return {'output': result.stdout.strip(), 'error': result.stderr.strip(), 'device_id': resolved_test_device}


@app.get("/dab/apps", response_model=dict)
async def dab_apps() -> dict:
    """Return installed/launchable application list from device."""
    resp = await get_dab_client().list_apps()
    return {"success": resp.success, "result": resp.data}


@app.get("/dab/voices", response_model=dict)
async def dab_voices() -> dict:
    """Return supported voice list from device."""
    resp = await get_dab_client().list_voices()
    return {"success": resp.success, "result": resp.data}


@app.post("/dab/device-system-state/refresh", response_model=dict)
async def dab_device_system_state_refresh(device_id: Optional[str] = None) -> dict:
    """Force-refresh and persist the current device system state snapshot JSON."""
    return await dab_device_system_state(device_id=device_id, refresh=True)


@app.get("/dab/device-capability-status", response_model=dict)
async def dab_device_capability_status(device_id: Optional[str] = None, refresh: bool = False) -> dict:
    """Alias endpoint returning normalized per-device DAB capability/status JSON."""
    return await dab_device_system_state(device_id=device_id, refresh=bool(refresh))


@app.get("/dab/device-capability-status/raw", response_model=dict)
async def dab_device_capability_status_raw(device_id: Optional[str] = None) -> dict:
    """Return raw stored JSON snapshot for debugging."""
    resolved_device_id = str(_resolve_selected_device_id(device_id) or "").strip()
    if not _is_valid_discovered_device_id(resolved_device_id):
        raise HTTPException(status_code=400, detail="No selected device available")
    snapshot_path = _device_system_state_path(resolved_device_id)
    if not snapshot_path.exists():
        return {"success": False, "device_id": resolved_device_id, "error": "snapshot not found", "json_file": str(snapshot_path), "raw": {}}
    try:
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"success": False, "device_id": resolved_device_id, "error": str(exc), "json_file": str(snapshot_path), "raw": {}}
    return {"success": True, "device_id": resolved_device_id, "json_file": str(snapshot_path), "raw": raw if isinstance(raw, dict) else {}}


@app.get("/dab/device-operations-grid", response_model=dict)
async def dab_device_operations_grid(device_id: Optional[str] = None, refresh: bool = False) -> dict:
    """Return operation rows with support enablement derived from operations/list."""
    snapshot = await dab_device_system_state(device_id=device_id, refresh=bool(refresh))
    rows = _build_operations_grid_rows(snapshot)
    return {
        "success": True,
        "device_id": snapshot.get("device_id"),
        "last_updated": snapshot.get("last_updated"),
        "rows": rows,
    }


@app.get("/dab/device-current-settings", response_model=dict)
async def dab_device_current_settings(device_id: Optional[str] = None, force: bool = False) -> dict:
    """Return current setting values from system/settings/get using cache+single-flight protection."""
    values = await dab_device_setting_values(device_id=device_id, force=bool(force))
    return {
        "success": bool(values.get("success", True)),
        "device_id": values.get("device_id"),
        "last_updated": values.get("captured_at"),
        "count": int(values.get("count") or 0),
        "failed": int(values.get("failed") or 0),
        "current_setting_values": list(values.get("values") or []),
    }


class DABExecuteRequest(BaseModel):
    operation: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    device_id: Optional[str] = None


@app.post("/dab/execute-request", response_model=dict)
async def dab_execute_request(request: DABExecuteRequest) -> dict:
    """Execute a frontend-provided DAB operation through existing manual action routing."""
    operation = str(request.operation or "").strip()
    if not operation:
        raise HTTPException(status_code=400, detail="operation is required")

    operation_lower = operation.lower()
    payload = dict(request.payload or {})

    action = ""
    params: Dict[str, Any] = {}
    if operation_lower == "operations/list":
        action = "OPERATIONS_LIST"
    elif operation_lower == "applications/list":
        action = "APPLICATIONS_LIST"
    elif operation_lower in {"application/launch", "applications/launch"}:
        action = "LAUNCH_APP"
        content_value = payload.get("content")
        if content_value is None:
            content_value = payload.get("content_id")
        if content_value is None:
            content_value = payload.get("contentId")
        params = {
            "app_id": payload.get("app_id") or payload.get("appId"),
            "content": content_value,
            "parameters": payload.get("parameters"),
        }
    elif operation_lower == "applications/launch-with-content":
        action = "LAUNCH_APP"
        params = {
            "app_id": payload.get("app_id") or payload.get("appId"),
            "content": payload.get("content") or payload.get("content_id") or payload.get("contentId"),
        }
    elif operation_lower in {"applications/get-state", "application/get-state"}:
        action = "GET_STATE"
        if payload.get("app_id") or payload.get("appId"):
            params = {"app_id": payload.get("app_id") or payload.get("appId")}
    elif operation_lower in {"application/exit", "applications/exit"}:
        action = "EXIT_APP"
        if payload.get("app_id") or payload.get("appId"):
            params = {"app_id": payload.get("app_id") or payload.get("appId")}
    elif operation_lower in {"voice/list", "voices/list"}:
        action = "VOICE_LIST"
    elif operation_lower == "system/settings/list":
        action = "SETTINGS_LIST"
    elif operation_lower == "system/settings/get":
        action = "GET_SETTING"
        params = {"key": payload.get("key") or payload.get("settingKey") or payload.get("setting_key")}
    elif operation_lower == "system/settings/set":
        action = "SET_SETTING"
        params = dict(payload)
        if payload.get("settingKey") and payload.get("value") is not None:
            params = {"key": payload.get("settingKey"), "value": payload.get("value")}
    elif operation_lower == "input/key/list":
        action = "KEY_LIST"
    elif operation_lower in {"input/key/press", "input/key-press"}:
        action = "KEY_PRESS_CODE"
        params = {"key_code": payload.get("key_code") or payload.get("keyCode")}
    elif operation_lower in {"input/long-key-press", "input/long-key/press"}:
        action = "LONG_KEY_PRESS"
        params = {
            "key_code": payload.get("key_code") or payload.get("keyCode"),
            "duration_ms": payload.get("duration_ms") or payload.get("durationMs") or 1500,
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operation mapping: {operation}")

    manual = await manual_action(
        ManualActionRequest(
            action=action,
            params=params,
            device_id=request.device_id,
        )
    )
    return {
        "success": bool(manual.success),
        "device_id": str(_resolve_selected_device_id(request.device_id) or ""),
        "operation": operation,
        "action": action,
        "result": manual.result,
        "error": manual.error,
    }
