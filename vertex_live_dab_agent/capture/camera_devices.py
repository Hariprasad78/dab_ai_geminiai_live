"""Unified device/camera mapping with env-var overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CAMERA_DEVICE_KEYS = ("adt4", "sonytv", "samsung", "kirkwood")
_CAMERA_ENV_OVERRIDES = {
    "adt4": "ADT4_CAM_PATH",
    "sonytv": "SONYTV_CAM_PATH",
    "samsung": "SAMSUNG_CAM_PATH",
    "kirkwood": "KIRKWOOD_CAM_PATH",
}
_CAMERA_LABELS = {
    "adt4": "ADT-4",
    "sonytv": "Sony TV",
    "samsung": "Samsung TV",
    "kirkwood": "Kirkwood",
}

_cached_config: Optional[Dict[str, str]] = None
_cached_contexts: Optional[List["DeviceContext"]] = None
_cached_config_path: Optional[Path] = None
_warned_missing_config_path: Optional[Path] = None


@dataclass(frozen=True)
class DeviceContext:
    contextId: str
    displayName: str
    dabDeviceId: str
    ytsDeviceId: str
    adbDeviceId: str = ""
    irDeviceId: str = ""
    cameraId: str = ""
    cameraPath: str = ""
    videoSource: str = "camera-capture"
    audioDevice: str = ""
    active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["geminiFeedSource"] = self.cameraPath
        payload["videoDevicePath"] = self.cameraPath
        return payload


def _default_camera_config_path() -> Path:
    override = (os.environ.get("CAMERA_DEVICES_CONFIG") or "").strip()
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    workspace_root = here.parents[3] / "camera_devices.json"
    app_root = here.parents[2] / "camera_devices.json"
    if workspace_root.exists():
        return workspace_root
    return app_root


def _env_prefix(context_id: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(context_id or "").upper())
    return normalized.strip("_")


def _env_override(context_id: str, suffix: str) -> str:
    prefix = _env_prefix(context_id)
    if not prefix:
        return ""
    return str(os.environ.get(f"{prefix}_{suffix}", "") or "").strip()


def _normalize_video_source(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"hdmi", "hdmi-capture", "capture-card"}:
        return "hdmi-capture"
    if normalized in {"camera", "camera-capture", "webcam", "usb-camera"}:
        return "camera-capture"
    if normalized in {"scrcpy", "adb", "android", "android-screen", "android-ui"}:
        return "scrcpy"
    return "camera-capture"


def _context_from_raw(raw: Dict[str, Any]) -> Optional[DeviceContext]:
    context_id = str(raw.get("contextId") or raw.get("context_id") or raw.get("id") or "").strip()
    display_name = str(raw.get("displayName") or raw.get("display_name") or raw.get("label") or context_id).strip()
    dab_id = str(_env_override(context_id, "DAB_DEVICE_ID") or raw.get("dabDeviceId") or raw.get("dab_device_id") or raw.get("dab") or context_id).strip()
    yts_id = str(_env_override(context_id, "YTS_DEVICE_ID") or raw.get("ytsDeviceId") or raw.get("yts_device_id") or raw.get("yts") or dab_id).strip()
    camera_path = str(
        _env_override(context_id, "VIDEO_PATH")
        or _env_override(context_id, "CAM_PATH")
        or raw.get("cameraPath")
        or raw.get("camera_path")
        or raw.get("videoDevicePath")
        or raw.get("video_device_path")
        or raw.get("path")
        or ""
    ).strip()
    video_source = _normalize_video_source(_env_override(context_id, "VIDEO_SOURCE") or raw.get("videoSource") or raw.get("video_source") or "")
    if not context_id or not dab_id:
        return None
    return DeviceContext(
        contextId=context_id,
        displayName=display_name or context_id,
        dabDeviceId=dab_id,
        ytsDeviceId=yts_id or dab_id,
        adbDeviceId=str(_env_override(context_id, "ADB_DEVICE_ID") or raw.get("adbDeviceId") or raw.get("adb_device_id") or raw.get("adb") or "").strip(),
        irDeviceId=str(_env_override(context_id, "IR_DEVICE_ID") or raw.get("irDeviceId") or raw.get("ir_device_id") or raw.get("ir") or "").strip(),
        cameraId=str(raw.get("cameraId") or raw.get("camera_id") or f"{context_id}_cam").strip(),
        cameraPath=camera_path,
        videoSource=video_source,
        audioDevice=str(_env_override(context_id, "AUDIO_DEVICE") or raw.get("audioDevice") or raw.get("audio_device") or raw.get("audio") or "").strip(),
        active=bool(raw.get("active", False)),
    )


def _load_camera_device_config() -> Dict[str, str]:
    global _cached_config, _cached_contexts, _cached_config_path, _warned_missing_config_path

    path = _default_camera_config_path()
    if _cached_config is not None and _cached_config_path == path:
        return dict(_cached_config)

    if not path.exists():
        if _warned_missing_config_path != path:
            logger.warning("camera_devices.json not found at %s", path)
        _warned_missing_config_path = path
        _cached_config = {}
        _cached_contexts = []
        _cached_config_path = path
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed reading camera_devices.json at %s: %s", path, exc)
        return {}

    if not isinstance(raw, dict):
        logger.error("Invalid camera_devices.json at %s: root must be an object", path)
        _cached_config = {}
        _cached_contexts = []
        _cached_config_path = path
        return {}

    out: Dict[str, str] = {}
    contexts: List[DeviceContext] = []
    raw_devices = raw.get("devices")
    if isinstance(raw_devices, list):
        for item in raw_devices:
            if not isinstance(item, dict):
                continue
            context = _context_from_raw(item)
            if context is None:
                continue
            contexts.append(context)
            aliases = {
                context.contextId.strip().lower(),
                context.dabDeviceId.strip().lower(),
                context.displayName.strip().lower().replace(" ", "").replace("-", ""),
            }
            for alias in aliases:
                if alias and context.cameraPath:
                    out[alias] = context.cameraPath
    else:
        for key in CAMERA_DEVICE_KEYS:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
                contexts.append(
                    DeviceContext(
                        contextId=key,
                        displayName=camera_label(key),
                        dabDeviceId="sony" if key == "sonytv" else ("adt-4" if key == "adt4" else key),
                        ytsDeviceId="sony" if key == "sonytv" else ("adt-4" if key == "adt4" else key),
                        irDeviceId=f"{key}_ir",
                        cameraId=f"{key}_cam",
                        cameraPath=value.strip(),
                        videoSource="camera-capture",
                    )
                )
    _cached_config = out
    _cached_contexts = contexts
    _cached_config_path = path
    _warned_missing_config_path = None
    logger.info("Loaded camera_devices.json from %s with %d device context(s)", path, len(contexts))
    return out


def load_device_contexts() -> List[DeviceContext]:
    """Return configured unified device contexts."""
    _load_camera_device_config()
    return list(_cached_contexts or [])


def find_device_context(value: str) -> Optional[DeviceContext]:
    """Find a context by context, DAB, YTS, ADB, IR, camera id/path, or display name."""
    token = str(value or "").strip()
    if not token:
        return None
    lower = token.lower()
    compact = lower.replace(" ", "").replace("-", "")
    for context in load_device_contexts():
        candidates = {
            context.contextId,
            context.displayName,
            context.dabDeviceId,
            context.ytsDeviceId,
            context.adbDeviceId,
            context.irDeviceId,
            context.cameraId,
            context.cameraPath,
        }
        for candidate in candidates:
            c = str(candidate or "").strip().lower()
            if c and (lower == c or compact == c.replace(" ", "").replace("-", "")):
                return context
    return None


def get_camera_path(camera_name: str) -> str:
    """Return camera path from env override or camera_devices.json mapping."""
    key = str(camera_name or "").strip().lower()
    if key not in CAMERA_DEVICE_KEYS:
        return ""

    env_name = _CAMERA_ENV_OVERRIDES[key]
    env_value = (os.environ.get(env_name) or "").strip()
    if env_value:
        return env_value

    config = _load_camera_device_config()
    return (config.get(key) or "").strip()


def get_camera_device_mapping() -> Dict[str, str]:
    """Return full effective camera mapping (including env overrides)."""
    return {key: get_camera_path(key) for key in CAMERA_DEVICE_KEYS}


def camera_label(camera_name: str) -> str:
    return _CAMERA_LABELS.get(str(camera_name or "").strip().lower(), str(camera_name or "").strip())


def _video_device_index(device: str) -> Optional[int]:
    try:
        resolved = os.path.realpath(str(device or "").strip())
        name = os.path.basename(resolved)
        if not name.startswith("video"):
            return None
        index_path = Path("/sys/class/video4linux") / name / "index"
        return int(index_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _video_device_name(device: str) -> str:
    try:
        resolved = os.path.realpath(str(device or "").strip())
        name = os.path.basename(resolved)
        if not name.startswith("video"):
            return ""
        name_path = Path("/sys/class/video4linux") / name / "name"
        return name_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _classify_video_source(device: str) -> str:
    text = " ".join([str(device or ""), _video_device_name(device)]).lower()
    if any(token in text for token in ("hdmi", "capture", "macrosilicon", "cam link", "elgato", "u3")):
        return "hdmi-capture"
    return "camera-capture"


def _candidate_video_devices() -> List[str]:
    candidates = set(glob.glob("/dev/v4l/by-id/*video-index0"))
    candidates.update(glob.glob("/dev/video*"))
    out: List[str] = []
    for device in sorted(candidates):
        if not os.path.exists(device):
            continue
        index = _video_device_index(device)
        if index is not None and index != 0:
            continue
        out.append(device)
    return out


def _match_score(context: DeviceContext, device: str) -> int:
    haystack = " ".join([device, os.path.realpath(device), _video_device_name(device)]).lower()
    tokens = []
    for raw in (context.contextId, context.displayName, context.cameraId, context.dabDeviceId):
        tokens.extend(str(raw or "").replace("_", " ").replace("-", " ").split())
    tokens.extend(re.split(r"[^a-zA-Z0-9]+", str(context.cameraPath or "")))

    score = 0
    ignored = {"dev", "v4l", "by", "id", "usb", "video", "index", "port", "path", "cam", "camera"}
    for token in tokens:
        cleaned = token.strip().lower()
        if len(cleaned) >= 3 and cleaned not in ignored and cleaned in haystack:
            score += 10
    if _classify_video_source(device) == context.videoSource:
        score += 3
    return score


def resolve_context_camera_path(context: DeviceContext) -> str:
    """Return an existing camera path for a context, auto-detecting when stale/missing."""
    configured = str(context.cameraPath or "").strip()
    if configured and os.path.exists(configured):
        return configured

    candidates = _candidate_video_devices()
    if not candidates:
        return configured

    source_matches = [device for device in candidates if _classify_video_source(device) == context.videoSource]
    search_space = source_matches or candidates
    ranked = sorted(
        ((device, _match_score(context, device)) for device in search_space),
        key=lambda item: item[1],
        reverse=True,
    )
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    if len(search_space) == 1 and not configured:
        return search_space[0]
    return configured


def validate_camera_devices() -> bool:
    """Validate configured camera paths and log logical->real mapping."""
    ok = True
    for key, path in get_camera_device_mapping().items():
        label = camera_label(key)
        if not path:
            logger.error("[ERROR] Camera path not configured for %s", label)
            ok = False
            continue

        exists = os.path.exists(path)
        resolved = ""
        try:
            resolved = os.path.realpath(path)
        except Exception:
            resolved = path

        if exists:
            logger.info("[INFO] Camera mapping: %s -> %s (real=%s)", key, path, resolved)
        else:
            logger.error("[ERROR] Missing camera path for %s: %s", label, path)
            ok = False
    return ok
