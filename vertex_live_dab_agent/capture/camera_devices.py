"""Centralized camera-device mapping with env-var overrides."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CAMERA_DEVICE_KEYS = ("adt4", "sonytv", "kirkwood")
_CAMERA_ENV_OVERRIDES = {
    "adt4": "ADT4_CAM_PATH",
    "sonytv": "SONYTV_CAM_PATH",
    "kirkwood": "KIRKWOOD_CAM_PATH",
}
_CAMERA_LABELS = {
    "adt4": "ADT-4",
    "sonytv": "Sony TV",
    "kirkwood": "Kirkwood",
}

_cached_config: Optional[Dict[str, str]] = None
_cached_config_path: Optional[Path] = None
_warned_missing_config_path: Optional[Path] = None


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


def _load_camera_device_config() -> Dict[str, str]:
    global _cached_config, _cached_config_path, _warned_missing_config_path

    path = _default_camera_config_path()
    if _cached_config is not None and _cached_config_path == path:
        return dict(_cached_config)

    if not path.exists():
        if _warned_missing_config_path != path:
            logger.warning("camera_devices.json not found at %s", path)
            _warned_missing_config_path = path
        _cached_config = {}
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
        _cached_config_path = path
        return {}

    out: Dict[str, str] = {}
    for key in CAMERA_DEVICE_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    _cached_config = out
    _cached_config_path = path
    _warned_missing_config_path = None
    return out


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
