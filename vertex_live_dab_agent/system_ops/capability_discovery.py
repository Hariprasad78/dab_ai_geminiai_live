"""Capability discovery and normalization for heterogeneous devices."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from vertex_live_dab_agent.orchestrator.app_resolver import AppResolver
from vertex_live_dab_agent.system_ops.capabilities import build_capability_snapshot
from vertex_live_dab_agent.system_ops.device_detection import get_device_platform_info

logger = logging.getLogger(__name__)


class CapabilitySnapshot(BaseModel):
    supported_operations: List[str] = Field(default_factory=list)
    supported_settings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    supported_keys: List[str] = Field(default_factory=list)
    supported_voices: List[str] = Field(default_factory=list)
    installed_applications: List[Dict[str, Any]] = Field(default_factory=list)
    platform_type: str = "unknown"
    is_android: bool = False
    can_use_adb: bool = False
    unsupported_or_missing_capabilities: List[str] = Field(default_factory=list)


def _mark_missing(missing: List[str], name: str, reason: str) -> None:
    message = f"{name}: {reason}".strip()
    if message not in missing:
        missing.append(message)


def _extract_settings(raw_settings: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_settings, list):
        return [s for s in raw_settings if isinstance(s, dict)]
    return []


def _extract_apps(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]
    return []


async def discover_capabilities(
    *,
    dab_client: Any,
    app_resolver: Optional[AppResolver],
    device_id: str,
    current_state: Optional[Dict[str, Any]] = None,
) -> CapabilitySnapshot:
    """Discover capabilities safely. Unsupported calls are recorded, not fatal."""
    state = current_state or {}
    missing: List[str] = []

    supported_operations = list(state.get("supported_operations") or [])
    supported_keys = list(state.get("supported_keys") or [])
    supported_settings = list(state.get("supported_settings") or [])
    supported_voices = list(state.get("supported_voices") or [])
    installed_applications = list(state.get("installed_applications") or state.get("app_catalog") or [])

    device_info = state.get("device_info") if isinstance(state.get("device_info"), dict) else {}

    if not device_info and hasattr(dab_client, "get_device_info"):
        try:
            info_resp = await dab_client.get_device_info()
            if bool(getattr(info_resp, "success", False)) and isinstance(getattr(info_resp, "data", None), dict):
                device_info = dict(info_resp.data)
            elif int(getattr(info_resp, "status", 0) or 0) >= 400:
                _mark_missing(missing, "device/info", str((getattr(info_resp, "data", {}) or {}).get("error") or "unsupported"))
        except Exception as exc:
            _mark_missing(missing, "device/info", str(exc))

    if not supported_operations:
        try:
            resp = await dab_client.list_operations()
            ops = (resp.data or {}).get("operations", []) if isinstance(getattr(resp, "data", None), dict) else []
            if isinstance(ops, list):
                supported_operations = [str(op).strip() for op in ops if str(op).strip()]
            if not supported_operations:
                _mark_missing(missing, "operations/list", "empty or unavailable")
        except Exception as exc:
            _mark_missing(missing, "operations/list", str(exc))

    op_set = {str(op).strip().lower() for op in supported_operations}

    if not installed_applications and app_resolver is not None:
        try:
            catalog = await app_resolver.load_app_catalog(device_id=device_id)
            installed_applications = [
                {
                    "appId": a.app_id,
                    "name": a.name,
                    "friendlyName": a.friendly_name,
                    "packageName": a.package_name,
                }
                for a in catalog
            ]
        except Exception as exc:
            _mark_missing(missing, "applications/list", str(exc))

    if not installed_applications and "applications/list" in op_set and hasattr(dab_client, "list_apps"):
        try:
            resp = await dab_client.list_apps()
            raw_apps = (resp.data or {}).get("applications", []) if isinstance(getattr(resp, "data", None), dict) else []
            installed_applications = _extract_apps(raw_apps)
            if not installed_applications:
                _mark_missing(missing, "applications/list", "empty or unavailable")
        except Exception as exc:
            _mark_missing(missing, "applications/list", str(exc))

    if not supported_keys:
        if "input/key/list" in op_set and hasattr(dab_client, "list_keys"):
            try:
                resp = await dab_client.list_keys()
                keys = (resp.data or {}).get("keys", []) if isinstance(getattr(resp, "data", None), dict) else []
                supported_keys = [str(k).strip() for k in keys if str(k).strip()] if isinstance(keys, list) else []
                if not supported_keys:
                    _mark_missing(missing, "input/key/list", "empty or unavailable")
            except Exception as exc:
                _mark_missing(missing, "input/key/list", str(exc))
        else:
            _mark_missing(missing, "input/key/list", "operation unsupported")

    if not supported_settings:
        if ("system/settings/list" in op_set or "settings/list" in op_set) and hasattr(dab_client, "list_settings"):
            try:
                resp = await dab_client.list_settings()
                raw = (resp.data or {}).get("settings", []) if isinstance(getattr(resp, "data", None), dict) else []
                supported_settings = _extract_settings(raw)
                if not supported_settings:
                    _mark_missing(missing, "system/settings/list", "empty or unavailable")
            except Exception as exc:
                _mark_missing(missing, "system/settings/list", str(exc))
        else:
            _mark_missing(missing, "system/settings/list", "operation unsupported")

    if not supported_voices:
        if "voice/list" in op_set and hasattr(dab_client, "list_voices"):
            try:
                resp = await dab_client.list_voices()
                raw = (resp.data or {}).get("voices", []) if isinstance(getattr(resp, "data", None), dict) else []
                supported_voices = [str(v).strip() for v in raw if str(v).strip()] if isinstance(raw, list) else []
                if not supported_voices:
                    _mark_missing(missing, "voice/list", "empty or unavailable")
            except Exception as exc:
                _mark_missing(missing, "voice/list", str(exc))
        else:
            _mark_missing(missing, "voice/list", "operation unsupported")

    platform_type = "unknown"
    is_android = bool(state.get("is_android"))
    can_use_adb = bool(state.get("can_use_adb"))
    adb_device_id = str(state.get("android_adb_device_id") or "").strip() or str(device_id or "").strip()
    adb_like_hint = ":" in adb_device_id or adb_device_id.lower().startswith("emulator-") or adb_device_id.lower().startswith("adb:")

    if adb_device_id and (bool(is_android) or adb_like_hint):
        try:
            pinfo = await get_device_platform_info(adb_device_id)
            is_android = bool(pinfo.get("is_android"))
            can_use_adb = bool(is_android and pinfo.get("reachable"))
            if bool(pinfo.get("is_android_tv")):
                platform_type = "android_tv"
            elif is_android:
                platform_type = "android"
        except Exception as exc:
            _mark_missing(missing, "adb/device_detection", str(exc))

    if platform_type == "unknown":
        model = str(device_info.get("model") or device_info.get("platform") or "").lower()
        if "android" in model:
            platform_type = "android"
            is_android = True
        elif model:
            platform_type = model[:64]

    normalized = build_capability_snapshot(
        supported_operations=supported_operations,
        supported_settings=supported_settings,
        supported_keys=supported_keys,
        supported_voices=supported_voices,
        installed_applications=installed_applications,
        platform_type=platform_type,
        is_android=is_android,
        can_use_adb=can_use_adb,
        unsupported_or_missing_capabilities=missing,
    )

    snapshot = CapabilitySnapshot(
        supported_operations=list(normalized.get("supported_operations") or []),
        supported_settings=dict(normalized.get("supported_settings") or {}),
        supported_keys=list(normalized.get("supported_keys") or []),
        supported_voices=list(normalized.get("supported_voices") or []),
        installed_applications=list(normalized.get("installed_applications") or []),
        platform_type=str(normalized.get("platform_type") or "unknown"),
        is_android=bool(normalized.get("is_android")),
        can_use_adb=bool(normalized.get("can_use_adb")),
        unsupported_or_missing_capabilities=list(normalized.get("unsupported_or_missing_capabilities") or []),
    )

    logger.info(
        "Capability discovery: ops=%d settings=%d keys=%d voices=%d apps=%d platform=%s android=%s adb=%s missing=%d",
        len(snapshot.supported_operations),
        len(snapshot.supported_settings),
        len(snapshot.supported_keys),
        len(snapshot.supported_voices),
        len(snapshot.installed_applications),
        snapshot.platform_type,
        snapshot.is_android,
        snapshot.can_use_adb,
        len(snapshot.unsupported_or_missing_capabilities),
    )
    return snapshot
