"""Persistent device profile registry for hybrid planning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeviceProfile(BaseModel):
    """Normalized persisted view of a device's usable capabilities."""

    profile_id: str
    device_id: str
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)
    source: str = "dab-capability-bootstrap"
    supported_operations: List[str] = Field(default_factory=list)
    supported_keys: List[str] = Field(default_factory=list)
    supported_settings: List[Dict[str, Any]] = Field(default_factory=list)
    known_apps: List[Dict[str, Any]] = Field(default_factory=list)
    features: Dict[str, bool] = Field(default_factory=dict)


class DeviceProfileRegistry:
    """Reads/writes per-device JSON profiles to disk."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def profile_path(self, device_id: str) -> Path:
        safe_device_id = str(device_id or "unknown-device").strip().replace("/", "_")
        return self._base_dir / f"{safe_device_id}.json"

    def load(self, device_id: str) -> Optional[DeviceProfile]:
        path = self.profile_path(device_id)
        if not path.exists():
            return None
        try:
            return DeviceProfile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def upsert_from_capabilities(
        self,
        *,
        device_id: str,
        supported_operations: List[str],
        supported_keys: List[str],
        supported_settings: List[Dict[str, Any]],
        app_catalog: List[Dict[str, Any]],
    ) -> DeviceProfile:
        existing = self.load(device_id)
        created_at = existing.created_at if existing is not None else _utc_now_iso()
        profile = DeviceProfile(
            profile_id=f"profile:{device_id}",
            device_id=device_id,
            created_at=created_at,
            updated_at=_utc_now_iso(),
            supported_operations=sorted({str(item).strip() for item in supported_operations if str(item).strip()}),
            supported_keys=sorted({str(item).strip() for item in supported_keys if str(item).strip()}),
            supported_settings=[
                item for item in supported_settings
                if isinstance(item, dict)
            ],
            known_apps=[
                item for item in app_catalog
                if isinstance(item, dict)
            ],
            features=self._derive_features(
                supported_operations=supported_operations,
                supported_settings=supported_settings,
                supported_keys=supported_keys,
            ),
        )
        self.profile_path(device_id).write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return profile

    @staticmethod
    def _derive_features(
        *,
        supported_operations: List[str],
        supported_settings: List[Dict[str, Any]],
        supported_keys: List[str],
    ) -> Dict[str, bool]:
        ops = {str(item).lower() for item in supported_operations}
        settings_keys = {
            str(item.get("key", "")).strip().lower()
            for item in supported_settings
            if isinstance(item, dict)
        }
        keys = {str(item).strip().upper() for item in supported_keys}
        return {
            "direct_app_launch": any("applications/launch" in item for item in ops),
            "direct_content_open": any("content/open" in item for item in ops) or any("launch-with-content" in item for item in ops),
            "direct_setting_get": any("system/settings/get" in item for item in ops),
            "direct_setting_set": any("system/settings/set" in item for item in ops),
            "visual_capture": any("output/image" in item for item in ops),
            "voice_control": any("voice/" in item for item in ops),
            "telemetry": any("telemetry/" in item for item in ops),
            "timezone_setting_known": "timezone" in settings_keys,
            "language_setting_known": "language" in settings_keys,
            "directional_navigation": {"KEY_UP", "KEY_DOWN", "KEY_LEFT", "KEY_RIGHT"}.issubset(keys) if keys else False,
            "ok_navigation": "KEY_ENTER" in keys if keys else False,
        }
