"""DAB adapter helpers for deterministic direct execution."""

from __future__ import annotations

from typing import Any, Dict


def make_launch_action(app_id: str) -> Dict[str, Any]:
    return {"action": "LAUNCH_APP", "params": {"app_id": str(app_id or "").strip()}}


def make_get_setting_action(setting_key: str) -> Dict[str, Any]:
    return {"action": "GET_SETTING", "params": {"key": str(setting_key or "").strip()}}


def make_set_setting_action(setting_key: str, value: Any) -> Dict[str, Any]:
    return {"action": "SET_SETTING", "params": {"key": str(setting_key or "").strip(), "value": value}}
