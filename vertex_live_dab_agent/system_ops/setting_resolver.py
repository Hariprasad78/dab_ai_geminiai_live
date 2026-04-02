"""Resolve normalized setting intents against capability snapshot."""

from __future__ import annotations

from typing import Any, Dict

from vertex_live_dab_agent.system_ops.capabilities import normalize_setting_key, normalize_setting_value


def resolve_setting_target(snapshot: Dict[str, Any], setting_name: str, setting_value: Any) -> Dict[str, Any]:
    key_norm = normalize_setting_key(snapshot, setting_name)
    if not bool(key_norm.get("success")):
        return {
            "success": False,
            "reason": str(key_norm.get("reason") or "unsupported setting key"),
            "resolved_target_setting": None,
            "resolved_target_value": None,
        }
    resolved_key = str(key_norm.get("key") or setting_name)

    value_norm = normalize_setting_value(snapshot, resolved_key, setting_value)
    if not bool(value_norm.get("success")):
        return {
            "success": False,
            "reason": str(value_norm.get("reason") or "unsupported setting value"),
            "resolved_target_setting": resolved_key,
            "resolved_target_value": None,
        }

    return {
        "success": True,
        "reason": "ok",
        "resolved_target_setting": resolved_key,
        "resolved_target_value": value_norm.get("value"),
        "value_corrected": bool(value_norm.get("corrected")),
        "value_confidence": float(value_norm.get("confidence") or 0.0),
    }
