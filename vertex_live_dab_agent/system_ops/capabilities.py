"""Capability snapshot and validation helpers for planner/executor safety."""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Iterable, List, Optional


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(str(value).strip())
    except Exception:
        return None


def _derive_setting_schema(item: Dict[str, Any]) -> Dict[str, Any]:
    key = str(item.get("key") or "").strip()
    friendly = str(item.get("friendlyName") or item.get("name") or "").strip()
    allowed_values = item.get("values")
    if not isinstance(allowed_values, list):
        allowed_values = item.get("allowedValues")
    if not isinstance(allowed_values, list):
        allowed_values = item.get("options")

    value_type = str(item.get("type") or item.get("valueType") or "").strip().lower()
    min_v = _to_float(item.get("min") if "min" in item else item.get("minValue"))
    max_v = _to_float(item.get("max") if "max" in item else item.get("maxValue"))

    if isinstance(allowed_values, list):
        values = [str(v).strip() for v in allowed_values if str(v).strip()]
        return {
            "key": key,
            "friendly_name": friendly,
            "type": "list",
            "values": values,
            "supported": bool(values),
        }

    if value_type in {"boolean", "bool"} or isinstance(item.get("supported"), bool):
        supported = bool(item.get("supported", True))
        return {
            "key": key,
            "friendly_name": friendly,
            "type": "boolean",
            "supported": supported,
        }

    if value_type in {"number", "int", "integer", "float"} or (min_v is not None and max_v is not None):
        supported = (min_v is not None and max_v is not None and min_v <= max_v)
        return {
            "key": key,
            "friendly_name": friendly,
            "type": "number",
            "min": min_v,
            "max": max_v,
            "supported": supported,
        }

    return {
        "key": key,
        "friendly_name": friendly,
        "type": "unknown",
        "supported": bool(item.get("supported", True)),
    }


def build_capability_snapshot(
    *,
    supported_operations: Iterable[str],
    supported_settings: Iterable[Dict[str, Any]],
    supported_keys: Iterable[str],
) -> Dict[str, Any]:
    operations = [str(op).strip() for op in (supported_operations or []) if str(op).strip()]
    keys = [str(k).strip().upper() for k in (supported_keys or []) if str(k).strip()]
    settings_map: Dict[str, Dict[str, Any]] = {}
    for raw in (supported_settings or []):
        if not isinstance(raw, dict):
            continue
        schema = _derive_setting_schema(raw)
        key = str(schema.get("key") or "").strip()
        if not key:
            continue
        settings_map[key] = schema

    return {
        "supported_operations": operations,
        "supported_settings": settings_map,
        "supported_keys": sorted(set(keys)),
    }


def has_operation(snapshot: Dict[str, Any], operation: str) -> bool:
    target = str(operation or "").strip().lower()
    if not target:
        return False
    return any(target == str(op).strip().lower() for op in (snapshot.get("supported_operations") or []))


def has_setting(snapshot: Dict[str, Any], setting_key: str) -> bool:
    key = str(setting_key or "").strip()
    schema = (snapshot.get("supported_settings") or {}).get(key)
    if not isinstance(schema, dict):
        return False
    return bool(schema.get("supported", True))


def has_key(snapshot: Dict[str, Any], key_code: str) -> bool:
    key = str(key_code or "").strip().upper()
    keys = {str(k).strip().upper() for k in (snapshot.get("supported_keys") or [])}
    return key in keys


def normalize_setting_key(snapshot: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    value = str(user_input or "").strip()
    settings = snapshot.get("supported_settings") or {}
    if not value or not isinstance(settings, dict):
        return {"success": False, "reason": "setting key is required"}

    token = _normalize_token(value)
    direct = {str(k).strip().lower(): str(k).strip() for k in settings.keys()}
    if value.lower() in direct:
        resolved = direct[value.lower()]
        return {"success": True, "key": resolved, "corrected": resolved != value, "confidence": 1.0}

    alias_map: Dict[str, str] = {}
    for key, schema in settings.items():
        alias_map[_normalize_token(key)] = key
        friendly = str((schema or {}).get("friendly_name") or "").strip()
        if friendly:
            alias_map[_normalize_token(friendly)] = key

    if token in alias_map:
        resolved = alias_map[token]
        return {"success": True, "key": resolved, "corrected": resolved != value, "confidence": 0.98}

    candidates = list(alias_map.keys())
    best = difflib.get_close_matches(token, candidates, n=1, cutoff=0.84)
    if best:
        resolved = alias_map[best[0]]
        score = difflib.SequenceMatcher(a=token, b=best[0]).ratio()
        return {"success": True, "key": resolved, "corrected": True, "confidence": score}

    return {"success": False, "reason": f"unsupported or ambiguous setting key '{value}'"}


def normalize_setting_value(snapshot: Dict[str, Any], setting_key: str, user_input: Any) -> Dict[str, Any]:
    settings = snapshot.get("supported_settings") or {}
    schema = settings.get(str(setting_key or "").strip()) if isinstance(settings, dict) else None
    if not isinstance(schema, dict) or not bool(schema.get("supported", True)):
        return {"success": False, "reason": f"setting '{setting_key}' is not supported"}

    setting_type = str(schema.get("type") or "unknown")
    raw = str(user_input or "").strip()
    if setting_type == "list":
        candidates = [str(v).strip() for v in (schema.get("values") or []) if str(v).strip()]
        if not candidates:
            return {"success": False, "reason": f"setting '{setting_key}' has no supported values"}
        lower_map = {c.lower(): c for c in candidates}
        if raw.lower() in lower_map:
            resolved = lower_map[raw.lower()]
            return {"success": True, "value": resolved, "corrected": resolved != raw, "confidence": 1.0}

        raw_token = _normalize_token(raw)
        token_map = {_normalize_token(c): c for c in candidates}
        if raw_token in token_map:
            resolved = token_map[raw_token]
            return {"success": True, "value": resolved, "corrected": True, "confidence": 0.98}

        best = difflib.get_close_matches(raw_token, list(token_map.keys()), n=1, cutoff=0.84)
        if best:
            resolved = token_map[best[0]]
            score = difflib.SequenceMatcher(a=raw_token, b=best[0]).ratio()
            return {"success": True, "value": resolved, "corrected": True, "confidence": score}
        return {"success": False, "reason": f"unsupported or ambiguous value '{raw}' for setting '{setting_key}'"}

    if setting_type == "boolean":
        token = raw.lower()
        if token in {"1", "true", "on", "enable", "enabled", "yes"}:
            return {"success": True, "value": True, "corrected": token != "true", "confidence": 1.0}
        if token in {"0", "false", "off", "disable", "disabled", "no"}:
            return {"success": True, "value": False, "corrected": token != "false", "confidence": 1.0}
        return {"success": False, "reason": f"invalid boolean value '{raw}' for setting '{setting_key}'"}

    if setting_type == "number":
        parsed = _to_float(raw)
        if parsed is None:
            return {"success": False, "reason": f"invalid numeric value '{raw}' for setting '{setting_key}'"}
        min_v = _to_float(schema.get("min"))
        max_v = _to_float(schema.get("max"))
        if min_v is None or max_v is None or min_v > max_v:
            return {"success": False, "reason": f"setting '{setting_key}' is not supported"}
        if parsed < min_v or parsed > max_v:
            return {"success": False, "reason": f"value {parsed} is out of range [{min_v}, {max_v}] for setting '{setting_key}'"}
        return {"success": True, "value": int(parsed) if float(parsed).is_integer() else parsed, "corrected": False, "confidence": 1.0}

    return {"success": True, "value": user_input, "corrected": False, "confidence": 0.8}


def validate_setting_value(snapshot: Dict[str, Any], setting_key: str, value: Any) -> Dict[str, Any]:
    normalized = normalize_setting_value(snapshot, setting_key, value)
    return {"success": bool(normalized.get("success")), "reason": normalized.get("reason"), "value": normalized.get("value")}


def normalize_key_code(snapshot: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    raw = str(user_input or "").strip().upper()
    keys = [str(k).strip().upper() for k in (snapshot.get("supported_keys") or []) if str(k).strip()]
    if not raw:
        return {"success": False, "reason": "key code is required"}
    if raw in keys:
        return {"success": True, "key_code": raw, "corrected": False, "confidence": 1.0}

    token = _normalize_token(raw)
    alias: Dict[str, str] = {}
    for key in keys:
        alias[_normalize_token(key)] = key
        if key.startswith("KEY_"):
            alias[_normalize_token(key[4:])] = key
    if token in alias:
        resolved = alias[token]
        return {"success": True, "key_code": resolved, "corrected": True, "confidence": 0.98}

    best = difflib.get_close_matches(token, list(alias.keys()), n=1, cutoff=0.86)
    if best:
        resolved = alias[best[0]]
        score = difflib.SequenceMatcher(a=token, b=best[0]).ratio()
        return {"success": True, "key_code": resolved, "corrected": True, "confidence": score}
    return {"success": False, "reason": f"unsupported or ambiguous key code '{user_input}'"}
