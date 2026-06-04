"""Shared YTS validation utility helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def coerce_confidence(value: Any, default: float = 0.0) -> float:
    """Normalize numeric or label-style confidence values into 0..1."""

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        normalized = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
        scale = {
            "certain": 1.0,
            "very high": 0.95,
            "high": 0.85,
            "medium high": 0.75,
            "medium": 0.6,
            "moderate": 0.6,
            "medium low": 0.45,
            "low": 0.3,
            "very low": 0.1,
            "unknown": default,
            "none": default,
        }
        confidence = scale.get(normalized, default)
    return max(0.0, min(1.0, confidence))


def normalize_missing_evidence(value: Any) -> List[str]:
    """Normalize model/gate missing evidence into readable string bullets."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        normalized = text.lower().rstrip(".")
        if normalized in {"", "none", "n/a", "na", "no", "nothing", "not applicable", "no missing evidence"}:
            return []
        return [text]
    if isinstance(value, dict):
        out: List[str] = []
        for key, item in value.items():
            key_text = str(key or "").strip()
            if isinstance(item, (list, tuple, set)):
                item_text = ", ".join(str(part).strip() for part in item if str(part).strip())
            elif isinstance(item, dict):
                item_text = "; ".join(f"{k}: {v}" for k, v in item.items())
            else:
                item_text = str(item or "").strip()
            if key_text and item_text:
                out.append(f"{key_text}: {item_text}")
            elif key_text:
                out.append(key_text)
            elif item_text:
                out.append(item_text)
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(normalize_missing_evidence(item))
        return out
    text = str(value or "").strip()
    return [text] if text else []


def resolve_option_label(option: Any, options_map: Any) -> str:
    """Resolve an option token to the current prompt label without assuming meanings."""

    token = str(option or "").strip()
    if not token:
        return ""
    if isinstance(options_map, dict):
        return str(options_map.get(token) or options_map.get(token.lower()) or token).strip()
    if isinstance(options_map, Iterable) and not isinstance(options_map, (str, bytes)):
        for item in options_map:
            if isinstance(item, dict):
                item_option = str(item.get("option") or "").strip()
                if item_option == token:
                    return str(item.get("label") or token).strip()
            else:
                item_text = str(item or "").strip()
                if item_text == token:
                    return token
    return token


def dedupe_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out
