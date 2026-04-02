"""Normalize raw user requests into canonical actionable intents."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from pydantic import BaseModel


_SPELLING_FIXES = {
    "launguge": "language",
    "langauge": "language",
    "lanugage": "language",
    "kannada": "Kannada",
    "timezone": "timezone",
    "time zone": "timezone",
    "time-zone": "timezone",
    "wi fi": "wifi",
}

_SETTING_SYNONYMS = {
    "locale": "language",
    "lang": "language",
    "time zone": "timezone",
    "time-zone": "timezone",
}


class NormalizedRequest(BaseModel):
    raw_user_text: str
    corrected_user_text: str
    action_type: str = "unknown"
    target_app: Optional[str] = None
    target_setting: Optional[str] = None
    target_value: Optional[str] = None
    normalized_user_goal: str


def _clean_text(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"\s+", " ", text)
    lowered = text.lower()
    for wrong, fixed in _SPELLING_FIXES.items():
        lowered = lowered.replace(wrong, fixed)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def normalize_user_request(raw_text: str) -> NormalizedRequest:
    cleaned = _clean_text(raw_text)
    corrected = cleaned

    action_type = "unknown"
    target_app: Optional[str] = None
    target_setting: Optional[str] = None
    target_value: Optional[str] = None

    # setting change intent
    m = re.search(r"\b(set|change|update|switch)\s+([a-z _-]+?)\s+to\s+(.+)$", cleaned)
    if m:
        action_type = "change_setting"
        target_setting = re.sub(r"\s+", " ", str(m.group(2) or "").strip().lower())
        target_value = str(m.group(3) or "").strip()
    else:
        # weaker parse: "set language kannada"
        m2 = re.search(r"\b(set|change|update|switch)\s+([a-z _-]+?)\s+([a-z0-9._-]+)$", cleaned)
        if m2:
            action_type = "change_setting"
            target_setting = str(m2.group(2) or "").strip().lower()
            target_value = str(m2.group(3) or "").strip()

    # app launch intent
    if action_type == "unknown":
        m3 = re.search(r"\b(open|launch|start)\s+([a-z0-9+ _.-]+)$", cleaned)
        if m3:
            action_type = "open_app"
            target_app = str(m3.group(2) or "").strip()

    if target_setting:
        target_setting = _SETTING_SYNONYMS.get(target_setting, target_setting)
        target_setting = target_setting.replace(" ", "_") if target_setting in {"screen saver", "screensaver"} else target_setting
        target_setting = "timezone" if target_setting in {"time zone", "time-zone"} else target_setting

    if target_value:
        v = str(target_value).strip()
        # Title-case only for language-like values.
        target_value = v.title() if target_setting in {"language", "locale"} else v

    corrected_parts = []
    if action_type == "change_setting" and target_setting and target_value:
        corrected_parts.append(f"set {target_setting} to {target_value}")
    elif action_type == "open_app" and target_app:
        corrected_parts.append(f"open {target_app}")
    elif cleaned:
        corrected_parts.append(cleaned)

    corrected_user_text = " ".join(corrected_parts).strip() or cleaned
    normalized_goal = corrected_user_text

    return NormalizedRequest(
        raw_user_text=str(raw_text or ""),
        corrected_user_text=corrected_user_text,
        action_type=action_type,
        target_app=target_app,
        target_setting=target_setting,
        target_value=target_value,
        normalized_user_goal=normalized_goal,
    )
