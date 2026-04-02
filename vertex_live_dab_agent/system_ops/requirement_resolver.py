"""Requirement-first resolver that converts normalized intent into executable requirement."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class RequirementType:
    CHANGE_SETTING = "CHANGE_SETTING"
    OPEN_APP = "OPEN_APP"
    UNKNOWN = "UNKNOWN"


class RequirementResolution(BaseModel):
    requirement_type: str = RequirementType.UNKNOWN
    target_setting: Optional[str] = None
    target_value: Optional[Any] = None
    target_app: Optional[str] = None
    verification_required: bool = False
    confidence: float = 0.0
    reason: str = ""


def resolve_requirement(goal: str, normalized_intent: Dict[str, Any]) -> RequirementResolution:
    intent = normalized_intent if isinstance(normalized_intent, dict) else {}
    action_type = str(intent.get("action_type") or "").strip().lower()

    if action_type == "change_setting":
        target_setting = str(intent.get("target_setting") or "").strip() or None
        target_value = intent.get("target_value")
        if target_setting:
            return RequirementResolution(
                requirement_type=RequirementType.CHANGE_SETTING,
                target_setting=target_setting,
                target_value=target_value,
                verification_required=True,
                confidence=0.95,
                reason="normalized setting-change requirement",
            )

    if action_type == "open_app":
        target_app = str(intent.get("target_app") or "").strip() or None
        if target_app:
            return RequirementResolution(
                requirement_type=RequirementType.OPEN_APP,
                target_app=target_app,
                verification_required=True,
                confidence=0.92,
                reason="normalized app-open requirement",
            )

    g = str(goal or "").lower()
    if any(k in g for k in ("set ", "change ", "update ")) and any(k in g for k in ("language", "locale", "timezone", "time zone", "brightness", "contrast", "screensaver")):
        return RequirementResolution(
            requirement_type=RequirementType.CHANGE_SETTING,
            verification_required=True,
            confidence=0.6,
            reason="heuristic setting-change fallback",
        )
    if any(k in g for k in ("open ", "launch ", "start ")):
        return RequirementResolution(
            requirement_type=RequirementType.OPEN_APP,
            verification_required=True,
            confidence=0.6,
            reason="heuristic app-open fallback",
        )

    return RequirementResolution(requirement_type=RequirementType.UNKNOWN, confidence=0.2, reason="unresolved requirement")
