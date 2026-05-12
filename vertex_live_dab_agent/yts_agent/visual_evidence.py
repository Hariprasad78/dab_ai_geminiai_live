"""Normalize continuous YTS visual monitor events into evidence records."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _truthy_unknown(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1", "visible", "active"}:
        return True
    if text in {"false", "no", "0", "not visible", "inactive"}:
        return False
    return None


def _classify_context(summary: str, analysis: Dict[str, Any]) -> str:
    explicit = str(analysis.get("detected_app_context") or analysis.get("current_app_context") or "").strip()
    if explicit:
        return explicit
    text = summary.lower()
    if re.search(r"\b(youtube|yt\s+player|video\s+player|playback|player controls)\b", text):
        return "YouTube"
    if re.search(r"\b(home screen|launcher|app launcher|system ui|device home)\b", text):
        return "system/launcher"
    if re.search(r"\b(settings|setup|permission dialog|system dialog)\b", text):
        return "system"
    return "unknown"


def _screen_type(summary: str, analysis: Dict[str, Any]) -> str:
    explicit = str(analysis.get("screen_type") or analysis.get("active_screen_type") or "").strip()
    if explicit:
        return explicit
    text = summary.lower()
    if bool(analysis.get("playback_visible")) or re.search(r"\b(video|playback|playing|player)\b", text):
        return "video_playback"
    if re.search(r"\b(home screen|launcher)\b", text):
        return "launcher"
    if re.search(r"\b(dialog|prompt|permission)\b", text):
        return "dialog"
    if re.search(r"\b(settings|menu)\b", text):
        return "settings_screen"
    return "unknown"


def observation_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    analysis = dict(event.get("analysis") or {})
    summary = str(event.get("summary") or analysis.get("summary") or "").strip()
    confidence = analysis.get("confidence", event.get("confidence"))
    try:
        confidence_value = float(confidence or 0.0)
    except Exception:
        confidence_value = 0.0
    playback = _truthy_unknown(analysis.get("video_playback_active"))
    if playback is None:
        playback = _truthy_unknown(event.get("video_playback_active"))
    if playback is None:
        playback = bool(analysis.get("playback_visible") or event.get("playback_visible"))
    youtube_active = _truthy_unknown(analysis.get("youtube_active"))
    context = _classify_context(summary, analysis)
    if youtube_active is None:
        youtube_active = context.lower().startswith("youtube")
    return {
        "timestamp": event.get("captured_at") or event.get("timestamp") or "",
        "source": event.get("source") or "unknown",
        "detected_app_context": context,
        "screen_type": _screen_type(summary, analysis),
        "video_playback_active": playback,
        "youtube_active": youtube_active,
        "detected_text": str(analysis.get("detected_text") or event.get("detected_text") or "").strip(),
        "visual_summary": summary,
        "confidence": confidence_value,
        "requirement_seen": bool(analysis.get("requirement_seen") or event.get("requirement_seen")),
        "requirement_evidence": str(analysis.get("requirement_evidence") or event.get("requirement_evidence") or "").strip(),
        "recommended_result": str(analysis.get("recommended_result") or event.get("recommended_result") or "").strip().lower(),
        "prompt_requirement_match": str(analysis.get("prompt_requirement_match") or "").strip().lower(),
    }


def build_visual_evidence(visual_context: Dict[str, Any]) -> Dict[str, Any]:
    timeline = list(visual_context.get("timeline") or [])
    observations: List[Dict[str, Any]] = [observation_from_event(item) for item in timeline]
    analysis = dict(visual_context.get("analysis") or {})
    if analysis:
        latest_event = {
            "captured_at": visual_context.get("captured_at") or "",
            "source": visual_context.get("source") or "unknown",
            "summary": analysis.get("summary") or visual_context.get("summary") or "",
            "analysis": analysis,
        }
        latest_observation = observation_from_event(latest_event)
        if not observations or observations[-1].get("visual_summary") != latest_observation.get("visual_summary"):
            observations.append(latest_observation)
    latest = observations[-1] if observations else observation_from_event(
        {
            "source": visual_context.get("source") or "unknown",
            "summary": visual_context.get("summary") or "",
            "analysis": analysis,
        }
    )
    positive = [item for item in observations if item.get("requirement_seen") or item.get("recommended_result") == "pass"]
    negative = [item for item in observations if item.get("recommended_result") == "fail"]
    return {
        "source": visual_context.get("source") or latest.get("source") or "unknown",
        "continuous_frame_count": int(visual_context.get("continuous_frame_count") or len(observations) or 0),
        "latest_observation": latest,
        "observations": observations,
        "positive_observations": positive,
        "negative_observations": negative,
        "capture_status": dict(visual_context.get("capture_status") or {}),
        "summary": visual_context.get("summary") or latest.get("visual_summary") or "",
    }
