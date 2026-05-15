"""Normalize continuous YTS visual monitor events into evidence records."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from vertex_live_dab_agent.yts_agent.utils import coerce_confidence

_AD_SIGNAL_RE = re.compile(
    r"\bads?\b|\bad\s+\d+\s+of\s+\d+\b|\badvert(?:isement|ising)?\b|"
    r"\bcommercial\s+break\b|\bsponsored\b|\bsponsor\b|\bpromo(?:tion)?\b|"
    r"\bskip\s+ads?\b|\bvisit\s+advertiser\b",
    re.IGNORECASE,
)


def _contains_ad_signal(*values: Any) -> bool:
    return any(_AD_SIGNAL_RE.search(str(value or "")) for value in values)


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
    if _contains_ad_signal(text, analysis.get("detected_text")):
        return "ad_interstitial"
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
    detected_text = str(analysis.get("detected_text") or event.get("detected_text") or "").strip()
    confidence = analysis.get("confidence", event.get("confidence"))
    confidence_value = coerce_confidence(confidence)
    playback = _truthy_unknown(analysis.get("video_playback_active"))
    if playback is None:
        playback = _truthy_unknown(event.get("video_playback_active"))
    if playback is None:
        playback = bool(analysis.get("playback_visible") or event.get("playback_visible"))
    youtube_active = _truthy_unknown(analysis.get("youtube_active"))
    context = _classify_context(summary, analysis)
    if youtube_active is None:
        youtube_active = context.lower().startswith("youtube")
    screen_type = _screen_type(summary, analysis)
    ad_or_interstitial_visible = bool(
        analysis.get("ad_or_interstitial_visible")
        or analysis.get("ad_visible")
        or analysis.get("advertisement_visible")
        or analysis.get("sponsored_visible")
        or analysis.get("blocking_overlay_visible")
        or event.get("ad_or_interstitial_visible")
        or _contains_ad_signal(
            summary,
            detected_text,
            context,
            screen_type,
            analysis.get("blocking_overlay_reason"),
            analysis.get("requirement_evidence"),
        )
    )
    if ad_or_interstitial_visible:
        screen_type = "ad_interstitial"
    recommended_result = str(analysis.get("recommended_result") or event.get("recommended_result") or "").strip().lower()
    requirement_seen = bool(analysis.get("requirement_seen") or event.get("requirement_seen"))
    if ad_or_interstitial_visible:
        requirement_seen = False
        if recommended_result == "pass":
            recommended_result = "fail"
    return {
        "timestamp": event.get("captured_at") or event.get("timestamp") or "",
        "prompt_timestamp": event.get("prompt_timestamp") or analysis.get("prompt_timestamp") or "",
        "prompt_id": event.get("prompt_id") or analysis.get("prompt_id"),
        "prompt_sequence_id": event.get("prompt_sequence_id") or analysis.get("prompt_sequence_id"),
        "source": event.get("source") or "unknown",
        "detected_app_context": context,
        "screen_type": screen_type,
        "video_playback_active": playback,
        "youtube_active": youtube_active,
        "detected_text": detected_text,
        "visual_summary": summary,
        "confidence": confidence_value,
        "requirement_seen": requirement_seen,
        "requirement_evidence": str(analysis.get("requirement_evidence") or event.get("requirement_evidence") or "").strip(),
        "recommended_result": recommended_result,
        "ad_or_interstitial_visible": ad_or_interstitial_visible,
        "prompt_requirement_match": str(analysis.get("prompt_requirement_match") or "").strip().lower(),
        "expected_visual_target": str(
            analysis.get("expected_visual_target")
            or analysis.get("expected_asset")
            or event.get("expected_visual_target")
            or event.get("expected_asset")
            or ""
        ).strip(),
        "observed_visual_target": str(
            analysis.get("observed_visual_target")
            or analysis.get("observed_asset")
            or analysis.get("visible_asset")
            or event.get("observed_visual_target")
            or event.get("observed_asset")
            or ""
        ).strip(),
        "target_match": analysis.get("target_match", event.get("target_match")),
        "fresh_after_prompt": event.get("fresh_after_prompt", analysis.get("fresh_after_prompt")),
    }


def build_visual_evidence(visual_context: Dict[str, Any]) -> Dict[str, Any]:
    timeline = list(visual_context.get("timeline") or [])
    observations: List[Dict[str, Any]] = [observation_from_event(item) for item in timeline]
    analysis = dict(visual_context.get("analysis") or {})
    if analysis:
        latest_event = {
            "captured_at": visual_context.get("captured_at") or "",
            "prompt_timestamp": visual_context.get("prompt_timestamp") or "",
            "prompt_id": visual_context.get("prompt_id"),
            "prompt_sequence_id": visual_context.get("prompt_sequence_id"),
            "fresh_after_prompt": visual_context.get("fresh_after_prompt"),
            "expected_visual_target": visual_context.get("expected_visual_target") or "",
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
    positive = [
        item
        for item in observations
        if not item.get("ad_or_interstitial_visible")
        and (item.get("requirement_seen") or item.get("recommended_result") == "pass")
    ]
    negative = [
        item
        for item in observations
        if item.get("ad_or_interstitial_visible")
        or item.get("recommended_result") == "fail"
        or str(item.get("prompt_requirement_match") or "").lower() in {"mismatch", "no", "false", "different"}
        or item.get("target_match") is False
    ]
    return {
        "source": visual_context.get("source") or latest.get("source") or "unknown",
        "continuous_frame_count": int(visual_context.get("continuous_frame_count") or len(observations) or 0),
        "latest_observation": latest,
        "observations": observations,
        "positive_observations": positive,
        "negative_observations": negative,
        "capture_status": dict(visual_context.get("capture_status") or {}),
        "summary": visual_context.get("summary") or latest.get("visual_summary") or "",
        "prompt_id": visual_context.get("prompt_id"),
        "prompt_sequence_id": visual_context.get("prompt_sequence_id"),
        "prompt_timestamp": visual_context.get("prompt_timestamp") or "",
        "fresh_after_prompt": visual_context.get("fresh_after_prompt"),
        "expected_visual_target": visual_context.get("expected_visual_target") or latest.get("expected_visual_target") or "",
    }
