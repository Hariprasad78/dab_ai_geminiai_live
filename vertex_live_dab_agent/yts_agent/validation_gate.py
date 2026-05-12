"""Evidence gates that prevent unsafe YTS Pass decisions."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from vertex_live_dab_agent.yts_agent.utils import dedupe_strings, normalize_missing_evidence, resolve_option_label

def _label_for_option(option: str, expectation: Dict[str, Any]) -> str:
    return resolve_option_label(option, expectation.get("allowed_answers") or []).lower()


def _requires_youtube(expectation: Dict[str, Any]) -> bool:
    context = str(expectation.get("required_app_context") or "").lower()
    text = " ".join(
        [
            str(expectation.get("test_type") or ""),
            str(expectation.get("required_state") or ""),
            " ".join(str(req.get("description") or "") for req in expectation.get("visual_requirements") or [] if isinstance(req, dict)),
        ]
    ).lower()
    return "youtube" in context or bool(re.search(r"\b(in-app|in app|youtube|video|playback|player)\b", text))


def _requires_playback(expectation: Dict[str, Any]) -> bool:
    state = str(expectation.get("required_state") or "").lower()
    test_type = str(expectation.get("test_type") or "").lower()
    text = " ".join(str(req.get("description") or "") for req in expectation.get("visual_requirements") or [] if isinstance(req, dict)).lower()
    return "video_playback_active" in state or test_type == "playback" or bool(re.search(r"\b(playback|playing|video|player|render|frame)\b", text))


def validate_decision_gate(expectation: Dict[str, Any], evidence: Dict[str, Any], decision: Dict[str, Any], *, min_confidence: float = 0.70) -> Dict[str, Any]:
    selected = str(decision.get("selected_option") or "").strip()
    label = str(decision.get("selected_label") or _label_for_option(selected, expectation)).strip().lower()
    resolved_label = _label_for_option(selected, expectation)
    if resolved_label:
        label = resolved_label
    latest = dict(evidence.get("latest_observation") or {})
    missing: List[str] = normalize_missing_evidence(decision.get("missing_evidence"))
    blocked = False

    if "pass" not in label:
        return {
            "allowed": True,
            "safety_blocked_pass": False,
            "missing_evidence": missing,
            "reason": f"Selected label '{label or selected}' is not a Pass label for this prompt, so Pass safety blocking was not needed.",
        }

    confidence = float(latest.get("confidence") or decision.get("confidence") or 0.0)
    current_context = str(latest.get("detected_app_context") or "").lower()
    screen_type = str(latest.get("screen_type") or "").lower()
    if _requires_youtube(expectation):
        youtube_active = latest.get("youtube_active")
        if youtube_active is not True or "launcher" in current_context or "launcher" in screen_type or "system" in current_context:
            blocked = True
            missing.append("Current live TV feed is not positively verified as the required YouTube/in-app context.")
    if _requires_playback(expectation) and latest.get("video_playback_active") is not True:
        blocked = True
        missing.append("Video playback is not positively verified as active in the live TV feed.")
    requirements = [req for req in expectation.get("visual_requirements") or [] if isinstance(req, dict) and req.get("evidence_required", True)]
    if requirements and not evidence.get("positive_observations"):
        blocked = True
        missing.append("No continuous visual observation positively confirmed the prompt requirement.")
    if confidence < min_confidence and not evidence.get("positive_observations"):
        blocked = True
        missing.append(f"Latest visual confidence {confidence:.2f} is below the Pass threshold.")
    if evidence.get("negative_observations"):
        blocked = True
        missing.append("Continuous visual history contains observations that contradict Pass.")

    return {
        "allowed": not blocked,
        "safety_blocked_pass": blocked,
        "missing_evidence": dedupe_strings(missing),
        "reason": "Pass blocked by live-evidence safety gate." if blocked else "Pass allowed because required live evidence was positively verified.",
    }
