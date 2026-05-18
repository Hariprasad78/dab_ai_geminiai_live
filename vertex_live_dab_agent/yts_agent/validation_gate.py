"""Evidence gates that prevent unsafe YTS Pass decisions."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List

from vertex_live_dab_agent.yts_agent.utils import coerce_confidence, dedupe_strings, normalize_missing_evidence, resolve_option_label

_AD_SIGNAL_RE = re.compile(
    r"\bads?\b|\bad\s+\d+\s+of\s+\d+\b|\badvert(?:isement|ising)?\b|"
    r"\bcommercial\s+break\b|\bsponsored\b|\bsponsor\b|\bpromo(?:tion)?\b|"
    r"\bskip\s+ads?\b|\bvisit\s+advertiser\b|\bvideo\s+will\s+play\s+after\b|"
    r"\bwww\.[a-z0-9.-]+\.[a-z]{2,}\b",
    re.IGNORECASE,
)


def _contains_ad_signal(*values: Any) -> bool:
    return any(_AD_SIGNAL_RE.search(str(value or "")) for value in values)


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
    return "video_playback_active" in state or test_type == "playback" or bool(re.search(r"\b(playback|playing|video|player|pause|resume|seek|buffer)\b", text))


def _requires_visual_validation(expectation: Dict[str, Any]) -> bool:
    test_type = str(expectation.get("test_type") or "").strip().lower()
    if expectation.get("expected_visual_target") or expectation.get("visual_requirements"):
        return True
    if _requires_youtube(expectation) or _requires_playback(expectation):
        return True
    return bool(re.search(r"\b(render|visual|image|video|playback|subtitle|caption|quality|thumbnail|player|youtube)\b", test_type))


def _is_pass_like_label(label: str) -> bool:
    return bool(re.search(r"\b(pass|yes|correct|ok|okay|success|succeed|true)\b", str(label or "").lower()))


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_target(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _target_matches(expected: str, observed: str) -> bool:
    exp = _normalize_target(expected)
    obs = _normalize_target(observed)
    if not exp or not obs:
        return False
    return exp == obs or exp in obs or obs in exp


def _missing_required_text_terms(expectation: Dict[str, Any], latest: Dict[str, Any]) -> List[str]:
    descriptions = " ".join(
        str(req.get("description") or "")
        for req in expectation.get("visual_requirements") or []
        if isinstance(req, dict)
    )
    required_terms = [
        term.strip()
        for term in re.findall(r"['\"]([^'\"]{2,80})['\"]", descriptions)
        if term.strip()
    ]
    evidence_text = " ".join(
        [
            str(latest.get("detected_text") or ""),
            str(latest.get("visual_summary") or ""),
            str(latest.get("requirement_evidence") or ""),
            str(latest.get("observed_visual_target") or ""),
        ]
    ).lower()
    missing: List[str] = []
    for term in required_terms:
        normalized = re.sub(r"\s+", " ", term).strip().lower()
        if normalized and normalized not in evidence_text:
            missing.append(term)
    return missing


def validate_decision_gate(expectation: Dict[str, Any], evidence: Dict[str, Any], decision: Dict[str, Any], *, min_confidence: float = 0.70) -> Dict[str, Any]:
    selected = str(decision.get("selected_option") or "").strip()
    label = str(decision.get("selected_label") or _label_for_option(selected, expectation)).strip().lower()
    resolved_label = _label_for_option(selected, expectation)
    if resolved_label:
        label = resolved_label
    latest = dict(evidence.get("latest_observation") or {})
    missing: List[str] = normalize_missing_evidence(decision.get("missing_evidence"))
    blocked = False

    if not _is_pass_like_label(label):
        return {
            "allowed": True,
            "safety_blocked_pass": False,
            "missing_evidence": missing,
            "reason": f"Selected label '{label or selected}' is not a Pass label for this prompt, so Pass safety blocking was not needed.",
        }

    confidence = coerce_confidence(latest.get("confidence"), coerce_confidence(decision.get("confidence")))
    current_context = str(latest.get("detected_app_context") or "").lower()
    screen_type = str(latest.get("screen_type") or "").lower()
    visual_summary = str(latest.get("visual_summary") or latest.get("requirement_evidence") or evidence.get("summary") or "")
    detected_text = str(latest.get("detected_text") or "")
    observed_target = str(latest.get("observed_visual_target") or "").strip()
    target_match: bool | None = None
    visual_required = _requires_visual_validation(expectation)
    ad_detected = bool(latest.get("ad_or_interstitial_visible")) or _contains_ad_signal(
        visual_summary,
        detected_text,
        current_context,
        screen_type,
        observed_target,
    )
    if not ad_detected:
        for item in list(evidence.get("observations") or [])[-5:]:
            if bool(item.get("ad_or_interstitial_visible")) or _contains_ad_signal(
                item.get("visual_summary"),
                item.get("detected_text"),
                item.get("detected_app_context"),
                item.get("screen_type"),
                item.get("observed_visual_target"),
            ):
                ad_detected = True
                break
    if ad_detected:
        blocked = True
        missing.append("Live TV feed shows an ad, sponsored screen, or interstitial instead of the required YTS target.")

    frame_ts = _parse_timestamp(latest.get("timestamp"))
    prompt_ts = _parse_timestamp(latest.get("prompt_timestamp") or evidence.get("prompt_timestamp"))
    if prompt_ts is not None:
        if frame_ts is None or frame_ts <= prompt_ts:
            blocked = True
            missing.append("Latest TV frame was not captured after the current prompt appeared.")
    elif evidence.get("fresh_after_prompt") is not True and latest.get("fresh_after_prompt") is not True:
        blocked = True
        missing.append("Latest TV frame is not explicitly bound to the current prompt.")

    expected_target = str(expectation.get("expected_visual_target") or evidence.get("expected_visual_target") or latest.get("expected_visual_target") or "").strip()
    target_match_raw = latest.get("target_match")
    prompt_match = str(latest.get("prompt_requirement_match") or "").strip().lower()
    if expected_target:
        if isinstance(target_match_raw, bool):
            target_match = target_match_raw
        elif str(target_match_raw).strip().lower() in {"true", "yes", "match", "matched"}:
            target_match = True
        elif str(target_match_raw).strip().lower() in {"false", "no", "mismatch", "different"}:
            target_match = False
        else:
            target_match = _target_matches(expected_target, observed_target)
        if not observed_target:
            blocked = True
            missing.append(f"Fresh frame did not identify the visible target for expected asset '{expected_target}'.")
        elif not target_match or prompt_match in {"mismatch", "no", "false", "different"}:
            blocked = True
            missing.append(f"Content mismatch: Expected asset '{expected_target}' did not match observed asset '{observed_target}'.")
    elif isinstance(target_match_raw, bool):
        target_match = target_match_raw
    elif str(target_match_raw).strip().lower() in {"true", "yes", "match", "matched"}:
        target_match = True
    elif str(target_match_raw).strip().lower() in {"false", "no", "mismatch", "different"}:
        target_match = False
    if visual_required and (target_match is False or prompt_match in {"mismatch", "no", "false", "different"}):
        blocked = True
        missing.append("Live TV feed shows the wrong content or does not match the prompt requirement.")
    if _requires_youtube(expectation):
        youtube_active = latest.get("youtube_active")
        if youtube_active is not True or "launcher" in current_context or "launcher" in screen_type or "system" in current_context:
            blocked = True
            missing.append("Current live TV feed is not positively verified as the required YouTube/in-app context.")
    if _requires_playback(expectation) and latest.get("video_playback_active") is not True:
        blocked = True
        missing.append("Video playback is not positively verified as active in the live TV feed.")
    requirements = [req for req in expectation.get("visual_requirements") or [] if isinstance(req, dict) and req.get("evidence_required", True)]
    missing_text_terms = _missing_required_text_terms(expectation, latest)
    if visual_required and missing_text_terms:
        blocked = True
        missing.append(f"Required on-screen text was not verified: {', '.join(missing_text_terms)}.")
    positive_observations = [
        item for item in (evidence.get("positive_observations") or []) if not item.get("ad_or_interstitial_visible")
    ]
    if requirements and not positive_observations:
        blocked = True
        missing.append("No continuous visual observation positively confirmed the prompt requirement.")
    if visual_required and not positive_observations and not (latest.get("requirement_seen") and not ad_detected) and not (expected_target and target_match is True):
        blocked = True
        missing.append("No positive live evidence confirmed the current YTS requirement.")
    if visual_required and str(latest.get("recommended_result") or "").strip().lower() == "fail":
        blocked = True
        missing.append("Latest visual analysis recommends Fail for the current YTS requirement.")
    if confidence < min_confidence and not positive_observations:
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
