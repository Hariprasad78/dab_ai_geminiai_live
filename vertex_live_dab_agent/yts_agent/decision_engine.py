"""Evidence-first YTS prompt decision engine."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from vertex_live_dab_agent.yts_agent.utils import coerce_confidence, normalize_missing_evidence, resolve_option_label
from vertex_live_dab_agent.yts_agent.validation_gate import validate_decision_gate


def _find_option(expectation: Dict[str, Any], fragments: Iterable[str]) -> Optional[Dict[str, str]]:
    wanted = [str(fragment).lower() for fragment in fragments]
    for item in expectation.get("allowed_answers") or []:
        option = str(item.get("option") or "").strip()
        label = str(item.get("label") or option).strip().lower()
        if option and any(fragment in label or fragment == option.lower() for fragment in wanted):
            return {"option": option, "label": label}
    return None


def _normalize_model_decision(model_decision: Dict[str, Any] | None, expectation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(model_decision, dict):
        return None
    selected = str(model_decision.get("selected_option") or model_decision.get("option") or "").strip()
    if not selected:
        return None
    allowed = {str(item.get("option") or "").strip(): str(item.get("label") or "").strip() for item in expectation.get("allowed_answers") or []}
    if selected not in allowed:
        return None
    resolved_label = resolve_option_label(selected, expectation.get("allowed_answers") or [])
    return {
        "selected_option": selected,
        "selected_label": resolved_label or selected,
        "confidence": coerce_confidence(model_decision.get("confidence")),
        "evidence_summary": str(model_decision.get("evidence_summary") or "").strip(),
        "missing_evidence": normalize_missing_evidence(model_decision.get("missing_evidence")),
        "reason": str(model_decision.get("reason") or "").strip(),
        "safety_blocked_pass": bool(model_decision.get("safety_blocked_pass")),
        "source": "gemini",
    }


def decide_yts_response(expectation: Dict[str, Any], evidence: Dict[str, Any], model_decision: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a safe terminal decision. Pass is never selected by default."""

    normalized_model = _normalize_model_decision(model_decision, expectation)
    if normalized_model:
        gate = validate_decision_gate(expectation, evidence, normalized_model)
        if gate.get("allowed"):
            normalized_model.update(gate)
            return normalized_model

    latest = dict(evidence.get("latest_observation") or {})
    recommended = str(latest.get("recommended_result") or "").lower()
    pass_option = _find_option(expectation, ["pass", "yes"])
    fail_option = _find_option(expectation, ["fail", "no"])
    skip_option = _find_option(expectation, ["skip"])

    if recommended == "pass" and pass_option:
        candidate = {
            "selected_option": pass_option["option"],
            "selected_label": pass_option["label"],
            "confidence": coerce_confidence(latest.get("confidence")),
            "evidence_summary": latest.get("requirement_evidence") or latest.get("visual_summary") or "",
            "missing_evidence": [],
            "reason": "Continuous visual monitor recommended Pass with positive evidence.",
            "safety_blocked_pass": False,
            "source": "evidence",
        }
        gate = validate_decision_gate(expectation, evidence, candidate)
        if gate.get("allowed"):
            candidate.update(gate)
            return candidate

    if fail_option:
        missing = []
        if normalized_model:
            missing.extend(normalize_missing_evidence(normalized_model.get("missing_evidence")))
        if latest:
            missing.append("Live TV evidence did not positively satisfy every Pass requirement.")
        else:
            missing.append("No live TV evidence was available.")
        return {
            "selected_option": fail_option["option"],
            "selected_label": fail_option["label"],
            "confidence": max(0.55, coerce_confidence(latest.get("confidence"))),
            "evidence_summary": latest.get("visual_summary") or evidence.get("summary") or "Evidence is insufficient for Pass.",
            "missing_evidence": list(dict.fromkeys(str(item) for item in missing if str(item).strip())),
            "reason": "Pass was not positively proven by current live TV evidence.",
            "safety_blocked_pass": bool(normalized_model),
            "source": "validation-gate",
        }

    if skip_option:
        return {
            "selected_option": skip_option["option"],
            "selected_label": skip_option["label"],
            "confidence": 0.5,
            "evidence_summary": evidence.get("summary") or "Validation cannot be completed safely.",
            "missing_evidence": ["No Fail option is available and Pass is not proven."],
            "reason": "Skipped because the agent cannot safely prove Pass and no Fail option is available.",
            "safety_blocked_pass": bool(normalized_model),
            "source": "validation-gate",
        }

    first = next(iter(expectation.get("allowed_answers") or []), {"option": "", "label": ""})
    return {
        "selected_option": str(first.get("option") or ""),
        "selected_label": str(first.get("label") or ""),
        "confidence": 0.0,
        "evidence_summary": evidence.get("summary") or "",
        "missing_evidence": ["No safe Pass/Fail/Skip option mapping was available."],
        "reason": "Fallback option selected only because no semantic safe option was available.",
        "safety_blocked_pass": bool(normalized_model),
        "source": "fallback",
    }
