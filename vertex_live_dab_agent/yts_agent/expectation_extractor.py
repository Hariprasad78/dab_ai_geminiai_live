"""Runtime extraction of YTS validation expectations."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if "```" in raw:
        candidates.extend(block.strip() for block in re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.I | re.S))
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _answer_label_map(allowed_answers: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in allowed_answers or []:
        if isinstance(item, dict):
            option = str(item.get("option") or item.get("value") or item.get("id") or "").strip()
            label = str(item.get("label") or item.get("text") or option).strip()
        else:
            option = str(item or "").strip()
            label = option
        if option:
            out.append({"option": option, "label": label})
    return out


def _infer_required_context(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(youtube|yt|in-app|in app|player|playback|video)\b", lowered):
        return "YouTube"
    if re.search(r"\b(browser|web\s*page|cobalt)\b", lowered):
        return "browser"
    if re.search(r"\b(settings|launcher|home\s*screen|system)\b", lowered):
        return "system"
    return "unknown"


def _infer_test_type(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(audio|sound|tone|volume|mute|loudness)\b", lowered):
        return "audio"
    if re.search(r"\b(playback|playing|video|buffer|seek|pause|resume|render|frame|aspect|visible|shown|match|image)\b", lowered):
        return "playback" if re.search(r"\b(playback|playing|video|pause|resume|seek|buffer)\b", lowered) else "visual"
    if re.search(r"\b(navigate|select|focus|menu|button|settings|dialog)\b", lowered):
        return "navigation"
    if re.search(r"\b(setup|connect|pair|install|permission)\b", lowered):
        return "setup"
    return "other"


def _extract_expected_visual_target(text: str) -> str:
    prompt = _compact(text)
    if not prompt:
        return ""
    parenthetical = [item.strip() for item in re.findall(r"\(([^()]{2,160})\)", prompt) if item.strip()]
    if parenthetical:
        return parenthetical[-1]
    quoted = [
        item.strip()
        for item in re.findall(r"['\"]([^'\"]{2,160})['\"]", prompt)
        if item.strip() and item.strip().lower() not in {"pass", "fail", "skip", "yes", "no"}
    ]
    if quoted:
        return quoted[-1]
    patterns = [
        r"\b(?:expected|target|reference)\s+(?:asset|image|video|visual|content)?\s*(?:is|=|:)\s*([^.;\n]{2,160})",
        r"\b(?:does|is)\s+the\s+(?:image|video|asset|visual|content)\s+(?:on\s+screen\s+)?(?:match|show|render)\s+([^?;\n]{2,160})",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.I)
        if match:
            candidate = _compact(match.group(1))
            candidate = re.sub(r"\s+(?:correctly|properly|on screen|now)$", "", candidate, flags=re.I).strip(" .,:;?")
            if candidate:
                return candidate
    return ""


def heuristic_extract_expectation(prompt_record: Dict[str, Any], guided_context: str = "") -> Dict[str, Any]:
    prompt_text = str(prompt_record.get("prompt_text") or "")
    combined = "\n".join(part for part in [guided_context, prompt_text] if str(part or "").strip())
    prompt_kind = str(prompt_record.get("prompt_kind") or "").strip()
    control_prompt = prompt_kind == "result_retry_control"
    test_type = "control" if control_prompt else _infer_test_type(combined)
    required_context = "unknown" if control_prompt else _infer_required_context(combined)
    requires_playback = bool(
        not control_prompt
        and re.search(r"\b(playback|playing|video|player|pause|resume|seek|buffer)\b", combined.lower())
    )
    requirement_source = combined if guided_context else prompt_text
    requirement_lines = [
        _compact(line)
        for line in requirement_source.splitlines()
        if _compact(line) and not re.match(r"^\s*(?:option\s*)?[0-9A-Za-z]+\s*[:.)-]", line, flags=re.I)
    ]
    description = " ".join(requirement_lines[:12]) or _compact(prompt_text) or "Validate the current YTS guided prompt."
    expected_visual_target = _extract_expected_visual_target(prompt_text)
    return {
        "test_title": str(prompt_record.get("test_title") or "YTS guided prompt"),
        "test_type": test_type,
        "required_app_context": required_context,
        "required_state": "video_playback_active" if requires_playback else "unknown",
        "expected_visual_target": expected_visual_target,
        "visual_requirements": [
            {
                "description": description,
                "must_be_visible": test_type in {"visual", "playback", "navigation"},
                "time_window": None,
                "evidence_required": test_type in {"visual", "playback", "navigation"},
            }
        ],
        "negative_conditions": [
            "launcher visible",
            "home screen visible",
            "wrong app",
            "advertisement playing",
            "sponsored content visible",
            "visit advertiser visible",
            "ad countdown visible",
            "no video playback",
            "unrelated screen",
            "required visual evidence missing",
        ],
        "allowed_answers": _answer_label_map(prompt_record.get("allowed_answers") or []),
        "minimum_evidence_policy": "Pass requires positive proof for all required conditions from the current live TV feed.",
        "source": "heuristic",
    }


def normalize_expectation(raw: Dict[str, Any], prompt_record: Dict[str, Any], guided_context: str = "") -> Dict[str, Any]:
    expectation = dict(raw or {})
    fallback = heuristic_extract_expectation(prompt_record, guided_context)
    for key, value in fallback.items():
        if key not in expectation or expectation.get(key) in (None, "", []):
            expectation[key] = value
    expectation["expected_visual_target"] = _compact(
        str(expectation.get("expected_visual_target") or expectation.get("expected_asset") or fallback.get("expected_visual_target") or "")
    )
    if not isinstance(expectation.get("visual_requirements"), list):
        expectation["visual_requirements"] = fallback["visual_requirements"]
    if not isinstance(expectation.get("negative_conditions"), list):
        expectation["negative_conditions"] = fallback["negative_conditions"]
    expectation["allowed_answers"] = _answer_label_map(expectation.get("allowed_answers") or prompt_record.get("allowed_answers") or [])
    normalized_requirements: List[Dict[str, Any]] = []
    for item in expectation.get("visual_requirements") or []:
        if isinstance(item, dict):
            normalized_requirements.append(item)
        elif str(item or "").strip():
            normalized_requirements.append(
                {
                    "description": str(item).strip(),
                    "must_be_visible": True,
                    "time_window": None,
                    "evidence_required": True,
                }
            )
    expectation["visual_requirements"] = normalized_requirements or fallback["visual_requirements"]
    expectation["negative_conditions"] = [
        str(item).strip()
        for item in (expectation.get("negative_conditions") or fallback["negative_conditions"])
        if str(item).strip()
    ]
    expectation["minimum_evidence_policy"] = str(
        expectation.get("minimum_evidence_policy")
        or "Pass requires positive proof for all required conditions from the current live TV feed."
    )
    return expectation


def extract_expectation_from_model_response(response_text: str, prompt_record: Dict[str, Any], guided_context: str = "") -> Dict[str, Any]:
    parsed = _json_object(response_text)
    if parsed is None:
        return heuristic_extract_expectation(prompt_record, guided_context)
    parsed["source"] = parsed.get("source") or "gemini"
    return normalize_expectation(parsed, prompt_record, guided_context)


def build_expectation_extraction_prompt(prompt_record: Dict[str, Any], guided_context: str, previous_memory: Dict[str, Any] | None = None) -> str:
    memory = previous_memory or {}
    return "\n\n".join(
        [
            "You are a test-agnostic YTS guided validation parser.",
            "Read the runtime YTS console prompt and extract what must be true before Pass is allowed.",
            "Do not hardcode test case names, app text, artwork names, or known YTS cases. Infer expectations only from the prompt, logs, and metadata.",
            "Return strict JSON with keys: test_title, test_type, required_app_context, required_state, expected_visual_target, visual_requirements, negative_conditions, allowed_answers, minimum_evidence_policy.",
            "Use required_app_context='YouTube' when the prompt requires in-app YouTube playback or video evidence. Use 'unknown' only when the prompt gives no app requirement.",
            "expected_visual_target MUST be the specific video title, test pattern, or asset name required by this test (e.g., '4:3 test video', 'SDR 60fps'). Do not leave it blank if the test involves a specific video.",
            "Preserve every concrete visual requirement from the active YTS log, including quoted text that must be visible, named objects, and time windows like 0:06-0:10.",
            "negative_conditions MUST include 'wrong video playing', 'advertisement playing', 'screensaver visible'.",
            "minimum_evidence_policy must say that Pass requires positive proof from the current live TV feed.",
            f"YTS prompt record:\n{json.dumps(prompt_record, ensure_ascii=False)}",
            f"Guided metadata/log context:\n{guided_context or '(none)'}",
            f"Previous memory for this test, as context only:\n{json.dumps(memory, ensure_ascii=False)[:12000]}",
        ]
    )
