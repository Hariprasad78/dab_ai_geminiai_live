"""Parse YTS terminal prompts into normalized runtime inputs."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List


_OPTION_RE = re.compile(r"^\s*(?:option\s*)?([0-9A-Za-z]+)\s*[:.)-]\s*(.+?)\s*$", re.IGNORECASE)
_CONTROL_RE = re.compile(r"\b(failed|marked\s+by\s+user|retry|done|previous\s+selection)\b", re.IGNORECASE)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def prompt_hash(prompt_text: str, options: Iterable[Any] | None = None) -> str:
    normalized_options = [_normalize_text(str(option)) for option in (options or []) if _normalize_text(str(option))]
    payload = _normalize_text(prompt_text) + "\n" + "\n".join(normalized_options)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _option_labels_from_prompt(prompt_text: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for line in str(prompt_text or "").splitlines():
        match = _OPTION_RE.match(line)
        if not match:
            continue
        option = match.group(1).strip()
        label = _normalize_text(match.group(2)).lower()
        if option and label:
            labels[option] = label
    return labels


def parse_yts_prompt(prompt_text: str, options: Iterable[Any] | None = None) -> Dict[str, Any]:
    """Return a JSON-friendly prompt record with normalized options and hash."""

    provided = [_normalize_text(str(option)) for option in (options or []) if _normalize_text(str(option))]
    labels = _option_labels_from_prompt(prompt_text)
    allowed: List[Dict[str, str]] = []

    if labels:
        for option, label in labels.items():
            if provided and option not in provided:
                continue
            allowed.append({"option": option, "label": label})
    elif provided:
        allowed = [{"option": option, "label": option.lower()} for option in provided]

    seen = set()
    deduped: List[Dict[str, str]] = []
    for item in allowed:
        option = str(item.get("option") or "").strip()
        if not option or option in seen:
            continue
        seen.add(option)
        deduped.append({"option": option, "label": str(item.get("label") or option).strip().lower()})

    hash_value = prompt_hash(prompt_text, [item["option"] for item in deduped] or provided)
    test_title = _normalize_text(str(prompt_text or "").splitlines()[0] if str(prompt_text or "").splitlines() else "")
    return {
        "prompt_text": str(prompt_text or ""),
        "prompt_hash": hash_value,
        "test_key": hash_value,
        "test_title": test_title or "YTS guided prompt",
        "prompt_kind": classify_yts_prompt_kind(prompt_text),
        "allowed_answers": deduped,
    }


def classify_yts_prompt_kind(prompt_text: str) -> str:
    """Classify prompt flow without hardcoding individual YTS test cases."""

    text = str(prompt_text or "")
    lowered = text.lower()
    if _CONTROL_RE.search(text):
        return "result_retry_control"
    if re.search(r"\b(pass|fail|expected|actual|visible|shown|render|match|validate|validation|playback|video)\b", lowered):
        return "visual_validation"
    if re.search(r"\b(choose|select|enter choice|enter selection|yes/no|continue|proceed)\b", lowered):
        return "control"
    return "unknown"
