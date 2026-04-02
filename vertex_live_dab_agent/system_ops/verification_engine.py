"""Post-action verification helpers for capability-aware agent flows."""

from __future__ import annotations

from typing import Any


def normalize_value_for_compare(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def setting_verification_passed(expected: Any, observed: Any) -> bool:
    return normalize_value_for_compare(expected) == normalize_value_for_compare(observed)
