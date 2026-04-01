"""Routing for system operations between DAB and Android ADB fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class ExecutionDecision:
    method: str  # "dab" | "adb" | "unsupported"
    reason: str


def operation_supported_by_dab(supported_operations: Iterable[str], operation: str) -> bool:
    target = str(operation or "").strip().lower()
    if not target:
        return False
    return any(target in str(item or "").strip().lower() for item in (supported_operations or []))


def is_timezone_key(setting_key: Optional[str]) -> bool:
    key = str(setting_key or "").strip().lower()
    return key in {"timezone", "time_zone", "time-zone", "time zone"}


def has_android_adb_fallback(operation: str, setting_key: Optional[str]) -> bool:
    op = str(operation or "").strip().lower()
    if op not in {"system/settings/get", "system/settings/set", "system/settings/list"}:
        return False
    return is_timezone_key(setting_key)


def resolve_execution_method(*, is_android: bool, dab_supported: bool, adb_fallback_available: bool) -> ExecutionDecision:
    if dab_supported:
        return ExecutionDecision(method="dab", reason="operation supported by DAB")
    if is_android and adb_fallback_available:
        return ExecutionDecision(method="adb", reason="DAB unsupported; Android ADB fallback available")
    if not is_android and adb_fallback_available:
        return ExecutionDecision(method="unsupported", reason="ADB fallback is Android-only")
    return ExecutionDecision(method="unsupported", reason="operation unsupported by DAB and no ADB fallback mapping")
