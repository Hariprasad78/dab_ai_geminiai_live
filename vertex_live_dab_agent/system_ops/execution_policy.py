"""Capability-aware routing policy for choosing DAB vs ADB execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from vertex_live_dab_agent.system_ops.capabilities import has_operation
from vertex_live_dab_agent.system_ops.fuzzy_matcher import fuzzy_match


@dataclass(frozen=True)
class RouteDecision:
    transport: str  # dab | adb | unsupported
    reason: str
    normalized: Dict[str, Any]


class CapabilityRouter:
    """Decides legal execution path per action using capability snapshot."""

    def __init__(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        self._snapshot = snapshot or {}

    @property
    def snapshot(self) -> Dict[str, Any]:
        return self._snapshot

    def update_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._snapshot = snapshot or {}

    def resolve_operation(self, operation: str) -> RouteDecision:
        op = str(operation or "").strip()
        if not op:
            return RouteDecision("unsupported", "missing operation", {"operation": op})
        if has_operation(self._snapshot, op):
            return RouteDecision("dab", "operation supported by DAB", {"operation": op})
        best, score = fuzzy_match(op, self._snapshot.get("supported_operations") or [], cutoff=0.82)
        if best and score >= 0.88:
            return RouteDecision("dab", f"mapped operation '{op}' -> '{best}'", {"operation": best, "confidence": score})
        return RouteDecision("unsupported", f"operation '{op}' not in capability snapshot", {"operation": op})

    def route_setting_operation(
        self,
        *,
        operation: str,
        setting_key: Optional[str],
        can_use_adb: bool,
        is_android: bool,
        adb_only_keys: Optional[Iterable[str]] = None,
    ) -> RouteDecision:
        op_decision = self.resolve_operation(operation)
        key = str(setting_key or "").strip()
        normalized = {"operation": operation, "setting_key": key}

        settings_map = self._snapshot.get("supported_settings") or {}
        supports_key = (not key) or (key in settings_map and bool((settings_map.get(key) or {}).get("supported", True)))
        if op_decision.transport == "dab" and supports_key:
            return RouteDecision("dab", "setting operation supported by DAB", normalized)

        adb_fallback_keys = {str(k).strip().lower() for k in (adb_only_keys or ["timezone", "time_zone", "time-zone", "time zone"]) if str(k).strip()}
        if is_android and can_use_adb and key.lower() in adb_fallback_keys:
            return RouteDecision("adb", "DAB unsupported/incomplete; using Android ADB fallback for setting", normalized)

        if not is_android and can_use_adb:
            return RouteDecision("unsupported", "ADB available but disallowed for non-Android platform", normalized)
        return RouteDecision("unsupported", "no legal transport for requested setting operation", normalized)
