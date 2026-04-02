"""Deterministic route selector: DIRECT_DAB > ADB_FALLBACK > UI_NAVIGATION_FALLBACK > BLOCKED."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from vertex_live_dab_agent.system_ops.requirement_resolver import RequirementResolution, RequirementType


class RouteName:
    DIRECT_DAB = "DIRECT_DAB"
    ADB_FALLBACK = "ADB_FALLBACK"
    UI_NAVIGATION_FALLBACK = "UI_NAVIGATION_FALLBACK"
    UNSATISFIABLE_WITH_CURRENT_CAPABILITIES = "UNSATISFIABLE_WITH_CURRENT_CAPABILITIES"


class RouteSelection(BaseModel):
    route: str
    rejected_routes: Dict[str, str] = Field(default_factory=dict)
    reason: str = ""


def _has_operation(snapshot: Dict[str, Any], op: str) -> bool:
    target = str(op).strip().lower()
    return any(target == str(x).strip().lower() for x in (snapshot.get("supported_operations") or []))


def select_route(*, requirement: RequirementResolution, snapshot: Dict[str, Any], ui_navigation_allowed: bool, is_android: bool, adb_available: bool) -> RouteSelection:
    rejected: Dict[str, str] = {}

    if requirement.requirement_type == RequirementType.CHANGE_SETTING:
        can_direct_set = _has_operation(snapshot, "system/settings/set")
        can_direct_get = _has_operation(snapshot, "system/settings/get")
        settings_map = snapshot.get("supported_settings") or {}
        key = str(requirement.target_setting or "").strip()
        key_known = bool(key and isinstance(settings_map, dict) and key in settings_map)

        if can_direct_set and can_direct_get and key_known:
            return RouteSelection(route=RouteName.DIRECT_DAB, rejected_routes=rejected, reason="direct setting operations available")
        rejected[RouteName.DIRECT_DAB] = "missing system/settings get/set or unresolved setting key"

        if is_android and adb_available:
            return RouteSelection(route=RouteName.ADB_FALLBACK, rejected_routes=rejected, reason="android adb fallback allowed")
        rejected[RouteName.ADB_FALLBACK] = "requires Android + ADB connectivity"

        if ui_navigation_allowed:
            can_launch = _has_operation(snapshot, "applications/launch")
            can_keypress = _has_operation(snapshot, "input/key-press")
            if can_launch or can_keypress:
                return RouteSelection(route=RouteName.UI_NAVIGATION_FALLBACK, rejected_routes=rejected, reason="ui fallback explicitly allowed")
            rejected[RouteName.UI_NAVIGATION_FALLBACK] = "missing both applications/launch and input/key-press"
        else:
            rejected[RouteName.UI_NAVIGATION_FALLBACK] = "disabled by user"

        return RouteSelection(
            route=RouteName.UNSATISFIABLE_WITH_CURRENT_CAPABILITIES,
            rejected_routes=rejected,
            reason="no legal route for requirement",
        )

    if requirement.requirement_type == RequirementType.OPEN_APP:
        if _has_operation(snapshot, "applications/launch"):
            return RouteSelection(route=RouteName.DIRECT_DAB, rejected_routes=rejected, reason="direct app launch available")
        rejected[RouteName.DIRECT_DAB] = "applications/launch unsupported"

        if is_android and adb_available:
            return RouteSelection(route=RouteName.ADB_FALLBACK, rejected_routes=rejected, reason="android adb fallback allowed")
        rejected[RouteName.ADB_FALLBACK] = "requires Android + ADB connectivity"

        if ui_navigation_allowed and _has_operation(snapshot, "input/key-press"):
            return RouteSelection(route=RouteName.UI_NAVIGATION_FALLBACK, rejected_routes=rejected, reason="ui fallback explicitly allowed")
        rejected[RouteName.UI_NAVIGATION_FALLBACK] = "disabled by user or missing input/key-press"

        return RouteSelection(
            route=RouteName.UNSATISFIABLE_WITH_CURRENT_CAPABILITIES,
            rejected_routes=rejected,
            reason="no legal route for app-open requirement",
        )

    rejected[RouteName.DIRECT_DAB] = "unknown requirement"
    rejected[RouteName.ADB_FALLBACK] = "unknown requirement"
    rejected[RouteName.UI_NAVIGATION_FALLBACK] = "unknown requirement"
    return RouteSelection(route=RouteName.UNSATISFIABLE_WITH_CURRENT_CAPABILITIES, rejected_routes=rejected, reason="unknown requirement")
