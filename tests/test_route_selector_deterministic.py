import pytest

from vertex_live_dab_agent.system_ops.requirement_resolver import RequirementResolution, RequirementType
from vertex_live_dab_agent.system_ops.route_selector import RouteName, select_route


def _snapshot(ops=None, settings=None):
    return {
        "supported_operations": ops or [],
        "supported_settings": settings or {},
        "supported_keys": [],
        "installed_applications": [],
    }


def test_direct_dab_selected_for_setting_when_supported():
    req = RequirementResolution(requirement_type=RequirementType.CHANGE_SETTING, target_setting="language", target_value="Kannada")
    route = select_route(
        requirement=req,
        snapshot=_snapshot(
            ops=["system/settings/get", "system/settings/set"],
            settings={"language": {"supported": True}},
        ),
        ui_navigation_allowed=False,
        is_android=False,
        adb_available=False,
    )
    assert route.route == RouteName.DIRECT_DAB


def test_adb_selected_only_for_android_with_adb():
    req = RequirementResolution(requirement_type=RequirementType.CHANGE_SETTING, target_setting="language", target_value="Kannada")
    route = select_route(
        requirement=req,
        snapshot=_snapshot(ops=[], settings={}),
        ui_navigation_allowed=False,
        is_android=True,
        adb_available=True,
    )
    assert route.route == RouteName.ADB_FALLBACK


def test_ui_selected_only_when_checkbox_enabled():
    req = RequirementResolution(requirement_type=RequirementType.CHANGE_SETTING, target_setting="language", target_value="Kannada")
    route = select_route(
        requirement=req,
        snapshot=_snapshot(ops=["applications/launch", "input/key-press"], settings={}),
        ui_navigation_allowed=True,
        is_android=False,
        adb_available=False,
    )
    assert route.route == RouteName.UI_NAVIGATION_FALLBACK


def test_blocked_when_no_legal_route():
    req = RequirementResolution(requirement_type=RequirementType.CHANGE_SETTING, target_setting="language", target_value="Kannada")
    route = select_route(
        requirement=req,
        snapshot=_snapshot(ops=[], settings={}),
        ui_navigation_allowed=False,
        is_android=False,
        adb_available=False,
    )
    assert route.route == RouteName.UNSATISFIABLE_WITH_CURRENT_CAPABILITIES
    assert "DIRECT_DAB" in route.rejected_routes
