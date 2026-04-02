from vertex_live_dab_agent.system_ops.request_normalizer import normalize_user_request
from vertex_live_dab_agent.system_ops.requirement_resolver import resolve_requirement
from vertex_live_dab_agent.system_ops.route_selector import RouteName, select_route


def test_yts_instruction_uses_shared_requirement_and_route_logic():
    normalized = normalize_user_request("set launguge to kannada")
    requirement = resolve_requirement(normalized.normalized_user_goal, {
        "action_type": normalized.action_type,
        "target_setting": normalized.target_setting,
        "target_value": normalized.target_value,
        "target_app": normalized.target_app,
    })
    route = select_route(
        requirement=requirement,
        snapshot={
            "supported_operations": ["system/settings/get", "system/settings/set"],
            "supported_settings": {"language": {"supported": True}},
            "supported_keys": [],
            "installed_applications": [],
        },
        ui_navigation_allowed=False,
        is_android=False,
        adb_available=False,
    )

    assert requirement.requirement_type == "CHANGE_SETTING"
    assert route.route == RouteName.DIRECT_DAB
