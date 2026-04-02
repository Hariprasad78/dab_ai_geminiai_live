from vertex_live_dab_agent.system_ops.execution_controller import enforce_action_guards


def test_no_keypress_when_keys_empty_and_no_fallback():
    ok, reason = enforce_action_guards(
        action="PRESS_RED",
        ui_navigation_allowed=True,
        chosen_route="UI_NAVIGATION_FALLBACK",
        supported_keys=[],
        snapshot={"supported_operations": []},
        is_android=False,
        requested_route="UI_NAVIGATION_FALLBACK",
    )
    assert ok is False
    assert "keypress forbidden" in reason


def test_no_ui_route_when_checkbox_false():
    ok, reason = enforce_action_guards(
        action="PRESS_RIGHT",
        ui_navigation_allowed=False,
        chosen_route="UI_NAVIGATION_FALLBACK",
        supported_keys=[],
        snapshot={"supported_operations": ["input/key-press"]},
        is_android=False,
        requested_route="UI_NAVIGATION_FALLBACK",
    )
    assert ok is False
    assert "UI fallback forbidden" in reason


def test_no_adb_route_on_non_android():
    ok, reason = enforce_action_guards(
        action="GET_SETTING",
        ui_navigation_allowed=False,
        chosen_route="ADB_FALLBACK",
        supported_keys=[],
        snapshot={"supported_operations": ["system/settings/get"]},
        is_android=False,
        requested_route="ADB_FALLBACK",
    )
    assert ok is False
    assert "non-Android" in reason
