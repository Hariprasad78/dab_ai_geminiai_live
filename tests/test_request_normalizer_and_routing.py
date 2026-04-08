import pytest

from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState
from vertex_live_dab_agent.system_ops.request_normalizer import normalize_user_request


class _Cfg:
    dab_device_id = "mock-device"
    youtube_app_id = "youtube"


def _orch():
    orch = object.__new__(Orchestrator)
    orch._config = _Cfg()
    return orch


def test_typo_normalization_language_kannada():
    normalized = normalize_user_request("set launguge to kannada")
    assert normalized.corrected_user_text == "set language to Kannada"
    assert normalized.action_type == "change_setting"
    assert normalized.target_setting == "language"
    assert normalized.target_value == "Kannada"


def test_language_goal_is_settings_goal():
    assert Orchestrator._is_settings_goal("set language to Kannada") is True


def test_open_app_normalization_detects_action_type():
    normalized = normalize_user_request("open youtube")
    assert normalized.corrected_user_text == "open youtube"
    assert normalized.action_type == "open_app"
    assert normalized.target_app == "youtube"


@pytest.mark.asyncio
async def test_route_prefers_direct_dab_when_supported():
    orch = _orch()
    state = RunState(goal="set launguge to kannada")
    state.supported_operations = ["system/settings/set", "system/settings/get", "system/settings/list"]
    state.supported_settings = [{"key": "language", "friendlyName": "Language", "allowedValues": ["Kannada", "English"]}]
    state.capability_snapshot = {
        "supported_operations": state.supported_operations,
        "supported_settings": {"language": {"key": "language", "type": "list", "values": ["Kannada", "English"], "supported": True}},
        "supported_keys": [],
        "supported_voices": [],
        "installed_applications": [],
        "platform_type": "android_tv",
        "is_android": True,
        "can_use_adb": True,
        "unsupported_or_missing_capabilities": [],
    }
    state.is_android_device = True
    state.android_adb_device_id = "127.0.0.1:5555"

    batch = await orch._select_execution_strategy(state)
    assert state.chosen_route == "DIRECT_DAB"
    assert state.requires_post_check is True
    assert state.can_mark_done_now is False
    assert [b["action"] for b in batch][:2] == ["SET_SETTING", "GET_SETTING"]


@pytest.mark.asyncio
async def test_non_android_never_selects_adb():
    orch = _orch()
    state = RunState(goal="set timezone to UTC")
    state.supported_operations = []
    state.supported_settings = [{"key": "timezone", "friendlyName": "Time Zone", "allowedValues": ["UTC"]}]
    state.capability_snapshot = {
        "supported_operations": [],
        "supported_settings": {"timezone": {"key": "timezone", "type": "list", "values": ["UTC"], "supported": True}},
        "supported_keys": [],
        "supported_voices": [],
        "installed_applications": [],
        "platform_type": "roku",
        "is_android": False,
        "can_use_adb": False,
        "unsupported_or_missing_capabilities": [],
    }
    state.is_android_device = False
    state.android_adb_device_id = None
    state.ui_navigation_allowed = False

    _ = await orch._select_execution_strategy(state)
    assert state.chosen_route is None
    assert "ADB_FALLBACK" in state.route_rejection_reasons


@pytest.mark.asyncio
async def test_ui_fallback_requires_checkbox_and_launches_settings_first():
    orch = _orch()
    state = RunState(goal="set language to Kannada")
    state.supported_operations = ["applications/launch"]
    state.supported_settings = []
    state.capability_snapshot = {
        "supported_operations": ["applications/launch"],
        "supported_settings": {},
        "supported_keys": [],
        "supported_voices": [],
        "installed_applications": [{"appId": "settings", "friendlyName": "Settings"}],
        "platform_type": "linux-tv",
        "is_android": False,
        "can_use_adb": False,
        "unsupported_or_missing_capabilities": [],
    }
    state.ui_navigation_allowed = True

    batch = await orch._select_execution_strategy(state)
    assert state.chosen_route == "UI_NAVIGATION_FALLBACK"
    assert batch and batch[0]["action"] == "LAUNCH_APP"
    assert str(batch[0]["params"].get("app_id", "")).lower() == "settings"


@pytest.mark.asyncio
async def test_open_app_goal_marks_done_when_already_foreground():
    orch = _orch()
    state = RunState(goal="open youtube")
    state.step_count = 1
    state.current_app = "youtube"
    state.current_app_id = "youtube"
    state.current_app_state = "FOREGROUND"

    batch = await orch._select_execution_strategy(state)
    assert batch and batch[0]["action"] == "DONE"


def test_open_app_goal_verification_requires_target_match():
    orch = _orch()
    state = RunState(goal="open youtube")
    state.current_app = "netflix"
    state.current_app_id = "netflix"
    state.current_app_state = "FOREGROUND"

    assert orch._is_app_goal_verified(state) is False


def test_infer_target_app_name_from_goal_without_catalog_uses_goal_hint():
    orch = _orch()
    assert orch._infer_target_app_name_from_goal("open youtube", []) == "youtube"
