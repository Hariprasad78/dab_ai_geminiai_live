"""Regression tests for unsupported direct settings operation fallback."""

import pytest

import vertex_live_dab_agent.config as cfg_mod
from vertex_live_dab_agent.api.api import _build_final_diagnosis
from vertex_live_dab_agent.dab.client import DABResponse
from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import ActionRecord, RunState, RunStatus
from vertex_live_dab_agent.planner.schemas import PlannedAction


class _Cfg:
    dab_device_id = "mock-device"
    youtube_app_id = "youtube"
    session_timeout_seconds = 120


def _orch() -> Orchestrator:
    orch = object.__new__(Orchestrator)
    orch._config = _Cfg()
    return orch


class _FakeDABUnsupportedTimezoneSet:
    def __init__(self) -> None:
        self.calls = 0

    async def set_setting(self, setting_key, value):
        self.calls += 1
        return DABResponse(
            success=False,
            status=501,
            data={"error": "system/settings/set not supported", "setting": setting_key, "value": value},
            topic="dab/mock/system/settings/set",
            request_id="set-1",
        )


def test_direct_setting_failure_cached_after_repeated_known_unsupported_error() -> None:
    orch = _orch()
    state = RunState(goal="get timezone setting")
    state.strategy_selected = "DIRECT_SETTING_OPERATION"

    resp = DABResponse(
        success=False,
        status=500,
        data={"error": "Error: No shell command implementation at getCecEnabled()"},
        topic="dab/sony/system/settings/get",
        request_id="r1",
    )

    orch._record_direct_setting_failure(state, "system/settings/get", "timezone", resp)
    assert "system/settings/get:timezone" in state.unsupported_direct_operations
    assert state.strategy_selected == "DIRECT_SETTING_OPERATION"


def test_sanitize_redirects_unsupported_get_setting_to_key_navigation() -> None:
    orch = _orch()
    state = RunState(goal="check timezone")
    state.is_android_device = False
    state.unsupported_direct_operations["system/settings/get:timezone"] = {"reason": "unsupported"}

    planned = PlannedAction(
        action="GET_SETTING",
        confidence=0.9,
        reason="strategy:USE_DIRECT_DAB_OPERATION",
        params={"key": "timezone"},
    )

    fallback = orch._sanitize_planned_action_for_goal(state, planned)
    assert fallback.action == "FAILED"
    assert "disabled" in fallback.reason.lower()


@pytest.mark.asyncio
async def test_strategy_selection_uses_ui_fallback_when_direct_get_marked_unavailable() -> None:
    orch = _orch()
    state = RunState(goal="get timezone setting")
    state.supported_operations = ["system/settings/get", "system/settings/set", "input/key/list"]
    state.supported_settings = [{"key": "timezone", "friendlyName": "Time Zone", "writable": True}]
    state.is_android_device = False
    state.unsupported_direct_operations["system/settings/get:timezone"] = {"reason": "unsupported"}

    batch = await orch._select_execution_strategy(state)

    assert state.strategy_selected == "UNSUPPORTED_SETTING_OPERATION"
    assert batch
    assert batch[0]["action"] == "FAILED"


@pytest.mark.asyncio
async def test_strategy_selection_keeps_timezone_set_path_when_android_fallback_possible() -> None:
    orch = _orch()
    state = RunState(goal="set timezone to America/Los_Angeles")
    state.supported_operations = ["applications/list", "input/key/list"]
    state.supported_settings = [{"key": "timezone", "friendlyName": "Time Zone", "writable": True}]
    state.is_android_device = True
    state.android_adb_device_id = "emulator-5554"

    batch = await orch._select_execution_strategy(state)

    assert state.strategy_selected == "DIRECT_SETTING_OPERATION"
    assert batch
    assert batch[0]["action"] == "SET_SETTING"
    assert batch[0]["params"]["key"] == "timezone"


@pytest.mark.asyncio
async def test_strategy_selection_keeps_direct_setting_path_after_step_zero_for_settings_goals() -> None:
    orch = _orch()
    state = RunState(goal="set timezone to america/losangles")
    state.step_count = 1
    state.supported_operations = ["applications/list", "input/key/list"]
    state.supported_settings = [{"key": "timezone", "friendlyName": "Time Zone", "writable": True}]
    state.is_android_device = True
    state.android_adb_device_id = "emulator-5554"

    batch = await orch._select_execution_strategy(state)

    assert state.strategy_selected == "DIRECT_SETTING_OPERATION"
    assert batch
    assert batch[0]["action"] == "SET_SETTING"
    assert batch[0]["params"]["key"] == "timezone"


def test_timeout_defaults_are_120_seconds(monkeypatch) -> None:
    monkeypatch.delenv("DAB_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("SESSION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_STEP_TIMEOUT_SECONDS", raising=False)
    cfg_mod.reset_config()
    cfg = cfg_mod.get_config()
    assert cfg.dab_request_timeout == 120.0
    assert cfg.session_timeout_seconds == 120
    assert cfg.orchestrator_step_timeout_seconds == 120.0


def test_diagnosis_is_human_readable_for_ui_fallback_and_uncertain_verification() -> None:
    state = RunState(goal="Check timezone in settings")
    state.status = RunStatus.DONE
    state.current_screen = "Settings"
    state.latest_visual_summary = "Settings menu opened"
    state.ai_transcript = [
        {
            "type": "direct-op-unsupported",
            "operation": "system/settings/get",
            "setting_key": "timezone",
            "reason": "No shell command implementation",
        },
        {
            "type": "strategy-transition",
            "from": "DIRECT_SETTING_OPERATION",
            "to": "UI_NAVIGATION_FALLBACK",
            "reason": "unsupported",
        },
    ]
    state.action_history = [
        ActionRecord(
            step=0,
            action="GET_SETTING",
            params={"key": "timezone"},
            confidence=0.9,
            reason="direct",
            result="FAIL",
            timestamp="2026-01-01T00:00:00Z",
        ),
        ActionRecord(
            step=1,
            action="PRESS_HOME",
            params={},
            confidence=0.8,
            reason="fallback",
            result="PASS",
            timestamp="2026-01-01T00:00:01Z",
        ),
    ]

    diagnosis = _build_final_diagnosis(state)

    assert "ui navigation fallback" in diagnosis.goal_based_reason.lower()
    assert "not supported" in diagnosis.failure_reason_user_friendly.lower()
    assert "limited" in diagnosis.final_summary.lower() or "could not be confirmed" in diagnosis.failure_reason_user_friendly.lower()
    assert "finished successfully" not in diagnosis.final_summary.lower()


@pytest.mark.asyncio
async def test_timezone_set_uses_adb_fallback_when_android_and_dab_unavailable(monkeypatch) -> None:
    orch = _orch()
    orch._dab = _FakeDABUnsupportedTimezoneSet()
    state = RunState(goal="set timezone to America/Los_Angeles")
    state.supported_operations = ["applications/list", "input/key/list"]
    state.supported_settings = [{"key": "timezone", "friendlyName": "Time Zone", "allowedValues": ["America/Los_Angeles", "UTC"]}]
    state.is_android_device = True
    state.android_adb_device_id = "emulator-5554"

    async def _online(_device_id):
        return True, "device"

    async def _list_tz(_device_id):
        return {"success": False, "timezones": [], "error": "tzdata unavailable"}

    async def _set_tz(_device_id, tz_value):
        return {
            "success": True,
            "requested_timezone": tz_value,
            "observed_timezone": tz_value,
            "verified": True,
        }

    monkeypatch.setattr("vertex_live_dab_agent.orchestrator.orchestrator.is_adb_device_online", _online)
    monkeypatch.setattr("vertex_live_dab_agent.orchestrator.orchestrator.list_timezones_via_adb", _list_tz)
    monkeypatch.setattr("vertex_live_dab_agent.orchestrator.orchestrator.set_timezone_via_adb", _set_tz)

    ok = await orch._execute_action(
        state,
        PlannedAction(
            action="SET_SETTING",
            confidence=0.9,
            reason="direct",
            params={"key": "timezone", "value": "America/Los_Angeles"},
        ),
    )

    assert ok is True
    assert orch._dab.calls == 0
    assert any(str(ev.get("type")) == "timezone-adb-fallback-success" for ev in state.ai_transcript)


@pytest.mark.asyncio
async def test_timezone_set_does_not_use_adb_fallback_for_non_android(monkeypatch) -> None:
    orch = _orch()
    orch._dab = _FakeDABUnsupportedTimezoneSet()
    state = RunState(goal="set timezone to UTC")
    state.supported_operations = ["applications/list", "input/key/list"]
    state.supported_settings = [{"key": "timezone", "friendlyName": "Time Zone", "allowedValues": ["UTC"]}]
    state.is_android_device = False
    state.android_adb_device_id = "emulator-5554"

    async def _online(_device_id):
        raise AssertionError("ADB online check should not run for non-Android")

    monkeypatch.setattr("vertex_live_dab_agent.orchestrator.orchestrator.is_adb_device_online", _online)

    ok = await orch._execute_action(
        state,
        PlannedAction(
            action="SET_SETTING",
            confidence=0.9,
            reason="direct",
            params={"key": "timezone", "value": "UTC"},
        ),
    )

    assert ok is False
    assert orch._dab.calls == 0


@pytest.mark.asyncio
async def test_stuck_diagnosis_uses_set_setting_with_android_fallback_when_settings_schema_missing() -> None:
    orch = _orch()
    state = RunState(goal="set timezone to america/loasangles")
    state.supported_operations = ["system/settings/get", "system/settings/set"]
    state.supported_settings = []
    state.is_android_device = True
    state.android_adb_device_id = "emulator-5554"

    decision, _reason, actions = await orch._decide_recovery_from_context(
        state,
        {
            "test_goal": state.goal,
            "current_app": None,
            "current_app_state": None,
            "current_screenshot_summary": "No visual summary from screenshot",
            "available_operations": state.supported_operations,
        },
    )

    assert decision == "USE_DIRECT_DAB_OPERATION"
    assert actions
    assert actions[0]["action"] == "SET_SETTING"
    assert actions[0]["params"]["key"] == "timezone"


@pytest.mark.asyncio
async def test_stuck_diagnosis_fails_fast_when_settings_operation_unsupported() -> None:
    orch = _orch()
    state = RunState(goal="get timezone setting")
    state.supported_operations = ["system/settings/get"]
    state.supported_settings = []
    state.is_android_device = False
    state.unsupported_direct_operations["system/settings/get:timezone"] = {"reason": "unsupported"}

    decision, reason, actions = await orch._decide_recovery_from_context(
        state,
        {
            "test_goal": state.goal,
            "current_app": None,
            "current_app_state": None,
            "current_screenshot_summary": "No visual summary from screenshot",
            "available_operations": state.supported_operations,
        },
    )

    assert decision == "FAIL_WITH_GROUNDED_REASON"
    assert "unsupported" in reason.lower()
    assert actions
    assert actions[0]["action"] == "FAILED"
