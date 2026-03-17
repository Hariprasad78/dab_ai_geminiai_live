"""Capability bootstrap and strategy-selection tests."""

import pytest

from vertex_live_dab_agent.dab.client import DABResponse
from vertex_live_dab_agent.orchestrator.app_resolver import AppResolver
from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState
from vertex_live_dab_agent.planner.schemas import PlannedAction


class _Cfg:
    dab_device_id = "mock-device"
    youtube_app_id = "youtube"


class _Resolved:
    def __init__(self, app_id: str):
        self.app_id = app_id


class _FakeDAB:
    def __init__(self) -> None:
        self.ops_calls = 0
        self.apps_calls = 0
        self.keys_calls = 0
        self.settings_calls = 0

    async def list_operations(self):
        self.ops_calls += 1
        return DABResponse(True, 200, {"operations": ["applications/launch", "applications/list", "input/key/list", "system/settings/list", "system/settings/get", "system/settings/set"]}, "t", "r")

    async def list_apps(self):
        self.apps_calls += 1
        return DABResponse(
            True,
            200,
            {
                "applications": [
                    {"appId": "PrimeVideo", "friendlyName": "Amazon Prime Video"},
                    {"appId": "youtube", "friendlyName": "YouTube"},
                ]
            },
            "t",
            "r",
        )

    async def list_keys(self):
        self.keys_calls += 1
        return DABResponse(True, 200, {"keys": ["KEY_UP", "KEY_DOWN", "KEY_ENTER", "KEY_HOME"]}, "t", "r")

    async def list_settings(self):
        self.settings_calls += 1
        return DABResponse(True, 200, {"settings": [{"key": "timezone", "friendlyName": "Time Zone", "writable": True}]}, "t", "r")


@pytest.mark.asyncio
async def test_bootstrap_capabilities_cached_per_run() -> None:
    dab = _FakeDAB()
    orch = object.__new__(Orchestrator)
    orch._dab = dab
    orch._config = _Cfg()
    orch._app_resolver = AppResolver(dab)

    state = RunState(goal="change time zone")
    await orch._bootstrap_capabilities_if_needed(state)
    await orch._bootstrap_capabilities_if_needed(state)

    assert dab.ops_calls == 1
    assert dab.apps_calls == 1
    assert dab.keys_calls == 1
    assert dab.settings_calls == 1
    assert state.supported_operations
    assert state.app_catalog
    assert state.supported_keys
    assert state.supported_settings


@pytest.mark.asyncio
async def test_strategy_selects_direct_launch_from_applications_list() -> None:
    dab = _FakeDAB()
    orch = object.__new__(Orchestrator)
    orch._dab = dab
    orch._config = _Cfg()
    orch._app_resolver = AppResolver(dab)

    state = RunState(goal="open Prime Video")
    await orch._bootstrap_capabilities_if_needed(state)
    batch = await orch._select_execution_strategy(state)

    assert state.strategy_selected == "DIRECT_APP_LAUNCH"
    assert batch
    assert batch[0]["action"] == "LAUNCH_APP"
    assert batch[0]["params"]["app_id"] == "PrimeVideo"


@pytest.mark.asyncio
async def test_screensaver_back_from_home_uses_direct_key_validation() -> None:
    dab = _FakeDAB()
    orch = object.__new__(Orchestrator)
    orch._dab = dab
    orch._config = _Cfg()
    orch._app_resolver = AppResolver(dab)

    state = RunState(
        goal=(
            "From the Home screen, press the Back button. "
            "Confirm that the screen saver is invoked."
        )
    )
    state.task_preplan = orch._build_task_preplan(state).model_dump()
    await orch._bootstrap_capabilities_if_needed(state)
    batch = await orch._select_execution_strategy(state)

    assert state.strategy_selected == "DIRECT_KEY_VALIDATION"
    assert batch
    assert not any(step.get("action") == "LAUNCH_APP" for step in batch)
    assert any(step.get("action") == "PRESS_BACK" for step in batch)


@pytest.mark.asyncio
async def test_stuck_diagnosis_uses_goal_aware_relaunch() -> None:
    dab = _FakeDAB()
    orch = object.__new__(Orchestrator)
    orch._dab = dab
    orch._config = _Cfg()
    orch._app_resolver = AppResolver(dab)

    state = RunState(goal="open YouTube")
    state.supported_operations = ["applications/launch", "applications/list"]
    state.app_catalog = [{"appId": "youtube", "friendlyName": "YouTube"}]
    state.current_app_state = "BACKGROUND"
    state.latest_ocr_text = "Home screen with app tiles"
    state.last_actions = ["GET_STATE", "NEED_BETTER_VIEW", "GET_STATE"]
    state.retries = 2

    ctx = orch._build_stuck_context(state)
    decision, reason, batch = await orch._decide_recovery_from_context(state, ctx)

    assert decision == "RELAUNCH_TARGET_APP"
    assert "relaunch" in reason.lower()
    assert batch and any(s.get("action") == "LAUNCH_APP" for s in batch)


def test_task_preplan_classifies_direct_key_validation() -> None:
    orch = object.__new__(Orchestrator)
    state = RunState(
        goal=(
            "From the Home screen, press the Back button. "
            "Confirm that the screen saver is invoked."
        )
    )
    pre = orch._build_task_preplan(state)
    assert pre.step_type.value == "DIRECT_KEY_VALIDATION"
    assert pre.required_action == "PRESS_BACK"
    assert pre.needs_app_launch is False
    assert pre.needs_settings_navigation is False


@pytest.mark.asyncio
async def test_youtube_stats_goal_stays_in_youtube_context() -> None:
    dab = _FakeDAB()
    orch = object.__new__(Orchestrator)
    orch._dab = dab
    orch._config = _Cfg()
    orch._app_resolver = AppResolver(dab)

    state = RunState(goal="Play any YouTube video and enable Stats for Nerds using the settings gear icon")
    state.task_preplan = orch._build_task_preplan(state).model_dump()
    await orch._bootstrap_capabilities_if_needed(state)
    batch = await orch._select_execution_strategy(state)

    assert state.strategy_selected == "YOUTUBE_PLAYER_WORKFLOW"
    assert batch and batch[0]["action"] == "LAUNCH_APP"
    assert batch[0]["params"]["app_id"] == "youtube"
    assert all("settings" not in str(step.get("params", {}).get("app_id", "")).lower() for step in batch)


def test_youtube_stats_preplan_has_required_subgoals() -> None:
    orch = object.__new__(Orchestrator)
    state = RunState(goal="Play any YouTube video and enable Stats for Nerds using the settings gear icon")
    pre = orch._build_task_preplan(state)
    assert pre.target_app == "youtube"
    assert "ENABLE_STATS_FOR_NERDS" in pre.required_subgoals
    assert any("do not open Android Settings" in d for d in pre.forbidden_detours)


def test_forbidden_detour_guard_blocks_unrelated_settings_launch() -> None:
    orch = object.__new__(Orchestrator)
    state = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    planned = PlannedAction(
        action="LAUNCH_APP",
        confidence=0.8,
        reason="detour",
        params={"app_id": "settings"},
    )
    blocked = orch._sanitize_planned_action_for_goal(state, planned)
    assert blocked.action == "NEED_VIDEO_PLAYBACK_CONFIRMED"


def test_bounded_back_guard_blocks_back_spam() -> None:
    orch = object.__new__(Orchestrator)
    state = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    state.last_actions = ["PRESS_BACK", "PRESS_BACK"]
    planned = PlannedAction(action="PRESS_BACK", confidence=0.7, reason="loop")
    blocked = orch._sanitize_planned_action_for_goal(state, planned)
    assert blocked.action == "CAPTURE_SCREENSHOT"


@pytest.mark.asyncio
async def test_youtube_play_video_goal_continues_after_launch() -> None:
    dab = _FakeDAB()
    orch = object.__new__(Orchestrator)
    orch._dab = dab
    orch._config = _Cfg()
    orch._app_resolver = AppResolver(dab)

    state = RunState(goal="Play video in YouTube")
    state.current_app_id = "youtube"
    state.current_app_state = "FOREGROUND"
    state.latest_ocr_text = "Home Shorts Subscriptions"
    state.step_count = 2
    state.task_preplan = orch._build_task_preplan(state).model_dump()

    batch = await orch._select_execution_strategy(state)
    assert state.strategy_selected == "YOUTUBE_PLAYER_WORKFLOW"
    assert batch and batch[0]["action"] == "PRESS_OK"
    assert batch[0]["params"]["ok_intent"] in {"SELECT_FOCUSED_CONTROL", "REVEAL_PLAYER_CONTROLS"}


def test_youtube_goal_verified_requires_playback_or_overlay() -> None:
    orch = object.__new__(Orchestrator)
    orch._config = _Cfg()

    play_goal = RunState(goal="Play video in YouTube")
    play_goal.current_app_id = "youtube"
    play_goal.current_app_state = "FOREGROUND"
    play_goal.latest_ocr_text = "Home Shorts Subscriptions"
    assert orch._is_youtube_goal_verified(play_goal) is False

    play_goal.latest_ocr_text = "Pause Settings Up next"
    assert orch._is_youtube_goal_verified(play_goal) is True

    stats_goal = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    stats_goal.current_app_id = "youtube"
    stats_goal.current_app_state = "FOREGROUND"
    stats_goal.latest_ocr_text = "Pause Settings"
    assert orch._is_youtube_goal_verified(stats_goal) is False

    stats_goal.latest_ocr_text = "Pause Settings Stats for Nerds"
    assert orch._is_youtube_goal_verified(stats_goal) is True


def test_youtube_playback_blocks_blind_ok_without_intent() -> None:
    orch = object.__new__(Orchestrator)
    orch._config = _Cfg()
    state = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    state.current_app_id = "youtube"
    state.latest_ocr_text = "Pause Up next"

    planned = PlannedAction(action="PRESS_OK", confidence=0.8, reason="blind commit")
    blocked = orch._sanitize_planned_action_for_goal(state, planned)
    assert blocked.action == "CAPTURE_SCREENSHOT"


def test_youtube_playback_one_ok_reveal_rule_enforced() -> None:
    orch = object.__new__(Orchestrator)
    orch._config = _Cfg()
    state = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    state.current_app_id = "youtube"
    state.latest_ocr_text = "Pause Up next"
    state.repeated_commit_count = 1

    planned = PlannedAction(
        action="PRESS_OK",
        confidence=0.8,
        reason="reveal controls",
        params={"ok_intent": "REVEAL_PLAYER_CONTROLS"},
    )
    blocked = orch._sanitize_planned_action_for_goal(state, planned)
    assert blocked.action == "NEED_PLAYER_CONTROLS_VISIBLE"


def test_youtube_playback_blocks_repeated_commit_without_progress() -> None:
    orch = object.__new__(Orchestrator)
    orch._config = _Cfg()
    state = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    state.current_app_id = "youtube"
    state.latest_ocr_text = "Pause Settings"
    state.focus_target_guess = "settings gear"
    state.repeated_commit_count = 2
    state.no_progress_count = 2

    planned = PlannedAction(
        action="PRESS_OK",
        confidence=0.8,
        reason="confirm menu",
        params={"ok_intent": "CONFIRM_MENU_ITEM"},
    )
    blocked = orch._sanitize_planned_action_for_goal(state, planned)
    assert blocked.action == "CAPTURE_SCREENSHOT"


@pytest.mark.asyncio
async def test_youtube_recovery_waits_when_ocr_missing() -> None:
    dab = _FakeDAB()
    orch = object.__new__(Orchestrator)
    orch._dab = dab
    orch._config = _Cfg()
    orch._app_resolver = AppResolver(dab)

    state = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    state.current_app_id = "youtube"
    state.current_app_state = "FOREGROUND"
    state.latest_ocr_text = "No OCR text from screenshot"

    ctx = orch._build_stuck_context(state)
    decision, reason, batch = await orch._decide_recovery_from_context(state, ctx)

    assert decision == "YOUTUBE_WAIT_FOR_VISUAL_STABILITY"
    assert "ocr" in reason.lower()
    assert batch and batch[0]["action"] == "WAIT"


def test_focus_before_select_uses_directional_correction_when_ocr_available() -> None:
    orch = object.__new__(Orchestrator)
    orch._config = _Cfg()
    state = RunState(goal="Play any YouTube video and enable Stats for Nerds")
    state.current_app_id = "youtube"
    state.latest_ocr_text = "Pause Up next"

    planned = PlannedAction(
        action="PRESS_OK",
        confidence=0.8,
        reason="confirm menu",
        params={"ok_intent": "CONFIRM_MENU_ITEM"},
    )
    blocked = orch._sanitize_planned_action_for_goal(state, planned)
    assert blocked.action == "PRESS_RIGHT"
