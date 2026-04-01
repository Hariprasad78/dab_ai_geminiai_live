"""Tests for guarded commit safety in orchestrator batching."""

from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState
from vertex_live_dab_agent.planner.schemas import PlannedAction


def test_guarded_commit_index_detects_directional_then_ok() -> None:
    orch = object.__new__(Orchestrator)
    idx = orch._guarded_commit_index(
        [
            {"action": "PRESS_RIGHT", "params": {}},
            {"action": "PRESS_RIGHT", "params": {}},
            {"action": "PRESS_OK", "params": {}},
        ]
    )
    assert idx == 2


def test_guarded_commit_index_none_when_no_directional_prefix() -> None:
    orch = object.__new__(Orchestrator)
    idx = orch._guarded_commit_index(
        [
            {"action": "WAIT", "params": {"seconds": 1}},
            {"action": "PRESS_OK", "params": {}},
        ]
    )
    assert idx is None


def test_guarded_commit_action_supports_enter_and_keywords() -> None:
    assert Orchestrator._is_guarded_commit_action("PRESS_OK") is True
    assert Orchestrator._is_guarded_commit_action("PRESS_ENTER") is True
    assert Orchestrator._is_guarded_commit_action("APPLY_SETTING") is True
    assert Orchestrator._is_guarded_commit_action("PRESS_RIGHT") is False


def test_enter_blocked_when_confidence_is_low() -> None:
    orch = object.__new__(Orchestrator)
    state = RunState(goal="Open settings")
    state.latest_visual_summary = "Settings list visible"

    planned = PlannedAction(
        action="PRESS_OK",
        confidence=0.6,
        reason="select candidate",
    )
    gated = orch._sanitize_planned_action_for_goal(state, planned)

    assert gated.action == "FAILED"
    assert "UI navigation is disabled" in str(gated.reason)
    assert gated.action != "PRESS_OK"


def test_repeated_failed_move_causes_strategy_adjustment() -> None:
    orch = object.__new__(Orchestrator)
    state = RunState(goal="Navigate to timezone")
    state.strategy_selected = "UI_NAVIGATION_ONLY"
    state.navigation_memory = [
        {"action": "PRESS_DOWN", "focus_changed": False},
        {"action": "PRESS_DOWN", "focus_changed": False},
        {"action": "PRESS_RIGHT", "focus_changed": True},
    ]

    planned = PlannedAction(
        action="PRESS_DOWN",
        confidence=0.8,
        reason="continue vertical scan",
    )
    adjusted = orch._sanitize_planned_action_for_goal(state, planned)

    assert adjusted.action == "FAILED"
    assert "UI navigation is disabled" in str(adjusted.reason)


def test_short_subplan_directional_batch_requires_recheck() -> None:
    orch = object.__new__(Orchestrator)
    batch = [
        {"action": "PRESS_DOWN", "params": {}},
        {"action": "PRESS_DOWN", "params": {}},
        {"action": "PRESS_RIGHT", "params": {}},
    ]
    assert orch._requires_batch_recheck(batch) is True


def test_settings_goal_blocks_non_settings_actions() -> None:
    orch = object.__new__(Orchestrator)
    state = RunState(goal="set timezone to America/Los_Angeles")

    planned = PlannedAction(
        action="GET_STATE",
        confidence=0.9,
        reason="verify app state",
        params={"app_id": "youtube"},
    )
    gated = orch._sanitize_planned_action_for_goal(state, planned)

    assert gated.action == "FAILED"
    assert "only direct settings operations" in str(gated.reason).lower()
