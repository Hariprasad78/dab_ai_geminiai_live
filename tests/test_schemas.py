"""Tests for planner action schemas."""
import pytest
from pydantic import ValidationError

from vertex_live_dab_agent.planner.schemas import ActionType, NavigationPlan, PlannedAction


def test_planned_action_valid():
    action = PlannedAction(
        action=ActionType.PRESS_OK,
        confidence=0.9,
        reason="Selecting item",
    )
    assert action.action == "PRESS_OK"
    assert action.confidence == 0.9
    assert action.params is None


def test_planned_action_with_params():
    action = PlannedAction(
        action=ActionType.LAUNCH_APP,
        confidence=0.8,
        reason="Launching Netflix",
        params={"app_id": "youtube"},
    )
    assert action.action == "LAUNCH_APP"
    assert action.params == {"app_id": "youtube"}


def test_planned_action_confidence_bounds():
    with pytest.raises(ValidationError):
        PlannedAction(action=ActionType.DONE, confidence=1.5, reason="too high")

    with pytest.raises(ValidationError):
        PlannedAction(action=ActionType.DONE, confidence=-0.1, reason="negative")


def test_all_action_types_valid():
    # Actions that require no params
    no_param_actions = {
        ActionType.PRESS_UP, ActionType.PRESS_DOWN, ActionType.PRESS_LEFT,
        ActionType.PRESS_RIGHT, ActionType.PRESS_OK, ActionType.PRESS_BACK,
        ActionType.PRESS_HOME, ActionType.GET_STATE, ActionType.CAPTURE_SCREENSHOT,
        ActionType.DONE, ActionType.FAILED, ActionType.NEED_BETTER_VIEW,
    }
    for action_type in no_param_actions:
        pa = PlannedAction(action=action_type, confidence=0.5, reason="test")
        assert pa.action == action_type.value

    # LAUNCH_APP requires app_id
    pa = PlannedAction(
        action=ActionType.LAUNCH_APP, confidence=0.5, reason="test",
        params={"app_id": "youtube"},
    )
    assert pa.action == ActionType.LAUNCH_APP.value

    # WAIT requires seconds
    pa = PlannedAction(
        action=ActionType.WAIT, confidence=0.5, reason="test",
        params={"seconds": 2},
    )
    assert pa.action == ActionType.WAIT.value


def test_planned_action_invalid_action():
    with pytest.raises(ValidationError):
        PlannedAction(action="INVALID_ACTION", confidence=0.5, reason="bad")


def test_planned_action_launch_app_allows_non_youtube_app_id():
    action = PlannedAction(
        action=ActionType.LAUNCH_APP,
        confidence=0.8,
        reason="Launch app",
        params={"app_id": "com.netflix.ninja"},
    )
    assert action.params["app_id"] == "com.netflix.ninja"


def test_planned_action_done():
    action = PlannedAction(action=ActionType.DONE, confidence=1.0, reason="Goal achieved")
    assert action.action == "DONE"


def test_planned_action_failed():
    action = PlannedAction(action=ActionType.FAILED, confidence=0.9, reason="Too many retries")
    assert action.action == "FAILED"


def test_planned_action_wait_params():
    action = PlannedAction(
        action=ActionType.WAIT,
        confidence=0.7,
        reason="Waiting for load",
        params={"seconds": 3},
    )
    assert action.params["seconds"] == 3


def test_planned_action_with_subplan():
    action = PlannedAction(
        action=ActionType.PRESS_HOME,
        confidence=0.8,
        reason="Go home first",
        subplan=[
            {"action": ActionType.PRESS_RIGHT.value},
            {"action": ActionType.PRESS_OK.value},
        ],
    )
    assert action.subplan is not None
    assert len(action.subplan) == 2


def test_planned_action_rejects_terminal_subplan_action():
    with pytest.raises(ValidationError):
        PlannedAction(
            action=ActionType.PRESS_HOME,
            confidence=0.8,
            reason="Invalid subplan",
            subplan=[{"action": ActionType.DONE.value}],
        )


def test_navigation_plan_normalizes_legacy_string_fallback():
    plan = NavigationPlan.model_validate(
        {
            "phase": "test",
            "intent": "normalize",
            "confidence": 0.8,
            "starting_assumption": "legacy fallback",
            "action_batch": [{"action": "GET_STATE", "params": {}}],
            "fallback_if_failed": "PRESS_HOME",
            "done": False,
        }
    )
    assert plan.fallback_if_failed is not None
    assert plan.fallback_if_failed.action == ActionType.PRESS_HOME.value
    assert plan.fallback_if_failed.params == {}


def test_navigation_plan_accepts_structured_fallback():
    plan = NavigationPlan.model_validate(
        {
            "phase": "test",
            "intent": "structured",
            "confidence": 0.8,
            "starting_assumption": "new schema",
            "action_batch": [{"action": "GET_STATE", "params": {}}],
            "fallback_if_failed": {"action": "PRESS_BACK", "params": {}},
            "done": False,
        }
    )
    assert plan.fallback_if_failed is not None
    assert plan.fallback_if_failed.action == ActionType.PRESS_BACK.value


def test_navigation_plan_supports_strategy_and_logical_target_fields():
    plan = NavigationPlan.model_validate(
        {
            "phase": "strategy",
            "intent": "launch-first",
            "execution_mode": "DIRECT_APP_LAUNCH",
            "target_app_name": "Settings",
            "target_app_domain": "system_settings",
            "target_app_hint": "settings",
            "launch_parameters": {},
            "confidence": 0.9,
            "starting_assumption": "task requires settings",
            "action_batch": [],
            "done": False,
        }
    )
    assert plan.execution_mode == "DIRECT_APP_LAUNCH"
    assert plan.target_app_name == "Settings"
