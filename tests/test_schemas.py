"""Tests for planner action schemas."""
import pytest
from pydantic import ValidationError

from vertex_live_dab_agent.planner.schemas import ActionType, PlannedAction


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
        params={"app_id": "com.netflix.ninja"},
    )
    assert action.action == "LAUNCH_APP"
    assert action.params == {"app_id": "com.netflix.ninja"}


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
        params={"app_id": "com.example"},
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
