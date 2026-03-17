"""Tests for RunState model."""
import pytest

from vertex_live_dab_agent.orchestrator.run_state import ActionRecord, RunState, RunStatus


def test_run_state_defaults():
    state = RunState(goal="Test goal")
    assert state.status == RunStatus.PENDING
    assert state.step_count == 0
    assert state.retries == 0
    assert state.last_actions == []
    assert state.pending_subplan == []
    assert state.action_history == []
    assert state.started_at is None
    assert state.finished_at is None
    assert state.run_id is not None


def test_run_state_start():
    state = RunState(goal="Test goal")
    state.start()
    assert state.status == RunStatus.RUNNING
    assert state.started_at is not None


def test_run_state_finish_done():
    state = RunState(goal="Test goal")
    state.start()
    state.finish(RunStatus.DONE)
    assert state.status == RunStatus.DONE
    assert state.finished_at is not None
    assert state.error is None


def test_run_state_finish_with_error():
    state = RunState(goal="Test goal")
    state.start()
    state.finish(RunStatus.ERROR, error="Something went wrong")
    assert state.status == RunStatus.ERROR
    assert state.error == "Something went wrong"


def test_run_state_record_action():
    state = RunState(goal="Test goal")
    state.start()
    state.record_action(
        action="PRESS_OK",
        params=None,
        confidence=0.9,
        reason="Selecting item",
        result="PASS",
    )
    assert state.step_count == 1
    assert len(state.action_history) == 1
    assert state.last_actions == ["PRESS_OK"]
    record = state.action_history[0]
    assert record.action == "PRESS_OK"
    assert record.confidence == 0.9
    assert record.result == "PASS"


def test_run_state_last_actions_capped_at_10():
    state = RunState(goal="Test goal")
    state.start()
    for i in range(15):
        state.record_action(
            action="PRESS_OK",
            params=None,
            confidence=0.9,
            reason=f"Step {i}",
            result="PASS",
        )
    assert len(state.last_actions) == 10
    assert state.step_count == 15


def test_run_state_record_action_with_params():
    state = RunState(goal="Test goal")
    state.start()
    state.record_action(
        action="LAUNCH_APP",
        params={"app_id": "com.netflix.ninja"},
        confidence=0.95,
        reason="Launching app",
        result="PASS",
    )
    record = state.action_history[0]
    assert record.params == {"app_id": "com.netflix.ninja"}


def test_run_state_unique_run_ids():
    state1 = RunState(goal="Goal 1")
    state2 = RunState(goal="Goal 2")
    assert state1.run_id != state2.run_id


def test_all_run_statuses():
    for status in RunStatus:
        state = RunState(goal="Test")
        state.finish(status)
        assert state.status == status
