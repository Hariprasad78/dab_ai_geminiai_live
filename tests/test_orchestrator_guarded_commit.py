"""Tests for guarded commit safety in orchestrator batching."""

from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator


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
