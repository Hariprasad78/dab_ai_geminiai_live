"""Tests for the LiveKit + Vertex AI live agent entrypoint."""
import asyncio

import pytest

import vertex_live_dab_agent.config as cfg_mod


@pytest.fixture(autouse=True)
def reset_cfg(monkeypatch):
    """Ensure a clean config singleton for every test."""
    cfg_mod.reset_config()
    yield
    cfg_mod.reset_config()


# ---------------------------------------------------------------------------
# AgentConfigError / _validate_config
# ---------------------------------------------------------------------------


def test_validate_config_raises_when_project_missing(monkeypatch):
    """_validate_config must raise AgentConfigError when GOOGLE_CLOUD_PROJECT is absent."""
    from vertex_live_dab_agent.livekit_agent.agent import AgentConfigError, _validate_config
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    cfg_mod.reset_config()
    with pytest.raises(AgentConfigError, match="GOOGLE_CLOUD_PROJECT"):
        _validate_config()


def test_validate_config_raises_when_location_missing(monkeypatch):
    """_validate_config raises when GOOGLE_CLOUD_PROJECT is absent.

    GOOGLE_CLOUD_LOCATION always falls back to the default 'asia-south1'
    so removing it from the environment does not produce a missing-var error.
    This test validates that PROJECT is still checked even when LOCATION is absent.
    """
    from vertex_live_dab_agent.livekit_agent.agent import AgentConfigError, _validate_config
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    # Removing LOCATION from env still leaves it as the default, so no error for it
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    cfg_mod.reset_config()
    with pytest.raises(AgentConfigError, match="GOOGLE_CLOUD_PROJECT"):
        _validate_config()


def test_validate_config_raises_when_both_missing(monkeypatch):
    """Error message lists GOOGLE_CLOUD_PROJECT when it is absent.

    GOOGLE_CLOUD_LOCATION always has a default ('asia-south1') so removing it
    from env does not add it to the missing list; only PROJECT is reported.
    """
    from vertex_live_dab_agent.livekit_agent.agent import AgentConfigError, _validate_config
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    cfg_mod.reset_config()
    with pytest.raises(AgentConfigError) as exc_info:
        _validate_config()
    msg = str(exc_info.value)
    assert "GOOGLE_CLOUD_PROJECT" in msg


def test_validate_config_passes_with_required_env_vars(monkeypatch):
    """_validate_config must not raise when all required vars are present."""
    from vertex_live_dab_agent.livekit_agent.agent import _validate_config
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    cfg_mod.reset_config()
    _validate_config()  # Must not raise


# ---------------------------------------------------------------------------
# AgentConfigError is a RuntimeError subclass
# ---------------------------------------------------------------------------


def test_agent_config_error_is_runtime_error():
    """AgentConfigError must be a subclass of RuntimeError."""
    from vertex_live_dab_agent.livekit_agent.agent import AgentConfigError
    err = AgentConfigError("test")
    assert isinstance(err, RuntimeError)
    assert str(err) == "test"


# ---------------------------------------------------------------------------
# OperatorSession
# ---------------------------------------------------------------------------


def test_operator_session_receive_message():
    """receive_operator_message sets pending_goal and records in history."""
    from vertex_live_dab_agent.livekit_agent.agent import OperatorSession
    from vertex_live_dab_agent.session.manager import SessionState

    ss = SessionState("test-session-1")
    op = OperatorSession(session_state=ss, room_name="room-1")

    op.receive_operator_message("Launch Netflix")

    assert op.pending_goal == "Launch Netflix"
    assert len(ss.conversation_history) == 1
    assert ss.conversation_history[0]["role"] == "operator"
    assert ss.conversation_history[0]["content"] == "Launch Netflix"


def test_operator_session_record_agent_response():
    """record_agent_response records agent message in history."""
    from vertex_live_dab_agent.livekit_agent.agent import OperatorSession
    from vertex_live_dab_agent.session.manager import SessionState

    ss = SessionState("test-session-2")
    op = OperatorSession(session_state=ss)

    op.record_agent_response("Launching Netflix now")

    assert len(ss.conversation_history) == 1
    assert ss.conversation_history[0]["role"] == "agent"
    assert ss.conversation_history[0]["content"] == "Launching Netflix now"


def test_operator_session_add_planned_action():
    """add_planned_action appends to planned_actions list."""
    from vertex_live_dab_agent.livekit_agent.agent import OperatorSession
    from vertex_live_dab_agent.planner.schemas import ActionType, PlannedAction
    from vertex_live_dab_agent.session.manager import SessionState

    ss = SessionState("test-session-3")
    op = OperatorSession(session_state=ss)

    action = PlannedAction(
        action=ActionType.PRESS_OK,
        confidence=0.9,
        reason="Selecting item",
    )
    op.add_planned_action(action)

    assert len(op.planned_actions) == 1
    assert op.planned_actions[0].action == "PRESS_OK"


def test_operator_session_multiple_messages():
    """Multiple operator messages update pending_goal to the latest."""
    from vertex_live_dab_agent.livekit_agent.agent import OperatorSession
    from vertex_live_dab_agent.session.manager import SessionState

    ss = SessionState("test-session-4")
    op = OperatorSession(session_state=ss)

    op.receive_operator_message("First goal")
    op.receive_operator_message("Second goal")

    assert op.pending_goal == "Second goal"
    assert len(ss.conversation_history) == 2


def test_operator_session_default_room_name():
    """OperatorSession defaults to empty room_name and no pending_goal."""
    from vertex_live_dab_agent.livekit_agent.agent import OperatorSession
    from vertex_live_dab_agent.session.manager import SessionState

    ss = SessionState("test-session-5")
    op = OperatorSession(session_state=ss)

    assert op.room_name == ""
    assert op.pending_goal is None
    assert op.planned_actions == []


# ---------------------------------------------------------------------------
# _build_livekit_worker
# ---------------------------------------------------------------------------


def test_build_livekit_worker_returns_none_without_sdk(monkeypatch):
    """_build_livekit_worker returns None when livekit-agents is not installed."""
    import sys
    # Ensure livekit is not importable
    monkeypatch.setitem(sys.modules, "livekit", None)
    monkeypatch.setitem(sys.modules, "livekit.agents", None)
    cfg_mod.reset_config()

    from vertex_live_dab_agent.livekit_agent.agent import _build_livekit_worker
    from vertex_live_dab_agent.planner.planner import Planner
    from vertex_live_dab_agent.session.manager import SessionManager

    config = cfg_mod.get_config()
    result = _build_livekit_worker(config, SessionManager(), Planner())
    assert result is None


# ---------------------------------------------------------------------------
# run_agent — text mode (no LiveKit creds, skip config validation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_text_mode_cancels_cleanly(monkeypatch):
    """run_agent in text mode must cancel cleanly with asyncio.CancelledError."""
    import vertex_live_dab_agent.config as cfg_mod
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    cfg_mod.reset_config()

    from vertex_live_dab_agent.livekit_agent.agent import run_agent

    task = asyncio.create_task(run_agent(skip_config_validation=True))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected — loop was cancelled cleanly


@pytest.mark.asyncio
async def test_run_agent_raises_on_missing_creds(monkeypatch):
    """run_agent must raise AgentConfigError when project env var is missing."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    cfg_mod.reset_config()

    from vertex_live_dab_agent.livekit_agent.agent import AgentConfigError, run_agent

    with pytest.raises(AgentConfigError):
        await run_agent()


# ---------------------------------------------------------------------------
# _run_text_mode_loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_text_mode_loop_cleanup(monkeypatch):
    """Text mode loop must call cleanup_expired and cancel cleanly."""
    from vertex_live_dab_agent.livekit_agent.agent import _run_text_mode_loop
    from vertex_live_dab_agent.planner.planner import Planner
    from vertex_live_dab_agent.session.manager import SessionManager

    sm = SessionManager()
    planner = Planner()

    # Use a very short cleanup interval so the loop iterates quickly
    task = asyncio.create_task(
        _run_text_mode_loop(sm, planner, cleanup_interval=0.02)
    )
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected


# ---------------------------------------------------------------------------
# Module-level import sanity
# ---------------------------------------------------------------------------


def test_agent_module_imports_cleanly():
    """The agent module must import without raising (no top-level side effects)."""
    import vertex_live_dab_agent.livekit_agent.agent as agent_mod
    assert hasattr(agent_mod, "run_agent")
    assert hasattr(agent_mod, "AgentConfigError")
    assert hasattr(agent_mod, "OperatorSession")
    assert hasattr(agent_mod, "_validate_config")
    assert hasattr(agent_mod, "_run_text_mode_loop")
    assert hasattr(agent_mod, "_build_livekit_worker")
