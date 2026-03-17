"""Tests for launch-action normalization before PlannedAction validation."""

import pytest

from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState


class _Cfg:
    dab_device_id = "mock-device"


class _Resolved:
    def __init__(self, app_id: str):
        self.app_id = app_id


class _ResolverSuccess:
    async def resolve_target_app(self, goal, planner_output, execution_state):
        name = (planner_output.get("target_app_name") or "").lower()
        if name == "netflix":
            return _Resolved("netflix")
        return None


class _ResolverFail:
    async def resolve_target_app(self, goal, planner_output, execution_state):
        return None


@pytest.mark.asyncio
async def test_normalize_launch_action_resolves_app_name() -> None:
    orch = object.__new__(Orchestrator)
    orch._app_resolver = _ResolverSuccess()
    orch._config = _Cfg()
    state = RunState(goal="open netflix")

    action, params, err = await orch._normalize_launch_action(
        state=state,
        action="LAUNCH_APP",
        params={"app_name": "Netflix"},
        planner_output={"target_app_name": "Netflix", "launch_parameters": {}},
    )

    assert err is None
    assert action == "LAUNCH_APP"
    assert params["app_id"] == "netflix"


@pytest.mark.asyncio
async def test_normalize_launch_action_falls_back_without_crash() -> None:
    orch = object.__new__(Orchestrator)
    orch._app_resolver = _ResolverFail()
    orch._config = _Cfg()
    state = RunState(goal="open unknown app")

    action, params, err = await orch._normalize_launch_action(
        state=state,
        action="LAUNCH_APP",
        params={"app_name": "Unknown"},
        planner_output={"target_app_name": "Unknown"},
    )

    assert action == "NEED_BETTER_VIEW"
    assert params == {}
    assert err is not None
