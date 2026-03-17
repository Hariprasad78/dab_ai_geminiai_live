"""Tests for runtime app resolver."""

import pytest

from vertex_live_dab_agent.dab.client import DABClientBase, DABResponse, MockDABClient
from vertex_live_dab_agent.orchestrator.app_resolver import AppInfo, AppResolver


class _FakeListAppsClient(DABClientBase):
    async def launch_app(self, app_id: str, parameters=None):
        return DABResponse(True, 200, {"appId": app_id}, "t", "r")

    async def get_app_state(self, app_id: str):
        return DABResponse(True, 200, {"appId": app_id, "state": "FOREGROUND"}, "t", "r")

    async def key_press(self, key_code: str):
        return DABResponse(True, 200, {}, "t", "r")

    async def long_key_press(self, key_code: str, duration_ms: int = 1500):
        return DABResponse(True, 200, {}, "t", "r")

    async def list_keys(self):
        return DABResponse(True, 200, {}, "t", "r")

    async def list_operations(self):
        return DABResponse(True, 200, {}, "t", "r")

    async def list_apps(self):
        return DABResponse(
            True,
            200,
            {
                "applications": [
                    {"appId": "youtube", "name": "YouTube"},
                    {"appId": "settings", "name": "Settings"},
                ]
            },
            "t",
            "r",
        )

    async def exit_app(self, app_id: str):
        return DABResponse(True, 200, {"appId": app_id}, "t", "r")

    async def capture_screenshot(self):
        return DABResponse(True, 200, {"image": ""}, "t", "r")

    async def close(self) -> None:
        return None


class _FlakyListAppsClient(_FakeListAppsClient):
    def __init__(self) -> None:
        self.calls = 0

    async def list_apps(self):
        self.calls += 1
        if self.calls == 1:
            return DABResponse(True, 200, {"applications": []}, "t", "r")
        return await super().list_apps()


@pytest.mark.asyncio
async def test_resolver_loads_catalog_from_dab() -> None:
    resolver = AppResolver(MockDABClient())
    catalog = await resolver.load_app_catalog("dev")
    assert catalog
    assert any(a.app_id == "youtube" for a in catalog)


@pytest.mark.asyncio
async def test_resolver_matches_settings_without_package_hardcode() -> None:
    resolver = AppResolver(_FakeListAppsClient())
    target = await resolver.resolve_target_app(
        goal="change time zone",
        planner_output={
            "target_app_name": "Settings",
            "target_app_domain": "system_settings",
            "target_app_hint": "settings",
        },
        execution_state={"session_id": "s1", "device_id": "d1"},
    )
    assert target is not None
    assert target.app_id == "settings"


def test_build_launch_action_merges_parameters() -> None:
    launch = AppResolver.build_launch_action(
        resolved_target=type("R", (), {"app_id": "youtube"})(),
        launch_parameters={"content": "lofi"},
    )
    assert launch["action"] == "LAUNCH_APP"
    assert launch["params"]["app_id"] == "youtube"
    assert launch["params"]["content"] == "lofi"


@pytest.mark.asyncio
async def test_resolver_refreshes_applications_list_on_miss() -> None:
    client = _FlakyListAppsClient()
    resolver = AppResolver(client)
    target = await resolver.resolve_target_app(
        goal="open youtube",
        planner_output={
            "target_app_name": "YouTube",
            "target_app_domain": "media",
            "target_app_hint": "youtube",
        },
        execution_state={"session_id": "s2", "device_id": "d2"},
    )
    assert target is not None
    assert target.app_id == "youtube"
    assert client.calls >= 2
