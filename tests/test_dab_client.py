"""Tests for DAB clients."""
import pytest

import vertex_live_dab_agent.config as cfg_mod
from vertex_live_dab_agent.dab.client import AdapterDABClient, DABError, DABResponse, MockDABClient, create_dab_client
from vertex_live_dab_agent.dab.topics import (
    TOPIC_APPLICATIONS_EXIT,
    KEY_MAP,
    TOPIC_APPLICATIONS_GET_STATE,
    TOPIC_APPLICATIONS_LIST,
    TOPIC_INPUT_KEY_LIST,
    TOPIC_APPLICATIONS_LAUNCH,
    TOPIC_INPUT_LONG_KEY_PRESS,
    TOPIC_INPUT_KEY_PRESS,
    TOPIC_OPERATIONS_LIST,
    TOPIC_OUTPUT_IMAGE,
)
from vertex_live_dab_agent.dab.transport import DABTransportBase, DABTransportError, TransportResponse


@pytest.fixture
def client():
    return MockDABClient()


@pytest.mark.asyncio
async def test_launch_app(client):
    resp = await client.launch_app("youtube", parameters={"content": "lofi"})
    assert resp.success is True
    assert resp.status == 200
    assert resp.data["appId"] == "youtube"
    assert resp.data["content"] == "lofi"
    assert resp.data["state"] == "FOREGROUND"
    assert resp.topic == TOPIC_APPLICATIONS_LAUNCH
    assert resp.request_id


@pytest.mark.asyncio
async def test_get_app_state(client):
    resp = await client.get_app_state("com.netflix.ninja")
    assert resp.success is True
    assert resp.data["appId"] == "netflix"
    assert resp.topic == TOPIC_APPLICATIONS_GET_STATE


@pytest.mark.asyncio
async def test_get_app_state_normalizes_settings_package(client):
    resp = await client.get_app_state("com.android.settings")
    assert resp.success is True
    assert resp.data["appId"] == "settings"


@pytest.mark.asyncio
async def test_launch_app_normalizes_tv_settings_package(client):
    resp = await client.launch_app("com.android.tv.settings")
    assert resp.success is True
    assert resp.data["appId"] == "settings"


@pytest.mark.asyncio
async def test_key_press(client):
    resp = await client.key_press("KEY_UP")
    assert resp.success is True
    assert resp.data["keyCode"] == "KEY_UP"
    assert resp.topic == TOPIC_INPUT_KEY_PRESS


@pytest.mark.asyncio
async def test_capture_screenshot(client):
    resp = await client.capture_screenshot()
    assert resp.success is True
    assert "image" in resp.data
    assert resp.data["format"] == "png"
    assert resp.topic == TOPIC_OUTPUT_IMAGE
    # Verify it's valid base64
    import base64
    decoded = base64.b64decode(resp.data["image"])
    assert len(decoded) > 0


@pytest.mark.asyncio
async def test_list_operations(client):
    resp = await client.list_operations()
    assert resp.success is True
    assert "operations" in resp.data
    assert resp.topic == TOPIC_OPERATIONS_LIST


@pytest.mark.asyncio
async def test_list_apps(client):
    resp = await client.list_apps()
    assert resp.success is True
    assert "applications" in resp.data
    assert resp.topic == TOPIC_APPLICATIONS_LIST


@pytest.mark.asyncio
async def test_exit_app(client):
    resp = await client.exit_app("youtube")
    assert resp.success is True
    assert resp.data["appId"] == "youtube"
    assert resp.topic == TOPIC_APPLICATIONS_EXIT


@pytest.mark.asyncio
async def test_list_keys(client):
    resp = await client.list_keys()
    assert resp.success is True
    assert "keys" in resp.data
    assert resp.topic == TOPIC_INPUT_KEY_LIST


@pytest.mark.asyncio
async def test_long_key_press(client):
    resp = await client.long_key_press("KEY_ENTER", duration_ms=1200)
    assert resp.success is True
    assert resp.data["keyCode"] == "KEY_ENTER"
    assert resp.data["durationMs"] == 1200
    assert resp.topic == TOPIC_INPUT_LONG_KEY_PRESS


@pytest.mark.asyncio
async def test_close(client):
    # Should not raise
    await client.close()


@pytest.mark.asyncio
async def test_all_key_codes(client):
    for action, key_code in KEY_MAP.items():
        resp = await client.key_press(key_code)
        assert resp.success is True
        assert resp.data["keyCode"] == key_code


def test_dab_response_repr():
    resp = DABResponse(
        success=True,
        status=200,
        data={},
        topic="test/topic",
        request_id="abc123",
    )
    assert "True" in repr(resp)
    assert "200" in repr(resp)


def test_create_dab_client_mock_mode():
    """create_dab_client should return MockDABClient in mock mode."""
    import os
    os.environ["DAB_MOCK_MODE"] = "true"
    # Reset singleton so the new env var is picked up
    from vertex_live_dab_agent.config import reset_config
    reset_config()
    client = create_dab_client()
    assert isinstance(client, MockDABClient)
    reset_config()


class _FakeTransport(DABTransportBase):
    async def send(self, request):
        return TransportResponse(
            topic=request.topic,
            payload={"error": "Error: No shell command implementation"},
            request_id=request.request_id,
            status=500,
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_adapter_list_settings_degrades_when_cec_shell_unsupported():
    client = AdapterDABClient(transport=_FakeTransport(), device_id="adb:10.0.0.1:5555", timeout=0.1, max_retries=0)
    resp = await client.list_settings()
    assert resp.success is True
    assert resp.status == 200
    assert resp.data.get("degraded") is True
    settings = resp.data.get("settings") or []
    keys = {str(item.get("key")) for item in settings if isinstance(item, dict)}
    assert "timezone" in keys
    assert "language" in keys
    assert "cec_enabled" in keys


class _AlwaysFailTransport(DABTransportBase):
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, request):
        self.calls += 1
        raise DABTransportError("no response")

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_key_press_uses_low_latency_retry_profile(monkeypatch):
    monkeypatch.setenv("DAB_KEYPRESS_MAX_RETRIES", "0")
    monkeypatch.setenv("DAB_KEYPRESS_TIMEOUT", "0.2")
    cfg_mod.reset_config()

    transport = _AlwaysFailTransport()
    client = AdapterDABClient(transport=transport, device_id="dev-1", timeout=2.0, max_retries=3)
    with pytest.raises(DABError):
        await client.key_press("KEY_HOME")

    # Key presses should not use the general 4-attempt retry storm.
    assert transport.calls == 1
    cfg_mod.reset_config()
