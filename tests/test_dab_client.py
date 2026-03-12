"""Tests for the MockDABClient."""
import pytest

from vertex_live_dab_agent.dab.client import DABResponse, MockDABClient, create_dab_client
from vertex_live_dab_agent.dab.topics import (
    KEY_MAP,
    TOPIC_APPLICATIONS_GET_STATE,
    TOPIC_APPLICATIONS_LAUNCH,
    TOPIC_INPUT_KEY_PRESS,
    TOPIC_OUTPUT_IMAGE,
)


@pytest.fixture
def client():
    return MockDABClient()


@pytest.mark.asyncio
async def test_launch_app(client):
    resp = await client.launch_app("com.netflix.ninja")
    assert resp.success is True
    assert resp.status == 200
    assert resp.data["appId"] == "com.netflix.ninja"
    assert resp.data["state"] == "FOREGROUND"
    assert resp.topic == TOPIC_APPLICATIONS_LAUNCH
    assert resp.request_id


@pytest.mark.asyncio
async def test_get_app_state(client):
    resp = await client.get_app_state("com.netflix.ninja")
    assert resp.success is True
    assert resp.data["appId"] == "com.netflix.ninja"
    assert resp.topic == TOPIC_APPLICATIONS_GET_STATE


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
