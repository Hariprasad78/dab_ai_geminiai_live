"""Tests for the DAB transport interface and AdapterDABClient."""
import asyncio
from typing import Any, Dict, List

import pytest

import vertex_live_dab_agent.config as cfg_mod
from vertex_live_dab_agent.dab.client import (
    AdapterDABClient,
    DABError,
    DABResponse,
    MockDABClient,
    create_dab_client,
)
from vertex_live_dab_agent.dab.topics import (
    TOPIC_APPLICATIONS_GET_STATE,
    TOPIC_APPLICATIONS_LAUNCH,
    TOPIC_INPUT_KEY_PRESS,
    TOPIC_OUTPUT_IMAGE,
    format_topic,
)
from vertex_live_dab_agent.dab.transport import (
    DABTransportBase,
    DABTransportError,
    MQTTTransport,
    TransportRequest,
    TransportResponse,
)


# ---------------------------------------------------------------------------
# Helpers / fake transports
# ---------------------------------------------------------------------------


class SucceedingTransport(DABTransportBase):
    """Always returns status=200 with the echoed payload."""

    def __init__(self) -> None:
        self.calls: List[TransportRequest] = []

    async def send(self, request: TransportRequest) -> TransportResponse:
        self.calls.append(request)
        return TransportResponse(
            topic=request.topic + "/response",
            payload={"status": 200, **request.payload},
            request_id=request.request_id,
            status=200,
        )

    async def close(self) -> None:
        pass


class FailingTransport(DABTransportBase):
    """Always raises DABTransportError."""

    def __init__(self, error_msg: str = "connection refused") -> None:
        self._error_msg = error_msg
        self.call_count = 0

    async def send(self, request: TransportRequest) -> TransportResponse:
        self.call_count += 1
        raise DABTransportError(self._error_msg)

    async def close(self) -> None:
        pass


class HangingTransport(DABTransportBase):
    """Never returns (simulates a non-responding device)."""

    def __init__(self) -> None:
        self.call_count = 0

    async def send(self, request: TransportRequest) -> TransportResponse:
        self.call_count += 1
        await asyncio.sleep(9999)  # will be cancelled by wait_for
        raise RuntimeError("should not reach here")  # pragma: no cover

    async def close(self) -> None:
        pass


class EventuallySucceedingTransport(DABTransportBase):
    """Fails for the first N calls then succeeds."""

    def __init__(self, fail_count: int) -> None:
        self._fail_count = fail_count
        self.call_count = 0

    async def send(self, request: TransportRequest) -> TransportResponse:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise DABTransportError(f"transient error #{self.call_count}")
        return TransportResponse(
            topic=request.topic + "/response",
            payload={"status": 200, **request.payload},
            request_id=request.request_id,
            status=200,
        )

    async def close(self) -> None:
        pass


class ErrorStatusTransport(DABTransportBase):
    """Returns a 4xx status code to test failure normalization."""

    def __init__(self, status: int = 404) -> None:
        self._status = status

    async def send(self, request: TransportRequest) -> TransportResponse:
        return TransportResponse(
            topic=request.topic + "/response",
            payload={"status": self._status, "message": "not found"},
            request_id=request.request_id,
            status=self._status,
        )

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# TransportRequest / TransportResponse data class tests
# ---------------------------------------------------------------------------


def test_transport_request_defaults():
    req = TransportRequest(
        topic="dab/device/input/key-press",
        payload={"keyCode": "KEY_UP"},
        request_id="abc-123",
    )
    assert req.topic == "dab/device/input/key-press"
    assert req.payload == {"keyCode": "KEY_UP"}
    assert req.request_id == "abc-123"
    assert req.timeout == 10.0  # default


def test_transport_request_custom_timeout():
    req = TransportRequest(
        topic="t",
        payload={},
        request_id="r",
        timeout=5.0,
    )
    assert req.timeout == 5.0


def test_transport_response_defaults():
    resp = TransportResponse(
        topic="dab/device/input/key-press/response",
        payload={"status": 200},
        request_id="abc-123",
    )
    assert resp.status == 200
    assert resp.topic.endswith("/response")


# ---------------------------------------------------------------------------
# MQTTTransport tests
# ---------------------------------------------------------------------------


def test_mqtt_transport_constructs_without_raising():
    """MQTTTransport.__init__ must not raise -- server must start cleanly."""
    transport = MQTTTransport(broker="localhost", port=1883)
    assert transport._broker == "localhost"
    assert transport._port == 1883


@pytest.mark.asyncio
async def test_mqtt_transport_send_handles_unavailable_backend_or_connection_error():
    """MQTTTransport.send() should fail cleanly when backend/broker is unavailable."""
    transport = MQTTTransport()
    req = TransportRequest(topic="dab/dev/t", payload={}, request_id="x", timeout=0.1)
    with pytest.raises((NotImplementedError, DABTransportError)):
        await transport.send(req)


@pytest.mark.asyncio
async def test_mqtt_transport_close_is_noop():
    """MQTTTransport.close() should not raise."""
    transport = MQTTTransport()
    await transport.close()  # must not raise


# ---------------------------------------------------------------------------
# AdapterDABClient with SucceedingTransport
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter() -> AdapterDABClient:
    return AdapterDABClient(
        transport=SucceedingTransport(),
        device_id="test-device",
        timeout=1.0,
        max_retries=0,
    )


@pytest.mark.asyncio
async def test_adapter_launch_app(adapter):
    resp = await adapter.launch_app("com.netflix.ninja")
    assert isinstance(resp, DABResponse)
    assert resp.success is True
    assert resp.status == 200
    assert resp.data["appId"] == "com.netflix.ninja"
    assert "test-device" in resp.topic
    assert TOPIC_APPLICATIONS_LAUNCH.split("{device_id}")[1] in resp.topic


@pytest.mark.asyncio
async def test_adapter_launch_app_with_parameters(adapter):
    resp = await adapter.launch_app("com.netflix.ninja", parameters={"content": "abc"})
    assert resp.success is True
    assert resp.data["parameters"] == {"content": "abc"}


@pytest.mark.asyncio
async def test_adapter_get_app_state(adapter):
    resp = await adapter.get_app_state("com.netflix.ninja")
    assert resp.success is True
    assert "test-device" in resp.topic
    assert TOPIC_APPLICATIONS_GET_STATE.split("{device_id}")[1] in resp.topic


@pytest.mark.asyncio
async def test_adapter_key_press(adapter):
    resp = await adapter.key_press("KEY_UP")
    assert resp.success is True
    assert resp.data["keyCode"] == "KEY_UP"
    assert "test-device" in resp.topic
    assert TOPIC_INPUT_KEY_PRESS.split("{device_id}")[1] in resp.topic


@pytest.mark.asyncio
async def test_adapter_capture_screenshot(adapter):
    resp = await adapter.capture_screenshot()
    assert resp.success is True
    assert "test-device" in resp.topic
    assert TOPIC_OUTPUT_IMAGE.split("{device_id}")[1] in resp.topic


@pytest.mark.asyncio
async def test_adapter_close():
    transport = SucceedingTransport()
    client = AdapterDABClient(transport=transport, device_id="d", timeout=1.0, max_retries=0)
    await client.close()  # must not raise


# ---------------------------------------------------------------------------
# Topic resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_resolves_device_id_in_topic():
    transport = SucceedingTransport()
    client = AdapterDABClient(transport=transport, device_id="living-room-tv", timeout=1.0, max_retries=0)
    resp = await client.key_press("KEY_ENTER")
    assert resp.topic == format_topic(TOPIC_INPUT_KEY_PRESS, "living-room-tv")


# ---------------------------------------------------------------------------
# Response normalization: error status codes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_normalizes_4xx_as_failure():
    client = AdapterDABClient(
        transport=ErrorStatusTransport(status=404),
        device_id="d",
        timeout=1.0,
        max_retries=0,
    )
    resp = await client.key_press("KEY_UP")
    assert resp.success is False
    assert resp.status == 404


@pytest.mark.asyncio
async def test_adapter_normalizes_5xx_as_failure():
    client = AdapterDABClient(
        transport=ErrorStatusTransport(status=500),
        device_id="d",
        timeout=1.0,
        max_retries=0,
    )
    resp = await client.launch_app("com.example")
    assert resp.success is False
    assert resp.status == 500


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_raises_dab_error_on_timeout():
    """A hanging transport should exhaust retries and raise DABError."""
    client = AdapterDABClient(
        transport=HangingTransport(),
        device_id="d",
        timeout=0.05,   # very short timeout
        max_retries=0,  # no retries
    )
    with pytest.raises(DABError, match="failed after 1 attempt"):
        await client.key_press("KEY_UP")


@pytest.mark.asyncio
async def test_adapter_retries_on_timeout():
    """Adapter should make max_retries + 1 total attempts before raising."""
    transport = HangingTransport()
    client = AdapterDABClient(
        transport=transport,
        device_id="d",
        timeout=0.05,
        max_retries=2,
    )
    with pytest.raises(DABError):
        await client.key_press("KEY_UP")
    assert transport.call_count == 3  # 1 initial + 2 retries


# ---------------------------------------------------------------------------
# Retry on transport error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_retries_on_transport_error():
    """Adapter should retry DABTransportError and eventually raise DABError."""
    transport = FailingTransport()
    client = AdapterDABClient(
        transport=transport,
        device_id="d",
        timeout=1.0,
        max_retries=2,
    )
    with pytest.raises(DABError, match="failed after 3 attempt"):
        await client.launch_app("com.example")
    assert transport.call_count == 3


@pytest.mark.asyncio
async def test_adapter_succeeds_after_transient_failures():
    """Adapter should succeed once transient failures clear."""
    transport = EventuallySucceedingTransport(fail_count=2)
    client = AdapterDABClient(
        transport=transport,
        device_id="d",
        timeout=1.0,
        max_retries=3,
    )
    resp = await client.launch_app("com.example")
    assert resp.success is True
    assert transport.call_count == 3  # 2 failures + 1 success


# ---------------------------------------------------------------------------
# NotImplementedError propagates without retrying
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_does_not_retry_not_implemented(monkeypatch):
    """AdapterDABClient must NOT retry NotImplementedError from transport."""
    transport = MQTTTransport()
    monkeypatch.setattr(
        transport,
        "_import_aiomqtt",
        lambda: (_ for _ in ()).throw(NotImplementedError("aiomqtt missing")),
    )
    client = AdapterDABClient(
        transport=transport,
        device_id="d",
        timeout=1.0,
        max_retries=5,  # would be 6 attempts if it retried
    )
    with pytest.raises(NotImplementedError):
        await client.key_press("KEY_UP")


# ---------------------------------------------------------------------------
# create_dab_client in real mode returns AdapterDABClient
# ---------------------------------------------------------------------------


def test_create_dab_client_real_mode_returns_adapter(monkeypatch):
    """create_dab_client with DAB_MOCK_MODE=false must return AdapterDABClient."""
    monkeypatch.setenv("DAB_MOCK_MODE", "false")
    cfg_mod.reset_config()
    try:
        client = create_dab_client()
        assert isinstance(client, AdapterDABClient)
    finally:
        cfg_mod.reset_config()


def test_create_dab_client_mock_mode_returns_mock(monkeypatch):
    """create_dab_client with DAB_MOCK_MODE=true must return MockDABClient."""
    monkeypatch.setenv("DAB_MOCK_MODE", "true")
    cfg_mod.reset_config()
    try:
        client = create_dab_client()
        assert isinstance(client, MockDABClient)
    finally:
        cfg_mod.reset_config()


# ---------------------------------------------------------------------------
# DABTransportError is importable and usable
# ---------------------------------------------------------------------------


def test_dab_transport_error_is_exception():
    exc = DABTransportError("network down")
    assert isinstance(exc, Exception)
    assert str(exc) == "network down"
