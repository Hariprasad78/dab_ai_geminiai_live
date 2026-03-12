"""DAB client abstraction: high-level command API, normalization, and retry.

Architecture
------------
The DAB layer has two sub-layers:

* **Transport** (``dab/transport.py``) -- raw I/O only. Sends a resolved topic +
  payload, awaits the device response, returns raw bytes/dict. No protocol
  knowledge. See :class:`~vertex_live_dab_agent.dab.transport.DABTransportBase`.

* **Client** (this module) -- DAB-protocol commands, response normalization,
  timeout, and retry. Consumers (planner, orchestrator, API) see only the
  :class:`DABClientBase` interface; no transport details leak upward.

Implementations
---------------
* :class:`MockDABClient` -- zero-dependency in-process mock. Used by default
  (``DAB_MOCK_MODE=true``). Ideal for development and CI.

* :class:`AdapterDABClient` -- production client. Wraps any
  :class:`~vertex_live_dab_agent.dab.transport.DABTransportBase`, adds
  configurable timeout + exponential-backoff retries, and normalises every
  response into a :class:`DABResponse`. Pair with
  :class:`~vertex_live_dab_agent.dab.transport.MQTTTransport` for real devices.

Factory
-------
:func:`create_dab_client` reads ``DAB_MOCK_MODE`` from the environment and
returns the appropriate client. No other module needs to import concrete
client classes.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.topics import (
    KEY_MAP,
    TOPIC_APPLICATIONS_GET_STATE,
    TOPIC_APPLICATIONS_LAUNCH,
    TOPIC_INPUT_KEY_PRESS,
    TOPIC_OUTPUT_IMAGE,
    format_topic,
)

if TYPE_CHECKING:
    from vertex_live_dab_agent.dab.transport import DABTransportBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DABError(Exception):
    """Raised by the DAB client layer when a request cannot be completed."""


# ---------------------------------------------------------------------------
# Normalized response
# ---------------------------------------------------------------------------


class DABResponse:
    """Normalized DAB response -- the single internal currency for all results.

    All client implementations (mock, adapter) return this type so that callers
    never need to know how the data was fetched.

    Attributes:
        success:    ``True`` when the device acknowledged the command (HTTP-style
                    status < 400).
        status:     Integer status code (200 = OK, 4xx/5xx = errors).
        data:       Command-specific payload dict.
        topic:      The DAB topic this response corresponds to.
        request_id: Unique ID for correlation and logging.
    """

    def __init__(
        self,
        success: bool,
        status: int,
        data: Dict[str, Any],
        topic: str,
        request_id: str,
    ) -> None:
        self.success = success
        self.status = status
        self.data = data
        self.topic = topic
        self.request_id = request_id

    def __repr__(self) -> str:
        return (
            f"DABResponse(success={self.success}, status={self.status}, "
            f"topic={self.topic!r}, request_id={self.request_id!r})"
        )


# ---------------------------------------------------------------------------
# Abstract client interface
# ---------------------------------------------------------------------------


class DABClientBase(ABC):
    """Abstract base class for DAB clients.

    Consumers of the DAB layer (orchestrator, API) depend *only* on this
    interface -- no transport-specific imports or knowledge required.
    """

    @abstractmethod
    async def launch_app(
        self, app_id: str, parameters: Optional[Dict[str, Any]] = None
    ) -> DABResponse:
        """Launch an application on the device."""
        ...

    @abstractmethod
    async def get_app_state(self, app_id: str) -> DABResponse:
        """Get the current state of an application."""
        ...

    @abstractmethod
    async def key_press(self, key_code: str) -> DABResponse:
        """Send a key press event to the device."""
        ...

    @abstractmethod
    async def capture_screenshot(self) -> DABResponse:
        """Capture a screenshot. Returns base64 PNG in ``data["image"]``."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources and close the client."""
        ...


# ---------------------------------------------------------------------------
# Mock client (default -- no external dependencies)
# ---------------------------------------------------------------------------


class MockDABClient(DABClientBase):
    """Mock DAB client for local development and testing.

    All operations succeed immediately with 50 ms simulated latency.
    Screenshots return a 1x1 white PNG placeholder encoded as base64.
    """

    def __init__(self) -> None:
        self._config = get_config()

    async def launch_app(
        self, app_id: str, parameters: Optional[Dict[str, Any]] = None
    ) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: launch_app app_id=%s req_id=%s", app_id, req_id)
        return DABResponse(
            success=True,
            status=200,
            data={"appId": app_id, "state": "FOREGROUND"},
            topic=TOPIC_APPLICATIONS_LAUNCH,
            request_id=req_id,
        )

    async def get_app_state(self, app_id: str) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: get_app_state app_id=%s req_id=%s", app_id, req_id)
        return DABResponse(
            success=True,
            status=200,
            data={"appId": app_id, "state": "FOREGROUND"},
            topic=TOPIC_APPLICATIONS_GET_STATE,
            request_id=req_id,
        )

    async def key_press(self, key_code: str) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: key_press key_code=%s req_id=%s", key_code, req_id)
        return DABResponse(
            success=True,
            status=200,
            data={"keyCode": key_code},
            topic=TOPIC_INPUT_KEY_PRESS,
            request_id=req_id,
        )

    async def capture_screenshot(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: capture_screenshot req_id=%s", req_id)
        # Minimal 1x1 white PNG (valid base64)
        white_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )
        return DABResponse(
            success=True,
            status=200,
            data={"image": white_png, "format": "png"},
            topic=TOPIC_OUTPUT_IMAGE,
            request_id=req_id,
        )

    async def close(self) -> None:
        logger.info("MockDABClient closed")

    async def _simulate_latency(self) -> None:
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Adapter client (production -- wraps any DABTransportBase)
# ---------------------------------------------------------------------------


class AdapterDABClient(DABClientBase):
    """Production DAB client that delegates I/O to a pluggable transport.

    Responsibilities
    ~~~~~~~~~~~~~~~~
    * Constructs DAB-protocol payloads and resolves topic templates.
    * Wraps every request in a configurable timeout (``asyncio.wait_for``).
    * Retries on timeout or transport error with exponential back-off.
    * Normalises transport responses into :class:`DABResponse`.
    * Emits structured log lines for every send / receive / retry / failure.

    The transport itself (:class:`~vertex_live_dab_agent.dab.transport.DABTransportBase`)
    has no knowledge of DAB topics, payloads, or retry policy.

    Args:
        transport:   A :class:`~vertex_live_dab_agent.dab.transport.DABTransportBase`
                     instance.
        device_id:   DAB device identifier used in topic substitution.
                     Defaults to ``config.dab_device_id``.
        timeout:     Per-request timeout in seconds.
                     Defaults to ``config.dab_request_timeout``.
        max_retries: Maximum number of additional attempts after the first
                     failure.  0 = no retries.
                     Defaults to ``config.dab_max_retries``.
    """

    def __init__(
        self,
        transport: DABTransportBase,
        device_id: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        config = get_config()
        self._transport = transport
        self._device_id = device_id or config.dab_device_id
        self._timeout = timeout if timeout is not None else config.dab_request_timeout
        self._max_retries = (
            max_retries if max_retries is not None else config.dab_max_retries
        )

    # ------------------------------------------------------------------
    # Public DAB command methods
    # ------------------------------------------------------------------

    async def launch_app(
        self, app_id: str, parameters: Optional[Dict[str, Any]] = None
    ) -> DABResponse:
        """Launch an application (``applications/launch``)."""
        payload: Dict[str, Any] = {"appId": app_id}
        if parameters:
            payload["parameters"] = parameters
        return await self._send_with_retry(TOPIC_APPLICATIONS_LAUNCH, payload)

    async def get_app_state(self, app_id: str) -> DABResponse:
        """Query current application state (``applications/get-state``)."""
        return await self._send_with_retry(
            TOPIC_APPLICATIONS_GET_STATE, {"appId": app_id}
        )

    async def key_press(self, key_code: str) -> DABResponse:
        """Send a key press event (``input/key-press``)."""
        return await self._send_with_retry(TOPIC_INPUT_KEY_PRESS, {"keyCode": key_code})

    async def capture_screenshot(self) -> DABResponse:
        """Request a screenshot capture (``output/image``)."""
        return await self._send_with_retry(TOPIC_OUTPUT_IMAGE, {})

    async def close(self) -> None:
        """Close the underlying transport."""
        await self._transport.close()
        logger.info("AdapterDABClient closed (device_id=%s)", self._device_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_with_retry(
        self,
        topic_template: str,
        payload: Dict[str, Any],
    ) -> DABResponse:
        """Resolve topic, send with timeout, retry on failure, return DABResponse.

        Retries on :class:`asyncio.TimeoutError` and transport errors.
        :class:`NotImplementedError` and :class:`asyncio.CancelledError`
        propagate immediately without retrying.

        Raises:
            DABError: When all attempts are exhausted.
        """
        from vertex_live_dab_agent.dab.transport import TransportRequest

        topic = format_topic(topic_template, self._device_id)
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            req_id = str(uuid.uuid4())
            request = TransportRequest(
                topic=topic,
                payload=payload,
                request_id=req_id,
                timeout=self._timeout,
            )
            try:
                logger.info(
                    "DAB send: topic=%s req_id=%s attempt=%d/%d payload=%r",
                    topic,
                    req_id,
                    attempt + 1,
                    self._max_retries + 1,
                    payload,
                )
                transport_resp = await asyncio.wait_for(
                    self._transport.send(request),
                    timeout=self._timeout,
                )
                resp = DABResponse(
                    success=transport_resp.status < 400,
                    status=transport_resp.status,
                    data=transport_resp.payload,
                    topic=topic,
                    request_id=req_id,
                )
                logger.info("DAB recv: %s", resp)
                return resp

            except (NotImplementedError, asyncio.CancelledError):
                # Programming error or task cancellation -- do not retry.
                raise

            except asyncio.TimeoutError:
                last_exc = asyncio.TimeoutError(
                    f"DAB request timed out after {self._timeout}s (topic={topic})"
                )
                logger.warning(
                    "DAB timeout: topic=%s req_id=%s attempt=%d/%d timeout=%.1fs",
                    topic,
                    req_id,
                    attempt + 1,
                    self._max_retries + 1,
                    self._timeout,
                )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "DAB error: topic=%s req_id=%s attempt=%d/%d error=%s",
                    topic,
                    req_id,
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )

            if attempt < self._max_retries:
                back_off = 0.1 * (2**attempt)
                logger.info(
                    "DAB retry in %.2fs (next attempt %d/%d)",
                    back_off,
                    attempt + 2,
                    self._max_retries + 1,
                )
                await asyncio.sleep(back_off)

        raise DABError(
            f"DAB request failed after {self._max_retries + 1} attempt(s) "
            f"(topic={topic}): {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_dab_client() -> DABClientBase:
    """Return the appropriate DAB client based on configuration.

    * ``DAB_MOCK_MODE=true``  -> :class:`MockDABClient` (default)
    * ``DAB_MOCK_MODE=false`` -> :class:`AdapterDABClient` wrapping
      :class:`~vertex_live_dab_agent.dab.transport.MQTTTransport`.
      The server starts even if the transport is not yet wired; requests will
      raise :class:`NotImplementedError` with clear guidance until wired.
    """
    config = get_config()
    if config.dab_mock_mode:
        logger.info("Creating MockDABClient (DAB_MOCK_MODE=true)")
        return MockDABClient()

    from vertex_live_dab_agent.dab.transport import MQTTTransport

    logger.info(
        "Creating AdapterDABClient with MQTTTransport "
        "(broker=%s:%d device_id=%s)",
        config.dab_mqtt_broker,
        config.dab_mqtt_port,
        config.dab_device_id,
    )
    transport = MQTTTransport(
        broker=config.dab_mqtt_broker,
        port=config.dab_mqtt_port,
    )
    return AdapterDABClient(
        transport=transport,
        device_id=config.dab_device_id,
        timeout=config.dab_request_timeout,
        max_retries=config.dab_max_retries,
    )
