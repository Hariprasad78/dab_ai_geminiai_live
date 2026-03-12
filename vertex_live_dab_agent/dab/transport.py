"""DAB transport interface layer.

The transport is the lowest layer in the DAB stack. It is responsible *only*
for the I/O mechanics: taking an already-resolved topic + payload and returning
a raw response payload. All DAB-protocol logic (topic construction, payload
shape, response normalization, timeout/retry) lives in the client layer above.

Layer diagram::

    Planner / API layer              ← goals, ActionType, PlannedAction
    ─────────────────────────────────────────────────────────────────────
    DABClientBase / AdapterDABClient ← DAB commands, normalization, retry
    ─────────────────────────────────────────────────────────────────────
    DABTransportBase                 ← I/O contract: send → receive
    MQTTTransport (stub)             ← user wires real MQTT library here
    ─────────────────────────────────────────────────────────────────────

Extension contract
------------------
To add a new transport backend (MQTT, HTTP, WebSocket …):

1. Subclass :class:`DABTransportBase`.
2. Implement :meth:`~DABTransportBase.send` and :meth:`~DABTransportBase.close`.
3. Pass your transport instance to
   :class:`~vertex_live_dab_agent.dab.client.AdapterDABClient`.
4. Optionally register it in :func:`~vertex_live_dab_agent.dab.client.create_dab_client`
   or inject it at test / startup time.

The *only* class that needs user wiring is :class:`MQTTTransport`.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport request / response data classes
# ---------------------------------------------------------------------------


@dataclass
class TransportRequest:
    """A single outbound DAB request.

    Attributes:
        topic:      Fully-resolved DAB topic string (``{device_id}`` already
                    substituted via :func:`~vertex_live_dab_agent.dab.topics.format_topic`).
        payload:    JSON-serialisable dict to send as the request body.
        request_id: Unique identifier for correlation / logging.
        timeout:    How long (seconds) to wait for the device response.
    """

    topic: str
    payload: Dict[str, Any]
    request_id: str
    timeout: float = 10.0


@dataclass
class TransportResponse:
    """A raw response received from the transport.

    Attributes:
        topic:      The topic this response was received on (may be the
                    ``<request_topic>/response`` suffix convention).
        payload:    Raw JSON payload dict from the device.
        request_id: Echo of the originating request ID (if supported by device).
        status:     HTTP-style status code extracted from the payload
                    (default 200 – transport implementations should set this
                    from ``payload["status"]`` when present).
    """

    topic: str
    payload: Dict[str, Any]
    request_id: str
    status: int = 200


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DABTransportError(Exception):
    """Raised by transport implementations on communication failure.

    Distinct from :class:`asyncio.TimeoutError` (which indicates the device
    did not respond within the configured deadline) and
    :class:`NotImplementedError` (which indicates the transport is not wired).
    """


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DABTransportBase(ABC):
    """Abstract base class for DAB transport backends.

    Implementations must:

    * Send a request payload to the given topic.
    * Wait for a response on the corresponding response topic (typically
      ``<request_topic>/response`` in the DAB spec).
    * Return a :class:`TransportResponse` on success.
    * Raise :class:`DABTransportError` on non-timeout communication failures.
    * Let :class:`asyncio.TimeoutError` propagate naturally so the client
      layer can apply its retry/timeout policy.
    """

    @abstractmethod
    async def send(self, request: TransportRequest) -> TransportResponse:
        """Send *request* and await the device response.

        Args:
            request: The fully-constructed request to send.

        Returns:
            A :class:`TransportResponse` containing the raw device payload.

        Raises:
            DABTransportError: On communication failure.
            asyncio.TimeoutError: If no response arrives within the deadline.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this transport (connections, sockets …)."""
        ...


# ---------------------------------------------------------------------------
# MQTTTransport — needs user wiring
# ---------------------------------------------------------------------------


class MQTTTransport(DABTransportBase):
    """MQTT transport adapter — **stub that needs user wiring**.

    This class provides the correct :class:`DABTransportBase` interface so
    that :class:`~vertex_live_dab_agent.dab.client.AdapterDABClient` and the
    rest of the stack can be constructed and exercised (e.g. in tests with a
    :class:`FakeTransport`) without a live broker.

    To wire up a real MQTT connection:

    1. Install an async MQTT library::

           pip install aiomqtt        # recommended
           # or: pip install paho-mqtt

    2. Connect the client in ``__init__``::

           import aiomqtt
           self._client = aiomqtt.Client(
               hostname=self._broker,
               port=self._port,
           )

    3. Implement ``send``::

           async def send(self, request: TransportRequest) -> TransportResponse:
               response_topic = request.topic + "/response"
               async with self._client:
                   await self._client.subscribe(response_topic)
                   await self._client.publish(
                       request.topic,
                       payload=json.dumps(request.payload),
                   )
                   async with asyncio.timeout(request.timeout):
                       async for msg in self._client.messages:
                           raw = json.loads(msg.payload)
                           return TransportResponse(
                               topic=response_topic,
                               payload=raw,
                               request_id=request.request_id,
                               status=raw.get("status", 200),
                           )

    4. Implement ``close``::

           async def close(self) -> None:
               await self._client.__aexit__(None, None, None)

    5. Set ``DAB_MOCK_MODE=false`` in ``.env`` — the factory
       :func:`~vertex_live_dab_agent.dab.client.create_dab_client` will
       automatically wrap this transport in :class:`AdapterDABClient`.
    """

    def __init__(self, broker: str = "localhost", port: int = 1883) -> None:
        self._broker = broker
        self._port = port
        logger.warning(
            "MQTTTransport: broker=%s port=%d — transport not yet implemented. "
            "Implement MQTTTransport.send() in dab/transport.py to enable "
            "real DAB device communication. See class docstring for instructions.",
            broker,
            port,
        )

    async def send(self, request: TransportRequest) -> TransportResponse:
        """Not implemented — see class docstring for wiring instructions."""
        raise NotImplementedError(
            "MQTTTransport.send() is not implemented. "
            "See the MQTTTransport class docstring in dab/transport.py for "
            "step-by-step wiring instructions, or set DAB_MOCK_MODE=true to "
            "use MockDABClient instead."
        )

    async def close(self) -> None:
        """No-op until the transport is implemented."""
        pass
