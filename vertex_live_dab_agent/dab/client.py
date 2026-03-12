"""DAB client abstraction with mock and real-device adapter interface.

Extension guide
---------------
To connect to a real Android TV / Google TV device you need to implement
:class:`MQTTDABClient` (or write an HTTP adapter that subclasses
:class:`DABClientBase`).

Steps:
1. Install an async MQTT library, e.g. ``pip install aiomqtt``.
2. In :class:`MQTTDABClient.__init__`, create and connect the MQTT client.
3. Implement each abstract method:  publish a request payload to the resolved
   topic, subscribe to the response topic, await the reply, and return a
   :class:`DABResponse`.
4. Set ``DAB_MOCK_MODE=false`` in your ``.env`` and the factory function
   :func:`create_dab_client` will return your real client automatically.
"""
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.topics import (
    KEY_MAP,
    TOPIC_APPLICATIONS_GET_STATE,
    TOPIC_APPLICATIONS_LAUNCH,
    TOPIC_INPUT_KEY_PRESS,
    TOPIC_OUTPUT_IMAGE,
)

logger = logging.getLogger(__name__)


class DABError(Exception):
    """DAB client error."""


class DABResponse:
    """Normalized DAB response."""

    def __init__(self, success: bool, status: int, data: Dict[str, Any], topic: str, request_id: str):
        self.success = success
        self.status = status
        self.data = data
        self.topic = topic
        self.request_id = request_id

    def __repr__(self) -> str:
        return f"DABResponse(success={self.success}, status={self.status}, topic={self.topic})"


class DABClientBase(ABC):
    """Abstract base class for DAB clients."""

    @abstractmethod
    async def launch_app(self, app_id: str, parameters: Optional[Dict[str, Any]] = None) -> DABResponse:
        """Launch an application."""
        ...

    @abstractmethod
    async def get_app_state(self, app_id: str) -> DABResponse:
        """Get application state."""
        ...

    @abstractmethod
    async def key_press(self, key_code: str) -> DABResponse:
        """Send a key press."""
        ...

    @abstractmethod
    async def capture_screenshot(self) -> DABResponse:
        """Capture a screenshot. Returns base64 image in data['image']."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the client."""
        ...


class MockDABClient(DABClientBase):
    """Mock DAB client for local development and testing.

    All operations succeed immediately with simulated 50 ms latency.
    Screenshots return a 1×1 white PNG placeholder.
    """

    def __init__(self) -> None:
        self._config = get_config()

    async def launch_app(self, app_id: str, parameters: Optional[Dict[str, Any]] = None) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB request: launch_app app_id=%s req_id=%s", app_id, req_id)
        resp = DABResponse(
            success=True,
            status=200,
            data={"appId": app_id, "state": "FOREGROUND"},
            topic=TOPIC_APPLICATIONS_LAUNCH,
            request_id=req_id,
        )
        logger.info("DAB response: %s", resp)
        return resp

    async def get_app_state(self, app_id: str) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB request: get_app_state app_id=%s req_id=%s", app_id, req_id)
        resp = DABResponse(
            success=True,
            status=200,
            data={"appId": app_id, "state": "FOREGROUND"},
            topic=TOPIC_APPLICATIONS_GET_STATE,
            request_id=req_id,
        )
        logger.info("DAB response: %s", resp)
        return resp

    async def key_press(self, key_code: str) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB request: key_press key_code=%s req_id=%s", key_code, req_id)
        resp = DABResponse(
            success=True,
            status=200,
            data={"keyCode": key_code},
            topic=TOPIC_INPUT_KEY_PRESS,
            request_id=req_id,
        )
        logger.info("DAB response: %s", resp)
        return resp

    async def capture_screenshot(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB request: capture_screenshot req_id=%s", req_id)
        # 1x1 white PNG as placeholder
        white_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )
        resp = DABResponse(
            success=True,
            status=200,
            data={"image": white_png, "format": "png"},
            topic=TOPIC_OUTPUT_IMAGE,
            request_id=req_id,
        )
        logger.info("DAB response: %s", resp)
        return resp

    async def close(self) -> None:
        logger.info("MockDABClient closed")

    async def _simulate_latency(self) -> None:
        await asyncio.sleep(0.05)


class MQTTDABClient(DABClientBase):
    """Real DAB client using MQTT transport.

    **This is a stub.** Implement the methods below when connecting to a real
    MQTT broker.  See the module docstring for a step-by-step guide.

    Required config (from environment variables):
        ``DAB_MQTT_BROKER``   – hostname / IP of the MQTT broker
        ``DAB_MQTT_PORT``     – port (default 1883)
        ``DAB_DEVICE_ID``     – DAB device identifier
        ``DAB_REQUEST_TIMEOUT`` – seconds to wait for a response

    Typical MQTT flow for each command::

        topic_req  = format_topic(TOPIC_INPUT_KEY_PRESS, device_id)
        topic_resp = topic_req + "/response"
        payload    = json.dumps({"keyCode": key_code})

        await mqtt_client.publish(topic_req, payload)
        response_msg = await asyncio.wait_for(
            mqtt_client.messages.__anext__(), timeout=request_timeout
        )
        return DABResponse(
            success=response_msg.payload["status"] == 200,
            status=response_msg.payload["status"],
            data=response_msg.payload,
            topic=topic_req,
            request_id=str(uuid.uuid4()),
        )
    """

    def __init__(self) -> None:
        self._config = get_config()
        # TODO: Replace the line below with real MQTT client initialisation.
        # Example using aiomqtt:
        #   self._broker = self._config.dab_mqtt_broker
        #   self._port   = self._config.dab_mqtt_port
        #   self._client = aiomqtt.Client(hostname=self._broker, port=self._port)
        raise NotImplementedError(
            "MQTTDABClient is not yet implemented. "
            "See the module docstring for a step-by-step integration guide, "
            "or set DAB_MOCK_MODE=true to use the mock client instead."
        )

    async def launch_app(self, app_id: str, parameters: Optional[Dict[str, Any]] = None) -> "DABResponse":
        # TODO: publish to format_topic(TOPIC_APPLICATIONS_LAUNCH, device_id)
        raise NotImplementedError

    async def get_app_state(self, app_id: str) -> "DABResponse":
        # TODO: publish to format_topic(TOPIC_APPLICATIONS_GET_STATE, device_id)
        raise NotImplementedError

    async def key_press(self, key_code: str) -> "DABResponse":
        # TODO: publish to format_topic(TOPIC_INPUT_KEY_PRESS, device_id)
        raise NotImplementedError

    async def capture_screenshot(self) -> "DABResponse":
        # TODO: publish to format_topic(TOPIC_OUTPUT_IMAGE, device_id)
        raise NotImplementedError

    async def close(self) -> None:
        # TODO: disconnect MQTT client
        raise NotImplementedError


def create_dab_client() -> DABClientBase:
    """Factory function to create the appropriate DAB client."""
    config = get_config()
    if config.dab_mock_mode:
        logger.info("Creating MockDABClient (DAB_MOCK_MODE=true)")
        return MockDABClient()
    else:
        logger.info("Creating MQTTDABClient (DAB_MOCK_MODE=false)")
        return MQTTDABClient()
