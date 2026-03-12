"""DAB client abstraction with mock and adapter interface."""
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
    """Mock DAB client for local development and testing."""

    def __init__(self) -> None:
        self._config = get_config()
        self._call_count: Dict[str, int] = {}

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
    """
    Real DAB client using MQTT transport.

    TODO: Implement with actual MQTT library (e.g., aiomqtt or paho-mqtt).
    This is a stub/adapter interface. Replace with real MQTT implementation
    when connecting to actual Android TV / Google TV devices.
    """

    def __init__(self) -> None:
        self._config = get_config()
        # TODO: Initialize actual MQTT client
        # self._mqtt = aiomqtt.Client(...)
        raise NotImplementedError(
            "MQTTDABClient requires a real MQTT broker. "
            "Set DAB_MOCK_MODE=true to use the mock client."
        )

    async def launch_app(self, app_id: str, parameters: Optional[Dict[str, Any]] = None) -> DABResponse:
        raise NotImplementedError

    async def get_app_state(self, app_id: str) -> DABResponse:
        raise NotImplementedError

    async def key_press(self, key_code: str) -> DABResponse:
        raise NotImplementedError

    async def capture_screenshot(self) -> DABResponse:
        raise NotImplementedError

    async def close(self) -> None:
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
