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
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.topics import (
        TOPIC_DEVICE_INFO,
    KEY_MAP,
    TOPIC_APPLICATIONS_EXIT,
    TOPIC_APPLICATIONS_GET_STATE,
    TOPIC_APPLICATIONS_LIST,
    TOPIC_APPLICATIONS_LAUNCH,
    TOPIC_CONTENT_OPEN,
    TOPIC_INPUT_KEY_LIST,
    TOPIC_INPUT_KEY_PRESS,
    TOPIC_INPUT_LONG_KEY_PRESS,
    TOPIC_OPERATIONS_LIST,
    TOPIC_OUTPUT_IMAGE,
    TOPIC_SYSTEM_SETTINGS_GET,
    TOPIC_SYSTEM_SETTINGS_LIST,
    TOPIC_SYSTEM_SETTINGS_SET,
    TOPIC_VOICE_LIST,
    format_topic,
)

if TYPE_CHECKING:
    from vertex_live_dab_agent.dab.transport import DABTransportBase

logger = logging.getLogger(__name__)


_APP_ID_ALIASES: Dict[str, str] = {
    "com.netflix.ninja": "netflix",
    "netflix": "netflix",
    "com.google.android.youtube": "youtube",
    "com.google.android.youtube.tv": "youtube",
    "youtube": "youtube",
    "com.android.settings": "settings",
    "com.android.tv.settings": "settings",
    "settings": "settings",
}


def normalize_app_id_alias(app_id: str) -> str:
    """Normalize known package-style app ids to logical ids expected by DAB."""
    normalized = str(app_id or "").strip()
    if not normalized:
        return normalized
    return _APP_ID_ALIASES.get(normalized.lower(), normalized)


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
    async def long_key_press(
        self, key_code: str, duration_ms: int = 1500
    ) -> DABResponse:
        """Send a long key press event to the device."""
        ...

    @abstractmethod
    async def list_keys(self) -> DABResponse:
        """List supported device key codes."""
        ...

    @abstractmethod
    async def list_operations(self) -> DABResponse:
        """List supported DAB operations for the device."""
        ...

    @abstractmethod
    async def list_apps(self) -> DABResponse:
        """List applications available on device."""
        ...

    @abstractmethod
    async def exit_app(self, app_id: str) -> DABResponse:
        """Exit an application on the device."""
        ...

    @abstractmethod
    async def capture_screenshot(self) -> DABResponse:
        """Capture a screenshot. Returns base64 PNG in ``data["image"]``."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources and close the client."""
        ...

    async def list_settings(self) -> DABResponse:
        """Optional: list supported system settings."""
        return DABResponse(False, 501, {"error": "system/settings/list not supported"}, "", "")

    async def get_setting(self, setting_key: str) -> DABResponse:
        """Optional: query a system setting."""
        return DABResponse(False, 501, {"error": "system/settings/get not supported", "setting": setting_key}, "", "")

    async def get_all_settings_values(self) -> DABResponse:
        """Optional: query all system setting values in a single request."""
        return DABResponse(False, 501, {"error": "bulk system/settings/get not supported"}, "", "")

    async def set_setting(self, setting_key: str, value: Any) -> DABResponse:
        """Optional: update a system setting."""
        return DABResponse(False, 501, {"error": "system/settings/set not supported", "setting": setting_key}, "", "")

    async def open_content(self, content: str, parameters: Optional[Dict[str, Any]] = None) -> DABResponse:
        """Optional: open content directly."""
        payload: Dict[str, Any] = {"content": content}
        if isinstance(parameters, dict):
            payload.update(parameters)
        return DABResponse(False, 501, {"error": "content/open not supported", **payload}, "", "")

    async def discover_devices(self, attempts: int = 1, wait_seconds: float = 1.0) -> DABResponse:
        """Optional: discover available DAB devices."""
        return DABResponse(False, 501, {"error": "discover not supported"}, "", "")

    async def get_device_info(self) -> DABResponse:
        """Optional: return selected device info."""
        return DABResponse(False, 501, {"error": "device/info not supported"}, "", "")

    async def list_voices(self) -> DABResponse:
        """Optional: list available voices on the device."""
        return DABResponse(False, 501, {"error": "voice/list not supported"}, "", "")


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
        app_id = normalize_app_id_alias(app_id)
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: launch_app app_id=%s req_id=%s", app_id, req_id)
        data: Dict[str, Any] = {"appId": app_id, "state": "FOREGROUND"}
        if parameters and parameters.get("content"):
            data["content"] = parameters.get("content")
        return DABResponse(
            success=True,
            status=200,
            data=data,
            topic=TOPIC_APPLICATIONS_LAUNCH,
            request_id=req_id,
        )

    async def get_app_state(self, app_id: str) -> DABResponse:
        await self._simulate_latency()
        app_id = normalize_app_id_alias(app_id)
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

    async def long_key_press(self, key_code: str, duration_ms: int = 1500) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info(
            "DAB mock: long_key_press key_code=%s duration_ms=%s req_id=%s",
            key_code,
            duration_ms,
            req_id,
        )
        return DABResponse(
            success=True,
            status=200,
            data={"keyCode": key_code, "durationMs": int(duration_ms)},
            topic=TOPIC_INPUT_LONG_KEY_PRESS,
            request_id=req_id,
        )

    async def list_keys(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: list_keys req_id=%s", req_id)
        keys = sorted(set(KEY_MAP.values()))
        return DABResponse(
            success=True,
            status=200,
            data={"keys": keys},
            topic=TOPIC_INPUT_KEY_LIST,
            request_id=req_id,
        )

    async def list_operations(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: list_operations req_id=%s", req_id)
        return DABResponse(
            success=True,
            status=200,
            data={
                "operations": [
                    "applications/launch",
                    "applications/get-state",
                    "applications/list",
                    "applications/exit",
                    "input/key/list",
                    "input/key-press",
                    "input/long-key-press",
                    "system/settings/list",
                    "system/settings/get",
                    "system/settings/set",
                    "output/image",
                    "operations/list",
                ]
            },
            topic=TOPIC_OPERATIONS_LIST,
            request_id=req_id,
        )

    async def list_apps(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: list_apps req_id=%s", req_id)
        return DABResponse(
            success=True,
            status=200,
            data={
                "applications": [
                    {
                        "appId": get_config().youtube_app_id,
                        "name": "YouTube",
                    }
                ]
            },
            topic=TOPIC_APPLICATIONS_LIST,
            request_id=req_id,
        )

    async def exit_app(self, app_id: str) -> DABResponse:
        await self._simulate_latency()
        app_id = normalize_app_id_alias(app_id)
        req_id = str(uuid.uuid4())
        logger.info("DAB mock: exit_app app_id=%s req_id=%s", app_id, req_id)
        return DABResponse(
            success=True,
            status=200,
            data={"appId": app_id, "state": "BACKGROUND"},
            topic=TOPIC_APPLICATIONS_EXIT,
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

    async def list_settings(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        return DABResponse(
            success=True,
            status=200,
            data={
                "settings": [
                    {"key": "timezone", "friendlyName": "Time Zone", "writable": True},
                    {"key": "language", "friendlyName": "Language", "writable": True},
                    {"key": "brightness", "friendlyName": "Brightness", "writable": True},
                ]
            },
            topic=TOPIC_SYSTEM_SETTINGS_LIST,
            request_id=req_id,
        )

    async def get_setting(self, setting_key: str) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        return DABResponse(
            success=True,
            status=200,
            data={"key": setting_key, "value": "mock"},
            topic=TOPIC_SYSTEM_SETTINGS_GET,
            request_id=req_id,
        )

    async def get_all_settings_values(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        return DABResponse(
            success=True,
            status=200,
            data={
                "settings": {
                    "timezone": "UTC",
                    "language": "en-US",
                    "brightness": 50,
                }
            },
            topic=TOPIC_SYSTEM_SETTINGS_GET,
            request_id=req_id,
        )

    async def set_setting(self, setting_key: str, value: Any) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        return DABResponse(
            success=True,
            status=200,
            data={"key": setting_key, "value": value, "updated": True},
            topic=TOPIC_SYSTEM_SETTINGS_SET,
            request_id=req_id,
        )

    async def open_content(self, content: str, parameters: Optional[Dict[str, Any]] = None) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        data: Dict[str, Any] = {"content": content}
        if isinstance(parameters, dict):
            data["parameters"] = parameters
        return DABResponse(
            success=True,
            status=200,
            data=data,
            topic=TOPIC_CONTENT_OPEN,
            request_id=req_id,
        )

    async def discover_devices(self, attempts: int = 1, wait_seconds: float = 1.0) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        device_id = str(get_config().dab_device_id or "mock-device").strip() or "mock-device"
        return DABResponse(
            success=True,
            status=200,
            data={
                "devices": [
                    {
                        "deviceId": device_id,
                        "name": "Mock Android TV",
                        "platform": "android-tv",
                        "transport": "mock",
                    }
                ]
            },
            topic="dab/discover",
            request_id=req_id,
        )

    async def get_device_info(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        device_id = str(get_config().dab_device_id or "mock-device").strip() or "mock-device"
        return DABResponse(
            success=True,
            status=200,
            data={
                "deviceId": device_id,
                "name": "Mock Android TV",
                "model": "MockDevice",
                "platform": "android-tv",
                "version": "mock-1.0",
                "transport": "mock",
                "capabilities": ["applications/list", "input/key-press", "output/image", "device/info"],
            },
            topic=TOPIC_DEVICE_INFO,
            request_id=req_id,
        )

    async def list_voices(self) -> DABResponse:
        await self._simulate_latency()
        req_id = str(uuid.uuid4())
        return DABResponse(
            success=True,
            status=200,
            data={"voices": ["default"]},
            topic=TOPIC_VOICE_LIST,
            request_id=req_id,
        )

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
        app_id = normalize_app_id_alias(app_id)
        payload: Dict[str, Any] = {"appId": app_id}
        if parameters:
            if "content" in parameters and parameters["content"]:
                payload["content"] = parameters["content"]
            payload["parameters"] = parameters
        return await self._send_with_retry(TOPIC_APPLICATIONS_LAUNCH, payload)

    async def get_app_state(self, app_id: str) -> DABResponse:
        """Query current application state (``applications/get-state``)."""
        app_id = normalize_app_id_alias(app_id)
        return await self._send_with_retry(
            TOPIC_APPLICATIONS_GET_STATE, {"appId": app_id}
        )

    async def key_press(self, key_code: str) -> DABResponse:
        """Send a key press event (``input/key-press``)."""
        return await self._send_with_retry(TOPIC_INPUT_KEY_PRESS, {"keyCode": key_code})

    async def long_key_press(self, key_code: str, duration_ms: int = 1500) -> DABResponse:
        """Send a long key press event (``input/long-key-press``)."""
        payload = {"keyCode": key_code, "durationMs": int(duration_ms)}
        return await self._send_with_retry(TOPIC_INPUT_LONG_KEY_PRESS, payload)

    async def list_keys(self) -> DABResponse:
        """List supported key codes (``input/key/list``)."""
        return await self._send_with_retry(TOPIC_INPUT_KEY_LIST, {})

    async def list_operations(self) -> DABResponse:
        """List supported operations (``operations/list``)."""
        return await self._send_with_retry(TOPIC_OPERATIONS_LIST, {})

    async def list_apps(self) -> DABResponse:
        """List installed/launchable applications (``applications/list``)."""
        return await self._send_with_retry(TOPIC_APPLICATIONS_LIST, {})

    async def exit_app(self, app_id: str) -> DABResponse:
        """Exit an app (``applications/exit``)."""
        app_id = normalize_app_id_alias(app_id)
        return await self._send_with_retry(TOPIC_APPLICATIONS_EXIT, {"appId": app_id})

    async def capture_screenshot(self) -> DABResponse:
        """Request a screenshot capture (``output/image``)."""
        return await self._send_with_retry(TOPIC_OUTPUT_IMAGE, {})

    async def list_settings(self) -> DABResponse:
        """List direct system settings capabilities.

        Some bridge implementations fail the whole endpoint when one probe
        (commonly CEC via shell command) is unsupported. In that case return a
        safe partial response instead of propagating a hard failure.
        """
        resp = await self._send_with_retry(TOPIC_SYSTEM_SETTINGS_LIST, {})
        if resp.success:
            return resp

        error_text = str((resp.data or {}).get("error") or "")
        lowered = error_text.lower()
        cec_shell_unsupported = (
            "no shell command implementation" in lowered
            or "cec" in lowered
            or "listsupportedsystemsettings" in lowered
            or "listsystemsettings" in lowered
            or "getcecenabled" in lowered
        )
        if not cec_shell_unsupported:
            return resp

        logger.warning(
            "DAB settings degraded: CEC/shell probe unsupported, returning partial settings"
        )
        partial_settings = [
            {
                "key": "timezone",
                "friendlyName": "Time Zone",
                "writable": True,
            },
            {
                "key": "language",
                "friendlyName": "Language",
                "writable": True,
            },
            {
                "key": "cec_enabled",
                "friendlyName": "CEC",
                "writable": False,
                "available": False,
                "reason": "unsupported shell command",
            },
        ]
        return DABResponse(
            success=True,
            status=200,
            data={
                "settings": partial_settings,
                "degraded": True,
                "warning": error_text or "system/settings/list partially unavailable",
            },
            topic=resp.topic,
            request_id=resp.request_id,
        )

    async def get_setting(self, setting_key: str) -> DABResponse:
        """Get one system setting value."""
        return await self._send_with_retry(TOPIC_SYSTEM_SETTINGS_GET, {"key": str(setting_key)})

    async def get_all_settings_values(self) -> DABResponse:
        """Get all settings in one system/settings/get request when supported by the device."""
        return await self._send_with_retry(TOPIC_SYSTEM_SETTINGS_GET, {})

    async def set_setting(self, setting_key: str, value: Any) -> DABResponse:
        """Set one system setting value."""
        canonical_key = str(setting_key)
        # Conformance payload shape expects {"<settingName>": <value>}.
        try:
            resp = await self._send_with_retry(TOPIC_SYSTEM_SETTINGS_SET, {canonical_key: value})
            if bool(getattr(resp, "success", False)):
                return resp
            err_text = str((getattr(resp, "data", {}) or {}).get("error") or "").lower()
            should_fallback_legacy = (
                ("missing" in err_text and "key" in err_text)
                or ("missing" in err_text and "value" in err_text)
                or ("required" in err_text and "key" in err_text)
                or ("required" in err_text and "value" in err_text)
            )
            if not should_fallback_legacy:
                return resp
        except Exception:
            # Fallback to legacy key/value shape below.
            pass
        return await self._send_with_retry(TOPIC_SYSTEM_SETTINGS_SET, {"key": canonical_key, "value": value})

    async def open_content(self, content: str, parameters: Optional[Dict[str, Any]] = None) -> DABResponse:
        """Open content directly when device supports content/open."""
        payload: Dict[str, Any] = {"content": content}
        if isinstance(parameters, dict):
            payload["parameters"] = parameters
        return await self._send_with_retry(TOPIC_CONTENT_OPEN, payload)

    async def discover_devices(self, attempts: int = 1, wait_seconds: float = 1.0) -> DABResponse:
        """Discover available devices using broadcast discovery semantics.

        Primary path:
        - publish to `dab/discovery`
        - set MQTTv5 `ResponseTopic` to a unique reply topic
        - collect multiple device replies for `wait_seconds` across `attempts`
        """
        try:
            devices = await self._discover_devices_broadcast(attempts=attempts, wait_seconds=wait_seconds)
            return DABResponse(
                success=True,
                status=200,
                data={"devices": devices},
                topic="dab/discovery",
                request_id=str(uuid.uuid4()),
            )
        except Exception as exc:
            logger.warning("DAB broadcast discovery failed, trying compatibility path: %s", exc)

        last_exc: Optional[Exception] = None
        for topic in ("dab/discovery", "dab/discover"):
            try:
                return await self._send_resolved_topic_with_retry(topic, {})
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return await self._send_resolved_topic_with_retry("dab/discover", {})

    async def get_device_info(self) -> DABResponse:
        """Fetch device metadata from device/info."""
        return await self._send_with_retry(TOPIC_DEVICE_INFO, {})

    async def list_voices(self) -> DABResponse:
        """List available voices (voice/list)."""
        return await self._send_with_retry(TOPIC_VOICE_LIST, {})

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
        return await self._send_resolved_topic_with_retry(topic, payload)

    async def _send_resolved_topic_with_retry(
        self,
        topic: str,
        payload: Dict[str, Any],
    ) -> DABResponse:
        """Send using a fully-resolved topic with timeout/retry policy."""
        from vertex_live_dab_agent.dab.transport import TransportRequest

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

    async def _discover_devices_broadcast(self, attempts: int = 1, wait_seconds: float = 1.0) -> list[Dict[str, Any]]:
        """Broadcast to dab/discovery and collect responses on unique response topic."""
        try:
            import aiomqtt  # type: ignore
        except Exception as exc:
            raise DABError("Broadcast discovery requires aiomqtt") from exc

        broker = getattr(self._transport, "_broker", None)
        port = getattr(self._transport, "_port", None)
        if not broker or not port:
            raise DABError("Broadcast discovery requires MQTT transport")

        try:
            import paho.mqtt.client as paho_mqtt  # type: ignore
            client = aiomqtt.Client(hostname=broker, port=int(port), protocol=paho_mqtt.MQTTv5)
        except Exception:
            client = aiomqtt.Client(hostname=broker, port=int(port))

        response_topic = f"dab/_response/discovery/{uuid.uuid4().hex}"
        found: Dict[str, Dict[str, Any]] = {}
        n_attempts = max(1, int(attempts or 1))
        wait_s = max(0.2, float(wait_seconds or 1.0))

        def _ingest_payload(payload_obj: Any) -> None:
            if not isinstance(payload_obj, dict):
                return
            candidates: list[Any]
            if isinstance(payload_obj.get("devices"), list):
                candidates = payload_obj.get("devices")
            else:
                candidates = [payload_obj]

            for item in candidates:
                if not isinstance(item, dict):
                    continue
                device_id = str(item.get("deviceId") or item.get("device_id") or item.get("id") or "").strip()
                if not device_id:
                    continue
                if device_id not in found:
                    found[device_id] = {
                        "deviceId": device_id,
                        "name": item.get("name") or item.get("label") or device_id,
                        "ip": item.get("ip") or item.get("ipAddress"),
                    }
                for k, v in item.items():
                    if k not in found[device_id] or found[device_id][k] in (None, ""):
                        found[device_id][k] = v

        async with client:
            await client.subscribe(response_topic)
            messages_iter = client.messages.__aiter__()

            publish_kwargs: Dict[str, Any] = {"payload": "{}"}
            try:
                from paho.mqtt.packettypes import PacketTypes
                from paho.mqtt.properties import Properties

                props = Properties(PacketTypes.PUBLISH)
                props.ResponseTopic = response_topic
                publish_kwargs["properties"] = props
            except Exception:
                pass

            for _ in range(n_attempts):
                await client.publish("dab/discovery", **publish_kwargs)
                deadline = time.monotonic() + wait_s
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        async with asyncio.timeout(remaining):
                            message = await messages_iter.__anext__()
                    except asyncio.TimeoutError:
                        break
                    except StopAsyncIteration:
                        break

                    raw_payload = getattr(message, "payload", b"{}")
                    if isinstance(raw_payload, (bytes, bytearray)):
                        payload_text = raw_payload.decode("utf-8", errors="replace")
                    else:
                        payload_text = str(raw_payload)
                    try:
                        payload_obj = json.loads(payload_text)
                    except Exception:
                        continue
                    _ingest_payload(payload_obj)

        return list(found.values())


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
