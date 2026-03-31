from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from vertex_live_dab_agent.multi_camera.device_registry import DeviceRegistry, parse_device_specs, resolve_device_spec
from vertex_live_dab_agent.multi_camera.main import create_app


class FakeRegistry:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self._devices = [
            {
                "device_id": "cam1",
                "locator": "by-id:/dev/v4l/by-id/cam1",
                "device_path": "/dev/video0",
                "kind": "camera",
                "required": True,
                "state": "AVAILABLE",
                "is_open": True,
                "first_frame_received": True,
                "frames_captured": 10,
                "fps": 29.5,
                "reconnect_attempts": 0,
                "last_frame_at": "2026-03-31T00:00:00+00:00",
                "last_frame_age_seconds": 0.25,
                "started_at": "2026-03-31T00:00:00+00:00",
                "last_state_change_at": "2026-03-31T00:00:00+00:00",
                "open_success_at": "2026-03-31T00:00:00+00:00",
                "first_frame_at": "2026-03-31T00:00:00+00:00",
                "last_error": None,
                "initialization_complete": True,
                "monitor_message": None,
                "frame_available": True,
            },
            {
                "device_id": "cam2",
                "locator": "by-id:/dev/v4l/by-id/cam2",
                "device_path": "/dev/video1",
                "kind": "camera",
                "required": False,
                "state": "DEGRADED",
                "is_open": False,
                "first_frame_received": False,
                "frames_captured": 0,
                "fps": 0.0,
                "reconnect_attempts": 2,
                "last_frame_at": None,
                "last_frame_age_seconds": None,
                "started_at": "2026-03-31T00:00:00+00:00",
                "last_state_change_at": "2026-03-31T00:01:00+00:00",
                "open_success_at": None,
                "first_frame_at": None,
                "last_error": "waiting for first frame",
                "initialization_complete": True,
                "monitor_message": "waiting for first frame",
                "frame_available": False,
            }
        ]

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def health(self) -> dict:
        return {
            "status": "ok",
            "required_device_count": 1,
            "available_required_count": 1,
            "failed_required_count": 0,
            "devices": [
                {
                    "device_id": "cam1",
                    "state": "AVAILABLE",
                    "frame_available": True,
                    "last_frame_at": "2026-03-31T00:00:00+00:00",
                    "last_frame_age_seconds": 0.25,
                    "last_error": None,
                    "device_path": "/dev/video0",
                    "locator": "by-id:/dev/v4l/by-id/cam1",
                }
            ],
        }

    def list_devices(self) -> list[dict]:
        return list(self._devices)

    def stream_status(self) -> dict:
        return {
            "status": "ok",
            "device_count": 2,
            "required_device_count": 1,
            "available_device_count": 1,
            "available_required_count": 1,
            "failed_required_count": 0,
            "startup_report": [],
            "devices": self.list_devices(),
        }

    def latest_frame(self, device_id: str) -> bytes | None:
        if device_id == "cam1":
            return b"jpeg-bytes"
        if device_id == "cam2":
            return None
        if device_id != "cam1":
            raise KeyError(device_id)
        return b"jpeg-bytes"


@pytest.mark.asyncio
async def test_multi_camera_app_endpoints():
    registry = FakeRegistry()
    app = create_app(registry=registry)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        devices = await client.get("/devices")
        assert devices.status_code == 200
        assert devices.json()["devices"][0]["device_id"] == "cam1"

        status = await client.get("/stream-status")
        assert status.status_code == 200
        assert status.json()["available_device_count"] == 1
        assert status.json()["devices"][0]["last_frame_age_seconds"] == 0.25

        snapshot = await client.get("/snapshot/cam1")
        assert snapshot.status_code == 200
        assert snapshot.content == b"jpeg-bytes"
        assert snapshot.headers["content-type"] == "image/jpeg"
        assert snapshot.headers["x-placeholder-frame"] == "false"

        placeholder = await client.get("/snapshot/cam2")
        assert placeholder.status_code == 200
        assert placeholder.headers["content-type"] == "image/jpeg"
        assert placeholder.headers["x-placeholder-frame"] == "true"
        assert placeholder.content

        missing = await client.get("/snapshot/missing")
        assert missing.status_code == 404


def test_parse_device_specs_defaults():
    parsed = parse_device_specs(None)
    assert [item.device_id for item in parsed] == ["cam1", "cam2", "hdmi"]
    assert parsed[-1].kind == "hdmi"
    assert parsed[0].locator.startswith("by-id:")


def test_validate_configured_devices_marks_missing_required(monkeypatch):
    monkeypatch.setattr("vertex_live_dab_agent.multi_camera.device_registry.Path.exists", lambda self: False)
    registry = DeviceRegistry.from_env()
    report = registry.validate_configured_devices()
    assert any(item["state"] == "FAILED" for item in report)


def test_resolve_device_spec_supports_usb_mapping(monkeypatch):
    monkeypatch.setattr(
        "vertex_live_dab_agent.multi_camera.device_registry.glob.glob",
        lambda pattern: [
            "/dev/v4l/by-id/usb-Elgato_HDMI_Capture-video-index0",
            "/dev/v4l/by-id/usb-Logitech_Camera-video-index0",
        ],
    )
    monkeypatch.setattr(Path, "resolve", lambda self: Path("/dev/video2") if "Elgato" in str(self) else Path("/dev/video0"))

    resolved = resolve_device_spec(parse_device_specs("hdmi|hdmi|usb:elgato")[0])
    assert resolved.device_path == "/dev/video2"
    assert resolved.resolution_error is None
