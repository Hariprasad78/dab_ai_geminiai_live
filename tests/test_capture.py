"""Tests for screenshot capture helpers."""

import time

import pytest

from vertex_live_dab_agent.capture.capture import ScreenCapture, extract_output_image_b64
from vertex_live_dab_agent.capture.hdmi_capture import HdmiCaptureSession


def test_extract_output_image_b64_supports_image_key():
    payload = {"image": "aGVsbG8="}
    assert extract_output_image_b64(payload) == "aGVsbG8="


def test_extract_output_image_b64_supports_output_image_key_and_data_uri():
    payload = {"outputImage": "data:image/png;base64, aGV sbG8"}
    # whitespace removed + base64 padding normalized
    assert extract_output_image_b64(payload) == "aGVsbG8="


def test_extract_output_image_b64_returns_none_on_missing_fields():
    assert extract_output_image_b64({"status": 200}) is None


@pytest.mark.asyncio
async def test_screen_capture_does_not_run_local_ocr(monkeypatch):
    class FakeDab:
        async def capture_screenshot(self):
            raise AssertionError("DAB fallback should not be used in this test")

    capture = ScreenCapture(FakeDab())
    monkeypatch.setattr(capture, "_capture_from_hdmi", lambda: "image-b64")
    capture._image_source = "hdmi-capture"

    result = await capture.capture()

    assert result.image_b64 == "image-b64"
    assert result.ocr_text is None


def test_hdmi_capture_session_forces_720p_resolution():
    session = HdmiCaptureSession(device="/dev/video0", width=1920, height=1080)
    assert session.width == 1280
    assert session.height == 720


def test_hdmi_capture_session_allows_initial_frame_warmup():
    session = HdmiCaptureSession(device="/dev/video0")
    session._cap = object()
    session._opened_at = time.monotonic()
    session._frame_ready.set()

    assert session.read_frame() is None
    assert session.last_error is None


def test_stream_frame_keeps_warming_capture_session():
    class FakeDab:
        async def capture_screenshot(self):
            return None

    class WarmingSession:
        device = "/dev/video0"
        last_error = None

        def __init__(self):
            self.closed = 0

        def capture_jpeg_bytes(self, quality=80):
            return None

        def close(self):
            self.closed += 1

    capture = ScreenCapture(FakeDab())
    session = WarmingSession()
    capture._hdmi = session

    assert capture.get_hdmi_stream_frame_jpeg() is None
    assert capture._hdmi is session
    assert session.closed == 0


def test_init_hdmi_session_does_not_fallback_when_explicit_device_selected(monkeypatch):
    class FakeDab:
        async def capture_screenshot(self):
            return None

    class FakeSession:
        attempts = []

        def __init__(self, device, width, height, fps, fourcc, rotation_degrees):
            self.device = device
            self.last_error = "open failed"

        def open(self):
            FakeSession.attempts.append(self.device)
            return False

        def read_frame(self):
            return None

        def close(self):
            return None

    capture = ScreenCapture(FakeDab())
    capture._image_source = "hdmi-capture"
    capture._selected_video_device = "/dev/video42"

    monkeypatch.setattr(capture, "_list_video_device_details", lambda: [])
    monkeypatch.setattr("vertex_live_dab_agent.capture.capture.HdmiCaptureSession", FakeSession)
    monkeypatch.setattr("vertex_live_dab_agent.capture.capture.os.path.exists", lambda p: str(p).startswith("/dev/video"))
    monkeypatch.setattr("vertex_live_dab_agent.capture.capture.os.access", lambda *_args, **_kwargs: True)

    session = capture._init_hdmi_session()

    assert session is None
    assert FakeSession.attempts == ["/dev/video42"]


def test_set_capture_preference_rejects_non_capture_endpoint(monkeypatch):
    class FakeDab:
        async def capture_screenshot(self):
            return None

    capture = ScreenCapture(FakeDab())
    monkeypatch.setattr("vertex_live_dab_agent.capture.capture.os.path.exists", lambda _p: True)
    monkeypatch.setattr(capture, "_is_capture_capable_device", lambda _dev: False)

    with pytest.raises(ValueError, match="not capture-capable"):
        capture.set_capture_preference(device="/dev/video5", persist=False)


def test_device_contexts_support_duplicate_models_with_explicit_paths(tmp_path, monkeypatch):
    import json
    import vertex_live_dab_agent.capture.camera_devices as camera_devices

    config_path = tmp_path / "camera_devices.json"
    config_path.write_text(json.dumps({"devices": [
        {"contextId": "lab-tv-a", "displayName": "Same Model", "dabDeviceId": "dab-a", "ytsDeviceId": "11", "cameraPath": "/dev/v4l/by-id/cam-a", "videoSource": "camera-capture"},
        {"contextId": "lab-tv-b", "displayName": "Same Model", "dabDeviceId": "dab-b", "ytsDeviceId": "12", "cameraPath": "/dev/v4l/by-id/cam-b", "videoSource": "camera-capture"},
    ]}), encoding="utf-8")
    monkeypatch.setenv("CAMERA_DEVICES_CONFIG", str(config_path))
    camera_devices.clear_device_context_cache()

    contexts = camera_devices.load_device_contexts()

    assert [ctx.contextId for ctx in contexts] == ["lab-tv-a", "lab-tv-b"]
    assert camera_devices.find_device_context("dab-b").cameraPath == "/dev/v4l/by-id/cam-b"
    assert camera_devices.find_device_context("Same Model") is not None
    assert camera_devices.get_camera_path("lab-tv-a") == "/dev/v4l/by-id/cam-a"


def test_upsert_device_context_binding_persists_discovered_hardware_ids(tmp_path, monkeypatch):
    import json
    import vertex_live_dab_agent.capture.camera_devices as camera_devices

    config_path = tmp_path / "camera_devices.json"
    config_path.write_text(json.dumps({"devices": [
        {"contextId": "slot-1", "displayName": "Bench Slot 1", "dabDeviceId": "old-dab", "ytsDeviceId": "old-yts"}
    ]}), encoding="utf-8")
    monkeypatch.setenv("CAMERA_DEVICES_CONFIG", str(config_path))
    camera_devices.clear_device_context_cache()

    updated = camera_devices.upsert_device_context_binding("slot-1", {
        "dabDeviceId": "new-dab-id",
        "ytsDeviceId": "42",
        "adbDeviceId": "10.0.0.5:5555",
        "irDeviceId": "ir-slot-1",
        "cameraPath": "/dev/v4l/by-id/unique-camera",
        "videoSource": "hdmi-capture",
        "audioDevice": "hw:7,0",
    })

    saved = json.loads(config_path.read_text(encoding="utf-8"))["devices"][0]
    assert updated.dabDeviceId == "new-dab-id"
    assert saved["cameraPath"] == "/dev/v4l/by-id/unique-camera"
    assert saved["irDeviceId"] == "ir-slot-1"
    assert camera_devices.find_device_context("new-dab-id").contextId == "slot-1"
