"""Tests for screenshot capture helpers."""

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
