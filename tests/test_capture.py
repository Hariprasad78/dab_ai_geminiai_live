"""Tests for screenshot capture helpers."""

from vertex_live_dab_agent.capture.capture import extract_output_image_b64


def test_extract_output_image_b64_supports_image_key():
    payload = {"image": "aGVsbG8="}
    assert extract_output_image_b64(payload) == "aGVsbG8="


def test_extract_output_image_b64_supports_output_image_key_and_data_uri():
    payload = {"outputImage": "data:image/png;base64, aGV sbG8"}
    # whitespace removed + base64 padding normalized
    assert extract_output_image_b64(payload) == "aGVsbG8="


def test_extract_output_image_b64_returns_none_on_missing_fields():
    assert extract_output_image_b64({"status": 200}) is None
