"""Tests for capability snapshot validation and normalization."""

from vertex_live_dab_agent.system_ops.capabilities import (
    build_capability_snapshot,
    has_key,
    has_operation,
    has_setting,
    normalize_key_code,
    normalize_setting_key,
    normalize_setting_value,
)


def _snapshot():
    return build_capability_snapshot(
        supported_operations=["system/settings/get", "system/settings/set", "input/key-press"],
        supported_settings=[
            {"key": "timezone", "friendlyName": "Time Zone", "allowedValues": ["America/Los_Angeles", "UTC"]},
            {"key": "brightness", "type": "number", "min": 0, "max": 100},
            {"key": "cecEnabled", "type": "boolean", "supported": True},
        ],
        supported_keys=["KEY_HOME", "KEY_BACK"],
    )


def test_snapshot_core_support_checks():
    snap = _snapshot()
    assert has_operation(snap, "system/settings/get") is True
    assert has_setting(snap, "timezone") is True
    assert has_key(snap, "KEY_HOME") is True


def test_misspelled_setting_key_is_normalized():
    snap = _snapshot()
    result = normalize_setting_key(snap, "timezoen")
    assert result["success"] is True
    assert result["key"] == "timezone"


def test_misspelled_timezone_value_is_normalized():
    snap = _snapshot()
    result = normalize_setting_value(snap, "timezone", "america/losangle")
    assert result["success"] is True
    assert result["value"] == "America/Los_Angeles"


def test_unsupported_setting_is_rejected():
    snap = _snapshot()
    result = normalize_setting_key(snap, "volume")
    assert result["success"] is False


def test_unsupported_key_code_is_rejected():
    snap = _snapshot()
    result = normalize_key_code(snap, "homee")
    assert result["success"] is True
    assert result["key_code"] == "KEY_HOME"


def test_out_of_range_numeric_setting_is_rejected():
    snap = _snapshot()
    result = normalize_setting_value(snap, "brightness", "300")
    assert result["success"] is False
    assert "out of range" in str(result.get("reason", "")).lower()
