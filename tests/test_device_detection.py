"""Tests for ADB-backed device detection helpers."""

import pytest

from vertex_live_dab_agent.system_ops import device_detection as dd


def test_get_device_connection_type_supports_tcp_and_usb() -> None:
    assert dd.get_device_connection_type("10.99.57.61:5555") == "tcp"
    assert dd.get_device_connection_type("GZ2407080N470030") == "usb"
    assert dd.get_device_connection_type("") == "unknown"


@pytest.mark.asyncio
async def test_get_device_platform_info_detects_android_tv(monkeypatch) -> None:
    async def _fake_run(device_id, args, timeout_seconds=12.0):
        if args == ["get-state"]:
            return 0, "device", ""
        if args == ["shell", "getprop", "ro.build.version.sdk"]:
            return 0, "34", ""
        if args == ["shell", "getprop", "ro.product.device"]:
            return 0, "sabrina", ""
        if args == ["shell", "getprop", "ro.build.characteristics"]:
            return 0, "tv", ""
        if args == ["shell", "pm", "list", "features"]:
            return 0, "feature:android.software.leanback\nfeature:android.hardware.type.television", ""
        return 1, "", "unexpected"

    monkeypatch.setattr(dd, "_run_adb", _fake_run)
    info = await dd.get_device_platform_info("10.99.57.61:5555")

    assert info["reachable"] is True
    assert info["connection_type"] == "tcp"
    assert info["is_android"] is True
    assert info["is_android_tv"] is True
    assert info["sdk"] == "34"
    assert info["product"] == "sabrina"
    assert "android.software.leanback" in info["tv_features"]


@pytest.mark.asyncio
async def test_get_device_platform_info_detects_android_non_tv(monkeypatch) -> None:
    async def _fake_run(device_id, args, timeout_seconds=12.0):
        if args == ["get-state"]:
            return 0, "device", ""
        if args == ["shell", "getprop", "ro.build.version.sdk"]:
            return 0, "33", ""
        if args == ["shell", "getprop", "ro.product.device"]:
            return 0, "oriole", ""
        if args == ["shell", "getprop", "ro.build.characteristics"]:
            return 0, "nosdcard", ""
        if args == ["shell", "pm", "list", "features"]:
            return 0, "feature:android.hardware.camera", ""
        return 1, "", "unexpected"

    monkeypatch.setattr(dd, "_run_adb", _fake_run)
    info = await dd.get_device_platform_info("GZ2407080N470030")

    assert info["reachable"] is True
    assert info["connection_type"] == "usb"
    assert info["is_android"] is True
    assert info["is_android_tv"] is False
    assert info["tv_features"] == []


@pytest.mark.asyncio
async def test_get_device_platform_info_unreachable_device(monkeypatch) -> None:
    async def _fake_run(device_id, args, timeout_seconds=12.0):
        if args == ["get-state"]:
            return 1, "", "device offline"
        return 1, "", "unexpected"

    monkeypatch.setattr(dd, "_run_adb", _fake_run)
    info = await dd.get_device_platform_info("10.1.2.3:5555")

    assert info["reachable"] is False
    assert info["is_android"] is False
    assert info["is_android_tv"] is False
    assert "offline" in str(info.get("error", "")).lower()
