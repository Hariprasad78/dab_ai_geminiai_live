"""Tests for Android timezone ADB helper module."""

import pytest

from vertex_live_dab_agent import android_timezone as tz


@pytest.mark.asyncio
async def test_list_timezones_uses_apex_fallback_when_primary_missing(monkeypatch) -> None:
    calls = []

    async def _fake_run(device_id, args, timeout_seconds=15.0):
        calls.append((device_id, args))
        cmd = " ".join(args)
        if "/system/usr/share/zoneinfo/tzdata" in cmd:
            return 1, "", "No such file"
        return 0, "America/Los_Angeles\nEurope/London\n", ""

    monkeypatch.setattr(tz, "_run_adb", _fake_run)
    result = await tz.list_timezones_via_adb("emulator-5554")

    assert result["success"] is True
    assert "America/Los_Angeles" in result["timezones"]
    assert result["source"] == "fallback"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_set_timezone_requires_exact_readback_match(monkeypatch) -> None:
    async def _fake_run(device_id, args, timeout_seconds=15.0):
        cmd = " ".join(args)
        if "settings put global time_zone" in cmd:
            return 0, "", ""
        if "settings get global time_zone" in cmd:
            return 0, "UTC", ""
        return 1, "", "unexpected"

    monkeypatch.setattr(tz, "_run_adb", _fake_run)
    result = await tz.set_timezone_via_adb("emulator-5554", "America/Los_Angeles")

    assert result["success"] is False
    assert result["requested_timezone"] == "America/Los_Angeles"
    assert result["observed_timezone"] == "UTC"


@pytest.mark.asyncio
async def test_set_timezone_succeeds_when_verified(monkeypatch) -> None:
    async def _fake_run(device_id, args, timeout_seconds=15.0):
        cmd = " ".join(args)
        if "settings put global time_zone" in cmd:
            return 0, "", ""
        if "settings get global time_zone" in cmd:
            return 0, "America/Los_Angeles", ""
        return 1, "", "unexpected"

    monkeypatch.setattr(tz, "_run_adb", _fake_run)
    result = await tz.set_timezone_via_adb("emulator-5554", "America/Los_Angeles")

    assert result["success"] is True
    assert result["verified"] is True
    assert result["observed_timezone"] == "America/Los_Angeles"
