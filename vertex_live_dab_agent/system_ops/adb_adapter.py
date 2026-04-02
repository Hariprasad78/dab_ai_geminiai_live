"""Small ADB adapter surface for deterministic fallback flows."""

from __future__ import annotations

from typing import Any, Dict, Optional

from vertex_live_dab_agent.android_timezone import (
    get_timezone_via_adb,
    set_timezone_via_adb,
)


async def adb_get_setting(*, setting_key: str, adb_device_id: Optional[str]) -> Dict[str, Any]:
    key = str(setting_key or "").strip().lower()
    if key in {"timezone", "time_zone", "time-zone", "time zone"}:
        ok, value = await get_timezone_via_adb(str(adb_device_id or ""))
        return {"success": bool(ok), "value": value, "source": "adb"}
    return {"success": False, "error": f"adb adapter does not support setting '{setting_key}'"}


async def adb_set_setting(*, setting_key: str, value: Any, adb_device_id: Optional[str]) -> Dict[str, Any]:
    key = str(setting_key or "").strip().lower()
    if key in {"timezone", "time_zone", "time-zone", "time zone"}:
        ok = await set_timezone_via_adb(str(adb_device_id or ""), str(value or ""))
        return {"success": bool(ok), "source": "adb"}
    return {"success": False, "error": f"adb adapter does not support setting '{setting_key}'"}
