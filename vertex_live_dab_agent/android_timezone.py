"""Android timezone handling via ADB with verification and safe fallbacks."""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from zoneinfo import available_timezones
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


_COMMON_TZ_ALIASES = {
    "asia/calcutta": "Asia/Kolkata",
    "asia/kolkata": "Asia/Kolkata",
    "america/losangles": "America/Los_Angeles",
    "america/los_angeles": "America/Los_Angeles",
    "america/los angeles": "America/Los_Angeles",
}


def _normalize_adb_device_id(device_id: str) -> str:
    value = str(device_id or "").strip()
    if value.lower().startswith("adb:"):
        value = value[4:].strip()
    return value


def _canonicalize_timezone_input(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw

    lowered = raw.lower()
    if lowered in _COMMON_TZ_ALIASES:
        return _COMMON_TZ_ALIASES[lowered]

    # Try direct match against known IANA zones, case-insensitive.
    all_tz = available_timezones()
    by_lower = {tz.lower(): tz for tz in all_tz}
    if lowered in by_lower:
        return by_lower[lowered]

    # Normalize separators and fuzzy match.
    candidate = lowered.replace(" ", "_")
    candidate = re.sub(r"_+", "_", candidate)
    if candidate in by_lower:
        return by_lower[candidate]

    best = difflib.get_close_matches(candidate, list(by_lower.keys()), n=1, cutoff=0.88)
    if best:
        return by_lower[best[0]]

    return raw


async def _run_adb(device_id: str, args: List[str], timeout_seconds: float = 15.0) -> Tuple[int, str, str]:
    """Run an adb command for a specific device and return (rc, stdout, stderr)."""
    resolved_device_id = _normalize_adb_device_id(device_id)
    if not resolved_device_id:
        return 2, "", "missing adb device id"

    cmd = ["adb", "-s", resolved_device_id, *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=max(1.0, float(timeout_seconds)))
        stdout = (stdout_b or b"").decode("utf-8", errors="replace").strip()
        stderr = (stderr_b or b"").decode("utf-8", errors="replace").strip()
        return int(proc.returncode or 0), stdout, stderr
    except asyncio.TimeoutError:
        return 124, "", f"adb command timed out after {int(timeout_seconds)}s"
    except FileNotFoundError:
        return 127, "", "adb binary not found"
    except Exception as exc:
        return 1, "", str(exc)


async def is_adb_device_online(device_id: str) -> Tuple[bool, str]:
    """Return whether adb device is online (`device` state)."""
    rc, stdout, stderr = await _run_adb(device_id, ["get-state"], timeout_seconds=8.0)
    state = str(stdout or "").strip().lower()
    if rc == 0 and state == "device":
        logger.info("ADB timezone fallback: device online device_id=%s", device_id)
        return True, "device"
    lowered = f"{stdout} {stderr}".lower()
    if "unauthorized" in lowered:
        detail = "device unauthorized"
    elif "offline" in lowered:
        detail = "device offline"
    elif "not found" in lowered:
        detail = "device not found"
    else:
        detail = stderr or stdout or f"rc={rc}"
    logger.warning("ADB timezone fallback: device unavailable device_id=%s detail=%s", device_id, detail)
    return False, detail


async def get_timezone_via_adb(device_id: str) -> Dict[str, Any]:
    """Get current timezone via `adb shell settings get global time_zone`."""
    rc, stdout, stderr = await _run_adb(device_id, ["shell", "settings", "get", "global", "time_zone"])
    value = str(stdout or "").strip()
    if rc == 0 and value:
        logger.info("ADB timezone get succeeded device_id=%s timezone=%s", device_id, value)
        return {"success": True, "timezone": value}
    err = stderr or stdout or f"rc={rc}"
    logger.warning("ADB timezone get failed device_id=%s error=%s", device_id, err)
    return {"success": False, "timezone": value or None, "error": err}


async def get_persist_timezone_via_adb(device_id: str) -> Dict[str, Any]:
    """Get active persisted timezone via `adb shell getprop persist.sys.timezone`."""
    rc, stdout, stderr = await _run_adb(device_id, ["shell", "getprop", "persist.sys.timezone"])
    value = str(stdout or "").strip()
    if rc == 0 and value:
        logger.info("ADB persist timezone get succeeded device_id=%s timezone=%s", device_id, value)
        return {"success": True, "timezone": value}
    err = stderr or stdout or f"rc={rc}"
    logger.warning("ADB persist timezone get failed device_id=%s error=%s", device_id, err)
    return {"success": False, "timezone": value or None, "error": err}


async def list_timezones_via_adb(device_id: str) -> Dict[str, Any]:
    """List available timezone IDs from Android tzdata.

    Primary path:
    - strings /system/usr/share/zoneinfo/tzdata | grep /

    Fallback path:
    - strings /apex/com.android.tzdata/etc/tz/tzdata | grep /
    """
    commands = [
        "strings /system/usr/share/zoneinfo/tzdata | grep /",
        "strings /apex/com.android.tzdata/etc/tz/tzdata | grep /",
    ]
    last_error = ""
    for idx, shell_cmd in enumerate(commands):
        rc, stdout, stderr = await _run_adb(device_id, ["shell", "sh", "-c", shell_cmd], timeout_seconds=20.0)
        lines = [line.strip() for line in str(stdout or "").splitlines() if "/" in line.strip()]
        deduped = sorted(set(lines))
        if rc == 0 and deduped:
            source = "primary" if idx == 0 else "fallback"
            logger.info(
                "ADB timezone list succeeded device_id=%s source=%s count=%d",
                device_id,
                source,
                len(deduped),
            )
            return {"success": True, "timezones": deduped, "source": source}
        last_error = stderr or stdout or f"rc={rc}"

    logger.warning("ADB timezone list failed device_id=%s error=%s", device_id, last_error)
    return {"success": False, "timezones": [], "error": last_error}


async def set_timezone_via_adb(device_id: str, timezone_id: str) -> Dict[str, Any]:
    """Set timezone via ADB and verify by immediate read-back.

    Command used for setting:
    - adb -s <device_id> shell settings put global time_zone "<TZ>"

    Verification command:
    - adb -s <device_id> shell settings get global time_zone
    """
    tz = _canonicalize_timezone_input(timezone_id)
    if not tz:
        return {"success": False, "error": "timezone value is required", "requested_timezone": timezone_id}

    logger.info("ADB timezone set requested device_id=%s timezone=%s", device_id, tz)
    rc, stdout, stderr = await _run_adb(
        device_id,
        ["shell", "settings", "put", "global", "time_zone", tz],
    )
    if rc != 0:
        err = stderr or stdout or f"rc={rc}"
        logger.warning("ADB timezone set failed device_id=%s timezone=%s error=%s", device_id, tz, err)
        return {
            "success": False,
            "requested_timezone": tz,
            "error": err,
        }

    readback = await get_timezone_via_adb(device_id)
    persist_readback = await get_persist_timezone_via_adb(device_id)
    observed = str(readback.get("timezone") or "").strip()
    observed_persist = str(persist_readback.get("timezone") or "").strip()
    matched_settings = bool(readback.get("success")) and observed.lower() == tz.lower()
    matched_persist = bool(persist_readback.get("success")) and observed_persist.lower() == tz.lower()
    matched = matched_settings and matched_persist
    logger.info(
        "ADB timezone verification device_id=%s requested=%s observed=%s persist_observed=%s matched=%s",
        device_id,
        tz,
        observed,
        observed_persist,
        matched,
    )
    if not matched:
        mismatch_detail = (
            f"settings={observed or 'unknown'} persist={observed_persist or 'unknown'}"
        )
        return {
            "success": False,
            "requested_timezone": tz,
            "observed_timezone": observed or None,
            "observed_persist_timezone": observed_persist or None,
            "error": str(readback.get("error") or persist_readback.get("error") or f"timezone verification mismatch ({mismatch_detail})").strip() or "timezone verification mismatch",
        }

    return {
        "success": True,
        "requested_timezone": tz,
        "observed_timezone": observed,
        "observed_persist_timezone": observed_persist,
        "verified": True,
    }


async def resolve_timezone_from_supported(
    requested_timezone: str,
    supported_timezones: List[str],
    ai_client: Any = None,
) -> Dict[str, Any]:
    """Resolve a user-requested timezone to an exact supported timezone ID.

    Returns:
        {"success": bool, "resolved_timezone": str|None, "reason": str}
    """
    requested = str(requested_timezone or "").strip()
    if not requested:
        return {"success": False, "resolved_timezone": None, "reason": "empty timezone request"}

    candidates = [str(item or "").strip() for item in (supported_timezones or []) if str(item or "").strip()]
    if not candidates:
        return {"success": False, "resolved_timezone": None, "reason": "no supported timezone list"}

    exact_map = {item.lower(): item for item in candidates}
    lowered = requested.lower()
    if lowered in exact_map:
        return {"success": True, "resolved_timezone": exact_map[lowered], "reason": "exact match"}

    normalized = re.sub(r"\s+", "_", requested)
    normalized = normalized.replace("-", "_")
    if normalized.lower() in exact_map:
        return {"success": True, "resolved_timezone": exact_map[normalized.lower()], "reason": "normalized exact match"}

    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", lowered) if t]
    scored: List[Tuple[float, str]] = []
    for tz in candidates:
        tz_l = tz.lower()
        token_score = sum(1 for t in tokens if t and t in tz_l)
        ratio = difflib.SequenceMatcher(a=lowered, b=tz_l).ratio()
        score = (token_score * 2.0) + ratio
        scored.append((score, tz))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] >= 2.2:
        return {"success": True, "resolved_timezone": scored[0][1], "reason": "token/fuzzy match"}

    if ai_client is not None:
        shortlist = [item for _, item in scored[:80]] or candidates[:80]
        prompt = (
            "Map the user timezone request to exactly one timezone ID from the provided list. "
            "Return only the exact timezone ID string. If none match, return NONE.\n"
            f"User request: {requested}\n"
            "Allowed timezone IDs:\n"
            + "\n".join(shortlist)
        )
        try:
            ai_raw = await ai_client.generate_content(prompt)
            ai_choice = str(ai_raw or "").strip().splitlines()[0].strip()
            if ai_choice.lower() in exact_map:
                return {"success": True, "resolved_timezone": exact_map[ai_choice.lower()], "reason": "ai resolved"}
            if ai_choice in candidates:
                return {"success": True, "resolved_timezone": ai_choice, "reason": "ai resolved"}
        except Exception as exc:
            logger.warning("AI timezone resolution failed: %s", exc)

    return {
        "success": False,
        "resolved_timezone": None,
        "reason": f"could not map requested timezone '{requested}' to a supported timezone ID",
    }


def _normalize_setting_key_for_adb(setting_key: str) -> str:
    key = str(setting_key or "").strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    return key


async def get_setting_via_adb(device_id: str, setting_key: str) -> Dict[str, Any]:
    """Read Android setting using adb shell settings get <namespace> <key>."""
    key = _normalize_setting_key_for_adb(setting_key)
    if not key:
        return {"success": False, "error": "setting key is required", "key": setting_key}

    namespaces = ["global", "system", "secure"]
    last_error = ""
    for namespace in namespaces:
        rc, stdout, stderr = await _run_adb(device_id, ["shell", "settings", "get", namespace, key])
        value = str(stdout or "").strip()
        if rc == 0 and value and value.lower() != "null":
            return {
                "success": True,
                "key": key,
                "namespace": namespace,
                "value": value,
            }
        last_error = stderr or stdout or f"rc={rc}"

    return {
        "success": False,
        "key": key,
        "error": last_error or "setting not found via adb",
    }


async def set_setting_via_adb(device_id: str, setting_key: str, value: Any) -> Dict[str, Any]:
    """Write Android setting using adb shell settings put <namespace> <key> <value> and verify."""
    key = _normalize_setting_key_for_adb(setting_key)
    v = str(value if value is not None else "").strip()
    if not key:
        return {"success": False, "error": "setting key is required", "key": setting_key}
    if not v:
        return {"success": False, "error": "setting value is required", "key": key}

    namespaces = ["global", "system", "secure"]
    last_error = ""
    for namespace in namespaces:
        rc, stdout, stderr = await _run_adb(device_id, ["shell", "settings", "put", namespace, key, v])
        if rc != 0:
            last_error = stderr or stdout or f"rc={rc}"
            continue

        readback = await get_setting_via_adb(device_id, key)
        observed = str(readback.get("value") or "").strip()
        if bool(readback.get("success")) and observed == v:
            return {
                "success": True,
                "key": key,
                "namespace": namespace,
                "requested_value": v,
                "observed_value": observed,
                "verified": True,
            }
        last_error = str(readback.get("error") or "verification mismatch").strip() or "verification mismatch"

    return {
        "success": False,
        "key": key,
        "requested_value": v,
        "error": last_error or "adb setting write failed",
    }
