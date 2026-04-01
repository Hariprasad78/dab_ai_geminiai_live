"""ADB-backed device platform detection helpers."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_HOST_PORT_RE = re.compile(r"^[^:\s]+:(\d{1,5})$")


def _normalize_adb_device_id(device_id: str) -> str:
    value = str(device_id or "").strip()
    if value.lower().startswith("adb:"):
        return value[4:].strip()
    return value


def get_device_connection_type(device_id: str) -> str:
    value = _normalize_adb_device_id(device_id)
    if not value:
        return "unknown"
    matched = _HOST_PORT_RE.match(value)
    if matched is not None:
        try:
            port = int(matched.group(1))
            if 1 <= port <= 65535:
                return "tcp"
        except ValueError:
            return "unknown"
    return "usb"


async def _run_adb(device_id: str, args: List[str], timeout_seconds: float = 12.0) -> Tuple[int, str, str]:
    resolved = _normalize_adb_device_id(device_id)
    if not resolved:
        return 2, "", "missing adb device id"

    cmd = ["adb", "-s", resolved, *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=max(1.0, float(timeout_seconds)))
        stdout = (stdout_b or b"").decode("utf-8", errors="replace").replace("\r", "").strip()
        stderr = (stderr_b or b"").decode("utf-8", errors="replace").replace("\r", "").strip()
        return int(proc.returncode or 0), stdout, stderr
    except asyncio.TimeoutError:
        return 124, "", f"adb command timed out after {int(timeout_seconds)}s"
    except FileNotFoundError:
        return 127, "", "adb binary not found"
    except Exception as exc:
        return 1, "", str(exc)


def _parse_adb_unreachable_detail(rc: int, stdout: str, stderr: str) -> str:
    lowered = f"{stdout} {stderr}".lower()
    if "unauthorized" in lowered:
        return "device unauthorized"
    if "offline" in lowered:
        return "device offline"
    if "not found" in lowered:
        return "device not found"
    if "cannot connect" in lowered or "failed to connect" in lowered:
        return "device unreachable"
    return (stderr or stdout or f"rc={rc}").strip()


def _parse_pm_features(raw: str) -> List[str]:
    features: List[str] = []
    for line in str(raw or "").splitlines():
        value = line.strip()
        if not value:
            continue
        if ":" in value:
            _, tail = value.split(":", 1)
            value = tail.strip()
        if value:
            features.append(value)
    return sorted(set(features))


async def get_device_platform_info(device_id: str) -> Dict[str, Any]:
    resolved = _normalize_adb_device_id(device_id)
    connection_type = get_device_connection_type(resolved)

    result: Dict[str, Any] = {
        "device_id": resolved,
        "connection_type": connection_type,
        "reachable": False,
        "is_android": False,
        "is_android_tv": False,
        "sdk": "",
        "product": "",
        "build_characteristics": "",
        "tv_features": [],
        "evidence": {},
        "error": None,
    }

    if not resolved:
        result["error"] = "missing adb device id"
        logger.warning("Device detection failed: missing device id")
        return result

    state_rc, state_out, state_err = await _run_adb(resolved, ["get-state"], timeout_seconds=8.0)
    if state_rc != 0 or str(state_out or "").strip().lower() != "device":
        detail = _parse_adb_unreachable_detail(state_rc, state_out, state_err)
        result["error"] = detail
        logger.warning(
            "Device detection unreachable: device_id=%s connection=%s error=%s",
            resolved,
            connection_type,
            detail,
        )
        return result

    result["reachable"] = True

    sdk_rc, sdk_out, sdk_err = await _run_adb(resolved, ["shell", "getprop", "ro.build.version.sdk"])
    product_rc, product_out, product_err = await _run_adb(resolved, ["shell", "getprop", "ro.product.device"])
    chars_rc, chars_out, chars_err = await _run_adb(resolved, ["shell", "getprop", "ro.build.characteristics"])
    features_rc, features_out, features_err = await _run_adb(
        resolved,
        ["shell", "pm", "list", "features"],
        timeout_seconds=16.0,
    )

    sdk = str(sdk_out or "").strip()
    product = str(product_out or "").strip()
    build_characteristics = str(chars_out or "").strip().lower()
    features = _parse_pm_features(features_out)

    is_android = bool(re.match(r"^\d+$", sdk or "")) and bool(product)
    tv_feature_markers = {"android.software.leanback", "android.hardware.type.television"}
    tv_features = sorted({f for f in features if f in tv_feature_markers})
    has_tv_characteristic = "tv" in {item.strip() for item in build_characteristics.split(",") if item.strip()}
    is_android_tv = bool(is_android and (tv_features or has_tv_characteristic))

    errors = [
        err for err in [sdk_err if sdk_rc != 0 else "", product_err if product_rc != 0 else "", chars_err if chars_rc != 0 else "", features_err if features_rc != 0 else ""] if str(err or "").strip()
    ]

    result.update(
        {
            "is_android": is_android,
            "is_android_tv": is_android_tv,
            "sdk": sdk,
            "product": product,
            "build_characteristics": build_characteristics,
            "tv_features": tv_features,
            "evidence": {
                "sdk_numeric": bool(re.match(r"^\d+$", sdk or "")),
                "product_non_empty": bool(product),
                "tv_features_found": tv_features,
                "characteristics_contains_tv": has_tv_characteristic,
            },
            "error": "; ".join(errors) if errors else None,
        }
    )

    logger.info(
        "Device detection: device_id=%s connection=%s is_android=%s is_android_tv=%s sdk=%s product=%s tv_features=%s characteristics=%s error=%s",
        resolved,
        connection_type,
        is_android,
        is_android_tv,
        sdk,
        product,
        tv_features,
        build_characteristics,
        result.get("error"),
    )
    return result


async def is_android_device(device_id: str) -> bool:
    info = await get_device_platform_info(device_id)
    return bool(info.get("is_android"))


async def is_android_tv(device_id: str) -> bool:
    info = await get_device_platform_info(device_id)
    return bool(info.get("is_android_tv"))
