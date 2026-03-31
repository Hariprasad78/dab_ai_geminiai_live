from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
import subprocess
from typing import Optional

_DEFAULT_DEVICE_SPECS = "cam1|camera|auto:camera1|optional;cam2|camera|auto:camera2|optional;hdmi|hdmi|auto:hdmi|optional"


@dataclass(frozen=True)
class VideoDeviceInfo:
    name: str
    path: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class DeviceConfig:
    device_id: str
    kind: str
    locator: str
    required: bool = True


@dataclass(frozen=True)
class AppSettings:
    devices: list[DeviceConfig]
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"
    open_retries: int = 4
    open_retry_delay_seconds: float = 0.5
    reconnect_interval_seconds: float = 2.0
    frame_timeout_seconds: float = 5.0
    startup_frame_timeout_seconds: float = 8.0
    status_interval_seconds: float = 1.0
    stun_urls: list[str] | None = None


@dataclass(frozen=True)
class ResolvedDevice:
    device_id: str
    kind: str
    locator: str
    required: bool
    device_path: Optional[str]
    resolution_error: Optional[str] = None


def load_settings() -> AppSettings:
    return AppSettings(
        devices=parse_device_specs(os.environ.get("WEBRTC_CAMERA_DEVICES")),
        host=os.environ.get("WEBRTC_CAMERA_HOST", "0.0.0.0"),
        port=int(os.environ.get("WEBRTC_CAMERA_PORT", "8080")),
        log_level=os.environ.get("WEBRTC_CAMERA_LOG_LEVEL", "INFO"),
        width=int(os.environ.get("WEBRTC_CAMERA_WIDTH", "1280")),
        height=int(os.environ.get("WEBRTC_CAMERA_HEIGHT", "720")),
        fps=float(os.environ.get("WEBRTC_CAMERA_FPS", "30.0")),
        fourcc=os.environ.get("WEBRTC_CAMERA_FOURCC", "MJPG"),
        open_retries=int(os.environ.get("WEBRTC_CAMERA_OPEN_RETRIES", "4")),
        open_retry_delay_seconds=float(os.environ.get("WEBRTC_CAMERA_OPEN_RETRY_DELAY_SECONDS", "0.5")),
        reconnect_interval_seconds=float(os.environ.get("WEBRTC_CAMERA_RECONNECT_SECONDS", "2.0")),
        frame_timeout_seconds=float(os.environ.get("WEBRTC_CAMERA_FRAME_TIMEOUT_SECONDS", "5.0")),
        startup_frame_timeout_seconds=float(os.environ.get("WEBRTC_CAMERA_STARTUP_TIMEOUT_SECONDS", "8.0")),
        status_interval_seconds=float(os.environ.get("WEBRTC_CAMERA_STATUS_INTERVAL_SECONDS", "1.0")),
        stun_urls=_parse_csv(os.environ.get("WEBRTC_CAMERA_STUN_URLS")),
    )


def parse_device_specs(raw: Optional[str]) -> list[DeviceConfig]:
    source = (raw or _DEFAULT_DEVICE_SPECS).strip()
    if not source:
        raise RuntimeError("WEBRTC_CAMERA_DEVICES must not be empty")

    specs: list[DeviceConfig] = []
    seen: set[str] = set()
    for chunk in source.split(";"):
        item = chunk.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|") if part.strip()]
        if len(parts) < 3:
            raise RuntimeError(
                "Each WEBRTC_CAMERA_DEVICES entry must look like 'device_id|kind|locator[|optional]'"
            )
        device_id, kind, locator = parts[:3]
        required = True
        if len(parts) >= 4:
            required = parts[3].lower() not in {"optional", "false", "0", "no"}
        if device_id in seen:
            raise RuntimeError(f"Duplicate device_id configured: {device_id}")
        seen.add(device_id)
        specs.append(DeviceConfig(device_id=device_id, kind=kind, locator=locator, required=required))

    if not specs:
        raise RuntimeError("No devices parsed from WEBRTC_CAMERA_DEVICES")
    return specs


def resolve_device(config: DeviceConfig) -> ResolvedDevice:
    locator = str(config.locator or "").strip()
    if not locator:
        return ResolvedDevice(
            device_id=config.device_id,
            kind=config.kind,
            locator=locator,
            required=config.required,
            device_path=None,
            resolution_error="empty locator",
        )

    if locator.startswith("/dev/"):
        path = Path(locator)
        return ResolvedDevice(
            device_id=config.device_id,
            kind=config.kind,
            locator=locator,
            required=config.required,
            device_path=str(path.resolve()) if path.exists() else str(path),
            resolution_error=None if path.exists() else f"device path does not exist: {locator}",
        )

    if locator.startswith("by-id:"):
        target = locator[len("by-id:") :].strip()
        path = Path(target)
        return ResolvedDevice(
            device_id=config.device_id,
            kind=config.kind,
            locator=locator,
            required=config.required,
            device_path=str(path.resolve()) if path.exists() else None,
            resolution_error=None if path.exists() else f"by-id path not found: {target}",
        )

    if locator.startswith("usb:"):
        token = locator[len("usb:") :].strip().lower()
        matches = []
        for candidate in sorted(glob.glob("/dev/v4l/by-id/*")):
            if token in os.path.basename(candidate).lower():
                matches.append(candidate)
        if not matches:
            return ResolvedDevice(
                device_id=config.device_id,
                kind=config.kind,
                locator=locator,
                required=config.required,
                device_path=None,
                resolution_error=f"no /dev/v4l/by-id entry matched usb token: {token}",
            )
        if len(matches) > 1:
            return ResolvedDevice(
                device_id=config.device_id,
                kind=config.kind,
                locator=locator,
                required=config.required,
                device_path=None,
                resolution_error=f"usb token is ambiguous: {token} -> {matches}",
            )
        return ResolvedDevice(
            device_id=config.device_id,
            kind=config.kind,
            locator=locator,
            required=config.required,
            device_path=str(Path(matches[0]).resolve()),
            resolution_error=None,
        )

    if locator.startswith("auto:"):
        token = locator[len("auto:") :].strip().lower()
        match = _resolve_auto_device(token=token, kind=config.kind)
        if match is None:
            candidates = ", ".join(f"{item.name} ({item.path})" for item in discover_video_devices()) or "none"
            return ResolvedDevice(
                device_id=config.device_id,
                kind=config.kind,
                locator=locator,
                required=config.required,
                device_path=None,
                resolution_error=f"auto device not found for {token}; detected candidates: {candidates}",
            )
        return ResolvedDevice(
            device_id=config.device_id,
            kind=config.kind,
            locator=locator,
            required=config.required,
            device_path=match.path,
            resolution_error=None,
        )

    return ResolvedDevice(
        device_id=config.device_id,
        kind=config.kind,
        locator=locator,
        required=config.required,
        device_path=None,
        resolution_error="locator must be /dev/videoN, by-id:/dev/v4l/by-id/..., usb:<substring>, or auto:<token>",
    )


def discover_video_devices() -> list[VideoDeviceInfo]:
    devices = _discover_from_v4l2_ctl()
    if devices:
        return devices
    return _discover_from_sysfs()


def _discover_from_v4l2_ctl() -> list[VideoDeviceInfo]:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return []

    if result.returncode != 0 and not result.stdout.strip():
        return []

    devices: list[VideoDeviceInfo] = []
    current_name: Optional[str] = None
    current_paths: list[str] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current_name and current_paths:
                primary = _select_primary_path(current_paths)
                if primary:
                    devices.append(VideoDeviceInfo(name=current_name, path=primary, paths=tuple(current_paths)))
            current_name = None
            current_paths = []
            continue
        if line.startswith((" ", "\t")):
            candidate = line.strip()
            if candidate.startswith("/dev/video"):
                current_paths.append(candidate)
            continue
        if line.startswith("Cannot open device "):
            continue
        if current_name and current_paths:
            primary = _select_primary_path(current_paths)
            if primary:
                devices.append(VideoDeviceInfo(name=current_name, path=primary, paths=tuple(current_paths)))
        current_name = line[:-1] if line.endswith(":") else line
        current_paths = []

    if current_name and current_paths:
        primary = _select_primary_path(current_paths)
        if primary:
            devices.append(VideoDeviceInfo(name=current_name, path=primary, paths=tuple(current_paths)))
    return devices


def _discover_from_sysfs() -> list[VideoDeviceInfo]:
    devices: list[VideoDeviceInfo] = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*"), key=_sort_video_path):
        name_file = Path(path) / "name"
        if not name_file.exists():
            continue
        name = name_file.read_text(encoding="utf-8", errors="ignore").strip()
        node = f"/dev/{Path(path).name}"
        if not Path(node).exists():
            continue
        devices.append(VideoDeviceInfo(name=name, path=node, paths=(node,)))
    return devices


def _resolve_auto_device(*, token: str, kind: str) -> Optional[VideoDeviceInfo]:
    devices = [item for item in discover_video_devices() if not _is_loopback(item.name)]
    if not devices:
        return None

    hdmi_devices = [item for item in devices if _looks_like_capture_device(item.name)]
    camera_devices = [item for item in devices if item not in hdmi_devices]

    if token in {"hdmi", "capture", "hdmi1", "capture1"}:
        return hdmi_devices[0] if hdmi_devices else None

    match = re.fullmatch(r"(?:camera|cam|webcam)(\d+)", token)
    if match:
        index = max(1, int(match.group(1))) - 1
        return camera_devices[index] if index < len(camera_devices) else None

    if token in {"camera", "cam", "webcam", "camera1", "cam1", "webcam1"}:
        return camera_devices[0] if camera_devices else None

    if kind.lower() == "hdmi":
        return hdmi_devices[0] if hdmi_devices else None
    if kind.lower() == "camera":
        return camera_devices[0] if camera_devices else None
    return devices[0]


def _looks_like_capture_device(name: str) -> bool:
    normalized = " ".join(name.lower().split())
    return any(
        token in normalized
        for token in (
            "capture",
            "hdmi",
            "cam link",
            "elgato",
            "grabber",
            "usb3.0 captur",
            "usb3. 0 captur",
        )
    )


def _is_loopback(name: str) -> bool:
    normalized = " ".join(name.lower().split())
    return "loopback" in normalized or "scrcpy" in normalized


def _select_primary_path(paths: list[str]) -> Optional[str]:
    video_paths = sorted((item for item in paths if item.startswith("/dev/video")), key=_sort_video_path)
    return video_paths[0] if video_paths else None


def _sort_video_path(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)) if match else 10**9, value)


def _parse_csv(raw: Optional[str]) -> list[str] | None:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None
