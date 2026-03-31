from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import threading
import time
from typing import Callable, Optional

from .capture_manager import CaptureManager, CaptureSpec
from .logger import get_logger
from .stream_state import STATE_AVAILABLE, STATE_DEGRADED, STATE_FAILED

logger = get_logger(__name__)
_DEFAULT_DEVICE_SPECS = "cam1|camera|by-id:/dev/v4l/by-id/cam1;cam2|camera|by-id:/dev/v4l/by-id/cam2;hdmi|hdmi|usb:hdmi"


@dataclass(frozen=True)
class DeviceSpec:
    device_id: str
    kind: str
    locator: str
    required: bool = True


@dataclass(frozen=True)
class ResolvedDeviceSpec:
    device_id: str
    kind: str
    locator: str
    required: bool
    device_path: Optional[str]
    resolution_error: Optional[str] = None


class DeviceRegistry:
    def __init__(self, specs: list[CaptureSpec], *, manager_factory: Callable[[CaptureSpec], CaptureManager] = CaptureManager) -> None:
        if not specs:
            raise RuntimeError("At least one video source must be configured")
        self._specs = list(specs)
        self._manager_factory = manager_factory
        self._managers = {spec.device_id: manager_factory(spec) for spec in self._specs}
        self._monitor_stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._startup_report: list[dict] = []

    @classmethod
    def from_env(cls) -> "DeviceRegistry":
        specs = [
            CaptureSpec(
                device_id=item.device_id,
                kind=item.kind,
                locator=item.locator,
                device_path=resolve_device_spec(item).device_path or item.locator,
                required=item.required,
                width=int(os.environ.get("MULTI_CAMERA_WIDTH", "1280")),
                height=int(os.environ.get("MULTI_CAMERA_HEIGHT", "720")),
                fps=float(os.environ.get("MULTI_CAMERA_FPS", "30.0")),
                jpeg_quality=int(os.environ.get("MULTI_CAMERA_JPEG_QUALITY", "85")),
                fourcc=os.environ.get("MULTI_CAMERA_FOURCC", "MJPG"),
                reconnect_interval_seconds=float(os.environ.get("MULTI_CAMERA_RECONNECT_SECONDS", "2.0")),
                startup_frame_timeout_seconds=float(os.environ.get("MULTI_CAMERA_STARTUP_TIMEOUT_SECONDS", "8.0")),
                open_retries=int(os.environ.get("MULTI_CAMERA_OPEN_RETRIES", "4")),
                open_retry_delay_seconds=float(os.environ.get("MULTI_CAMERA_OPEN_RETRY_DELAY_SECONDS", "0.6")),
                stale_frame_threshold_seconds=float(os.environ.get("MULTI_CAMERA_STALE_FRAME_SECONDS", "5.0")),
            )
            for item in parse_device_specs(os.environ.get("MULTI_CAMERA_DEVICES"))
        ]
        return cls(specs)

    def start(self) -> None:
        logger.info("starting device registry", extra={"event": "registry_start", "device_count": len(self._specs)})
        startup_results = self.validate_configured_devices()
        for manager in self._managers.values():
            manager.start()
        self._startup_report = list(startup_results)
        for spec in self._specs:
            manager = self._managers[spec.device_id]
            manager.wait_until_initialized(spec.startup_frame_timeout_seconds)
        self._start_monitoring()
        logger.info("device registry startup complete", extra={"event": "registry_start_complete", "devices": self.list_devices()})

    def stop(self) -> None:
        self._monitor_stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        for manager in self._managers.values():
            manager.stop()
        logger.info("device registry stopped", extra={"event": "registry_stop"})

    def validate_configured_devices(self) -> list[dict]:
        results: list[dict] = []
        for spec in self._specs:
            resolved = resolve_capture_spec(spec)
            manager = self._managers[spec.device_id]
            if resolved.device_path:
                manager.set_device_path(resolved.device_path)
                manager.state.mark_validating(f"resolved {resolved.locator} -> {resolved.device_path}")
                state = STATE_DEGRADED
            else:
                error = resolved.resolution_error or f"unable to resolve locator {resolved.locator}"
                if resolved.required:
                    manager.state.mark_open_failure(error)
                    state = STATE_FAILED
                else:
                    manager.state.mark_degraded(error)
                    state = STATE_DEGRADED
            logger.info(
                "startup device validation",
                extra={
                    "event": "startup_device_validation",
                    "device_id": spec.device_id,
                    "kind": spec.kind,
                    "locator": spec.locator,
                    "resolved_device_path": resolved.device_path,
                    "required": spec.required,
                    "state": state,
                    "resolution_error": resolved.resolution_error,
                },
            )
            results.append(
                {
                    "device_id": spec.device_id,
                    "kind": spec.kind,
                    "locator": spec.locator,
                    "device_path": resolved.device_path,
                    "required": spec.required,
                    "state": state,
                    "resolution_error": resolved.resolution_error,
                }
            )
        return results

    def list_devices(self) -> list[dict]:
        return [manager.snapshot() for manager in self._managers.values()]

    def stream_status(self) -> dict:
        devices = self.list_devices()
        required = [item for item in devices if item.get("required")]
        available_required = [item for item in required if item.get("state") == STATE_AVAILABLE]
        failed_required = [item for item in required if item.get("state") == STATE_FAILED]
        if failed_required:
            overall = "failed"
        elif len(available_required) == len(required):
            overall = "ok"
        else:
            overall = "degraded"
        return {
            "status": overall,
            "device_count": len(devices),
            "required_device_count": len(required),
            "available_device_count": len([item for item in devices if item.get("state") == STATE_AVAILABLE]),
            "available_required_count": len(available_required),
            "failed_required_count": len(failed_required),
            "startup_report": list(self._startup_report),
            "devices": devices,
        }

    def health(self) -> dict:
        status = self.stream_status()
        return {
            "status": status["status"],
            "required_device_count": status["required_device_count"],
            "available_required_count": status["available_required_count"],
            "failed_required_count": status["failed_required_count"],
            "devices": [
                {
                    "device_id": item.get("device_id"),
                    "state": item.get("state"),
                    "frame_available": item.get("frame_available"),
                    "last_frame_at": item.get("last_frame_at"),
                    "last_frame_age_seconds": item.get("last_frame_age_seconds"),
                    "last_error": item.get("last_error"),
                    "device_path": item.get("device_path"),
                    "locator": item.get("locator"),
                }
                for item in status["devices"]
            ],
        }

    def latest_frame(self, device_id: str) -> Optional[bytes]:
        manager = self._managers.get(device_id)
        if manager is None:
            raise KeyError(device_id)
        return manager.latest_frame()

    def _start_monitoring(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop_event.clear()
        interval = max(0.5, float(os.environ.get("MULTI_CAMERA_MONITOR_INTERVAL_SECONDS", "2.0")))
        stale_after = max(1.0, float(os.environ.get("MULTI_CAMERA_STALE_FRAME_SECONDS", "5.0")))
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval, stale_after),
            name="multi-camera-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self, interval: float, stale_after: float) -> None:
        logger.info("starting background monitor", extra={"event": "monitor_start", "interval_seconds": interval})
        while not self._monitor_stop_event.wait(interval):
            for spec in self._specs:
                manager = self._managers[spec.device_id]
                resolved = resolve_capture_spec(spec)
                manager.monitor(resolved_device_path=resolved.device_path, stale_after_seconds=stale_after)
        logger.info("background monitor stopped", extra={"event": "monitor_stop"})


def parse_device_specs(raw: Optional[str]) -> list[DeviceSpec]:
    source = (raw or _DEFAULT_DEVICE_SPECS).strip()
    if not source:
        raise RuntimeError("MULTI_CAMERA_DEVICES must not be empty")

    specs: list[DeviceSpec] = []
    seen: set[str] = set()
    for chunk in source.split(";"):
        item = chunk.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|") if part.strip()]
        if len(parts) < 3:
            raise RuntimeError(
                "Each MULTI_CAMERA_DEVICES entry must look like 'device_id|kind|locator[|optional]'"
            )
        device_id, kind, locator = parts[:3]
        required = True
        if len(parts) >= 4:
            required = parts[3].lower() not in {"optional", "false", "0", "no"}
        if device_id in seen:
            raise RuntimeError(f"Duplicate device_id configured: {device_id}")
        seen.add(device_id)
        specs.append(DeviceSpec(device_id=device_id, kind=kind, locator=locator, required=required))

    if not specs:
        raise RuntimeError("No devices parsed from MULTI_CAMERA_DEVICES")
    logger.info("loaded device configuration", extra={"event": "device_config_loaded", "devices": [spec.__dict__ for spec in specs]})
    return specs


def resolve_capture_spec(spec: CaptureSpec) -> ResolvedDeviceSpec:
    return resolve_device_spec(DeviceSpec(device_id=spec.device_id, kind=spec.kind, locator=spec.locator, required=spec.required))


def resolve_device_spec(spec: DeviceSpec) -> ResolvedDeviceSpec:
    locator = str(spec.locator or "").strip()
    if not locator:
        return ResolvedDeviceSpec(
            device_id=spec.device_id,
            kind=spec.kind,
            locator=locator,
            required=spec.required,
            device_path=None,
            resolution_error="empty device locator",
        )

    if locator.startswith("/dev/"):
        path = Path(locator)
        return ResolvedDeviceSpec(
            device_id=spec.device_id,
            kind=spec.kind,
            locator=locator,
            required=spec.required,
            device_path=str(path.resolve()) if path.exists() else str(path),
            resolution_error=None if path.exists() else f"device path does not exist: {locator}",
        )

    if locator.startswith("by-id:"):
        target = locator[len("by-id:") :].strip()
        path = Path(target)
        return ResolvedDeviceSpec(
            device_id=spec.device_id,
            kind=spec.kind,
            locator=locator,
            required=spec.required,
            device_path=str(path.resolve()) if path.exists() else None,
            resolution_error=None if path.exists() else f"by-id path not found: {target}",
        )

    if locator.startswith("usb:"):
        token = locator[len("usb:") :].strip().lower()
        matches = []
        for candidate in sorted(glob.glob("/dev/v4l/by-id/*")):
            name = os.path.basename(candidate).lower()
            if token in name:
                matches.append(candidate)
        if not matches:
            return ResolvedDeviceSpec(
                device_id=spec.device_id,
                kind=spec.kind,
                locator=locator,
                required=spec.required,
                device_path=None,
                resolution_error=f"no /dev/v4l/by-id entry matched usb token: {token}",
            )
        if len(matches) > 1:
            return ResolvedDeviceSpec(
                device_id=spec.device_id,
                kind=spec.kind,
                locator=locator,
                required=spec.required,
                device_path=None,
                resolution_error=f"usb token is ambiguous: {token} -> {matches}",
            )
        return ResolvedDeviceSpec(
            device_id=spec.device_id,
            kind=spec.kind,
            locator=locator,
            required=spec.required,
            device_path=str(Path(matches[0]).resolve()),
            resolution_error=None,
        )

    return ResolvedDeviceSpec(
        device_id=spec.device_id,
        kind=spec.kind,
        locator=locator,
        required=spec.required,
        device_path=None,
        resolution_error="locator must be /dev/videoN, by-id:/dev/v4l/by-id/..., or usb:<substring>",
    )
