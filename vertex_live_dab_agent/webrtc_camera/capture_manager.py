from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Optional

import numpy as np

from .config import AppSettings, DeviceConfig, resolve_device
from .logger import get_logger

STATE_STARTING = "STARTING"
STATE_AVAILABLE = "AVAILABLE"
STATE_DEGRADED = "DEGRADED"
STATE_FAILED = "FAILED"
STATE_STOPPED = "STOPPED"

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaptureSpec:
    device_id: str
    kind: str
    locator: str
    required: bool
    width: int
    height: int
    fps: float
    fourcc: str
    open_retries: int
    open_retry_delay_seconds: float
    reconnect_interval_seconds: float
    frame_timeout_seconds: float
    startup_frame_timeout_seconds: float


@dataclass(frozen=True)
class LatestFrame:
    image: np.ndarray
    captured_at: str
    captured_monotonic: float
    sequence: int


@dataclass(frozen=True)
class StreamSnapshot:
    device_id: str
    kind: str
    locator: str
    required: bool
    device_path: Optional[str]
    state: str
    is_open: bool
    first_frame_received: bool
    frame_available: bool
    frames_captured: int
    dropped_frames: int
    reconnect_attempts: int
    fps: float
    startup_duration_ms: Optional[float]
    last_frame_at: Optional[str]
    last_frame_age_seconds: Optional[float]
    last_error: Optional[str]
    last_warning: Optional[str]
    initialization_complete: bool

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "locator": self.locator,
            "required": self.required,
            "device_path": self.device_path,
            "state": self.state,
            "is_open": self.is_open,
            "first_frame_received": self.first_frame_received,
            "frame_available": self.frame_available,
            "frames_captured": self.frames_captured,
            "dropped_frames": self.dropped_frames,
            "reconnect_attempts": self.reconnect_attempts,
            "fps": self.fps,
            "startup_duration_ms": self.startup_duration_ms,
            "last_frame_at": self.last_frame_at,
            "last_frame_age_seconds": self.last_frame_age_seconds,
            "last_error": self.last_error,
            "last_warning": self.last_warning,
            "initialization_complete": self.initialization_complete,
        }


class CaptureManager:
    def __init__(self, spec: CaptureSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self._path_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[Any] = None
        self._cv2: Optional[Any] = None
        self._device_path: Optional[str] = None
        self._state = STATE_STARTING
        self._is_open = False
        self._first_frame_received = False
        self._initialization_complete = False
        self._last_error: Optional[str] = None
        self._last_warning: Optional[str] = None
        self._latest_frame: Optional[LatestFrame] = None
        self._frames_captured = 0
        self._dropped_frames = 0
        self._reconnect_attempts = 0
        self._fps = 0.0
        self._startup_started_monotonic = time.monotonic()
        self._startup_duration_ms: Optional[float] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name=f"webrtc-capture-{self.spec.device_id}", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
        self._release_capture()
        with self._lock:
            self._state = STATE_STOPPED
            self._is_open = False
            self._initialization_complete = True

    def wait_until_initialized(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.1, float(timeout))
        while time.monotonic() < deadline and not self._stop_event.is_set():
            if self.snapshot().initialization_complete:
                return True
            time.sleep(0.05)
        return self.snapshot().initialization_complete

    def latest_frame(self) -> Optional[LatestFrame]:
        with self._lock:
            if self._latest_frame is None:
                return None
            latest = self._latest_frame
            return LatestFrame(
                image=latest.image.copy(),
                captured_at=latest.captured_at,
                captured_monotonic=latest.captured_monotonic,
                sequence=latest.sequence,
            )

    def snapshot(self) -> StreamSnapshot:
        with self._lock:
            latest = self._latest_frame
            return StreamSnapshot(
                device_id=self.spec.device_id,
                kind=self.spec.kind,
                locator=self.spec.locator,
                required=self.spec.required,
                device_path=self._device_path,
                state=self._state,
                is_open=self._is_open,
                first_frame_received=self._first_frame_received,
                frame_available=latest is not None,
                frames_captured=self._frames_captured,
                dropped_frames=self._dropped_frames,
                reconnect_attempts=self._reconnect_attempts,
                fps=self._fps,
                startup_duration_ms=self._startup_duration_ms,
                last_frame_at=latest.captured_at if latest else None,
                last_frame_age_seconds=(max(0.0, time.monotonic() - latest.captured_monotonic) if latest else None),
                last_error=self._last_error,
                last_warning=self._last_warning,
                initialization_complete=self._initialization_complete,
            )

    def _run_loop(self) -> None:
        self._log_info("capture worker starting", event="capture_worker_start")
        while not self._stop_event.is_set():
            resolved = resolve_device(self._as_device_config())
            with self._path_lock:
                self._device_path = resolved.device_path
            if resolved.device_path is None:
                self._mark_failure(resolved.resolution_error or f"unable to resolve {self.spec.locator}")
                self._note_reconnect_attempt("waiting for device path to resolve", set_degraded=False)
                self._sleep(self.spec.reconnect_interval_seconds)
                continue
            capture = self._open_capture(resolved.device_path)
            if capture is None:
                self._note_reconnect_attempt(f"retrying open for {resolved.device_path}", set_degraded=False)
                self._sleep(self.spec.reconnect_interval_seconds)
                continue
            try:
                self._stream_frames(capture)
            finally:
                self._release_capture()
                if not self._stop_event.is_set():
                    self._note_reconnect_attempt("restarting device stream", set_degraded=True)
                    self._sleep(self.spec.reconnect_interval_seconds)
        self._log_info("capture worker stopped", event="capture_worker_stop")

    def _open_capture(self, device_path: str) -> Optional[Any]:
        cv2 = self._load_cv2()
        backend = getattr(cv2, "CAP_V4L2", None)
        max_attempts = max(1, int(self.spec.open_retries))
        start = time.monotonic()
        for attempt in range(1, max_attempts + 1):
            self._set_state(STATE_STARTING, is_open=False)
            self._log_info(
                "device open attempt",
                event="device_open_attempt",
                device_path=device_path,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            capture = cv2.VideoCapture(device_path, backend) if backend is not None else cv2.VideoCapture(device_path)
            if capture and capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.spec.width))
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.spec.height))
                capture.set(cv2.CAP_PROP_FPS, float(self.spec.fps))
                if len(self.spec.fourcc) == 4:
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec.fourcc))
                with self._lock:
                    self._capture = capture
                    self._is_open = True
                    self._last_error = None
                    self._last_warning = None
                self._log_info(
                    "device open success",
                    event="device_open_success",
                    device_path=device_path,
                    open_elapsed_ms=round((time.monotonic() - start) * 1000.0, 2),
                )
                return capture
            if capture is not None:
                with _suppress_release(capture):
                    capture.release()
            self._log_warning(
                "device open failed",
                event="device_open_failure",
                device_path=device_path,
                attempt=attempt,
            )
            if attempt < max_attempts:
                self._sleep(self.spec.open_retry_delay_seconds)

        self._mark_failure(f"unable to open {device_path}")
        return None

    def _stream_frames(self, capture: Any) -> None:
        frame_counter = 0
        interval_started = time.monotonic()
        last_frame_monotonic = interval_started
        while not self._stop_event.is_set():
            ok, frame = capture.read()
            now = time.monotonic()
            if not ok or frame is None:
                self._increment_dropped_frame("capture read returned no frame")
                if now - last_frame_monotonic >= max(0.5, float(self.spec.frame_timeout_seconds)):
                    self._mark_degraded(f"frame timeout on {self._device_path}")
                    self._log_warning(
                        "frame timeout detected",
                        event="frame_timeout",
                        device_path=self._device_path,
                        timeout_seconds=self.spec.frame_timeout_seconds,
                    )
                    return
                self._sleep(0.03)
                continue

            if not self._first_frame_received:
                self._startup_duration_ms = round((now - self._startup_started_monotonic) * 1000.0, 2)
                self._log_info(
                    "first frame received",
                    event="first_frame_success",
                    device_path=self._device_path,
                    startup_duration_ms=self._startup_duration_ms,
                )

            frame_counter += 1
            elapsed = max(now - interval_started, 1e-6)
            fps = frame_counter / elapsed
            last_frame_monotonic = now
            self._store_frame(frame=frame, fps=fps, captured_monotonic=now)
            if elapsed >= 5.0:
                self._log_info(
                    "capture fps sample",
                    event="capture_fps",
                    device_path=self._device_path,
                    fps=round(fps, 2),
                    frames=frame_counter,
                    dropped_frames=self.snapshot().dropped_frames,
                )
                frame_counter = 0
                interval_started = now

    def _store_frame(self, *, frame: np.ndarray, fps: float, captured_monotonic: float) -> None:
        latest = LatestFrame(
            image=frame.copy(),
            captured_at=_utc_now_iso(),
            captured_monotonic=captured_monotonic,
            sequence=self._frames_captured + 1,
        )
        with self._lock:
            self._latest_frame = latest
            self._frames_captured += 1
            self._fps = float(fps)
            self._first_frame_received = True
            self._initialization_complete = True
            self._state = STATE_AVAILABLE
            self._is_open = True
            self._last_error = None
            self._last_warning = None
        self._ready_event.set()

    def _increment_dropped_frame(self, reason: str) -> None:
        with self._lock:
            self._dropped_frames += 1
            self._last_warning = reason
            self._initialization_complete = True
        if self._dropped_frames == 1 or self._dropped_frames % 30 == 0:
            self._log_warning(
                "dropped frame",
                event="dropped_frame",
                dropped_frames=self._dropped_frames,
                reason=reason,
            )

    def _note_reconnect_attempt(self, reason: str, *, set_degraded: bool) -> None:
        with self._lock:
            self._reconnect_attempts += 1
            if set_degraded:
                self._state = STATE_DEGRADED
                self._is_open = False
            self._last_warning = reason
            self._initialization_complete = True
        self._log_warning(
            "reconnect scheduled",
            event="reconnect_attempt",
            reconnect_attempts=self._reconnect_attempts,
            reason=reason,
        )

    def _mark_failure(self, error: str) -> None:
        with self._lock:
            self._state = STATE_FAILED if self.spec.required else STATE_DEGRADED
            self._is_open = False
            self._last_error = error
            self._initialization_complete = True
        self._log_warning("capture failure", event="capture_failure", error=error)

    def _mark_degraded(self, warning: str) -> None:
        with self._lock:
            self._state = STATE_DEGRADED
            self._is_open = False
            self._last_warning = warning
            self._initialization_complete = True

    def _set_state(self, state: str, *, is_open: bool) -> None:
        with self._lock:
            self._state = state
            self._is_open = is_open

    def _release_capture(self) -> None:
        capture = None
        with self._lock:
            capture = self._capture
            self._capture = None
            self._is_open = False
        if capture is not None:
            try:
                capture.release()
            except Exception:
                self._log_warning("capture release failed", event="capture_release_failure")

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(timeout=max(0.0, float(seconds)))

    def _load_cv2(self) -> Any:
        if self._cv2 is not None:
            return self._cv2
        import os
        import cv2  # type: ignore

        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
        try:
            if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            elif hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        except Exception:
            pass
        self._cv2 = cv2
        return cv2

    def _as_device_config(self) -> DeviceConfig:
        return DeviceConfig(
            device_id=self.spec.device_id,
            kind=self.spec.kind,
            locator=self.spec.locator,
            required=self.spec.required,
        )

    def _log_info(self, message: str, **extra: Any) -> None:
        logger.info(message, extra={"device_id": self.spec.device_id, "kind": self.spec.kind, **extra})

    def _log_warning(self, message: str, **extra: Any) -> None:
        logger.warning(message, extra={"device_id": self.spec.device_id, "kind": self.spec.kind, **extra})


class CaptureRegistry:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._managers: dict[str, CaptureManager] = {
            spec.device_id: CaptureManager(
                CaptureSpec(
                    device_id=spec.device_id,
                    kind=spec.kind,
                    locator=spec.locator,
                    required=spec.required,
                    width=settings.width,
                    height=settings.height,
                    fps=settings.fps,
                    fourcc=settings.fourcc,
                    open_retries=settings.open_retries,
                    open_retry_delay_seconds=settings.open_retry_delay_seconds,
                    reconnect_interval_seconds=settings.reconnect_interval_seconds,
                    frame_timeout_seconds=settings.frame_timeout_seconds,
                    startup_frame_timeout_seconds=settings.startup_frame_timeout_seconds,
                )
            )
            for spec in settings.devices
        }

    def start(self) -> None:
        for manager in self._managers.values():
            manager.start()
        for manager in self._managers.values():
            manager.wait_until_initialized(self.settings.startup_frame_timeout_seconds)

    def stop(self) -> None:
        for manager in self._managers.values():
            manager.stop()

    def get_manager(self, device_id: str) -> CaptureManager:
        manager = self._managers.get(device_id)
        if manager is None:
            raise KeyError(device_id)
        return manager

    def list_devices(self) -> list[dict]:
        return [manager.snapshot().as_dict() for manager in self._managers.values()]

    def status(self) -> dict:
        devices = self.list_devices()
        required = [item for item in devices if item.get("required")]
        failed_required = [item for item in required if item.get("state") == STATE_FAILED]
        available_required = [item for item in required if item.get("state") == STATE_AVAILABLE]
        if failed_required:
            overall = "failed"
        elif len(available_required) == len(required):
            overall = "ok"
        else:
            overall = "degraded"
        return {
            "status": overall,
            "device_count": len(devices),
            "available_device_count": len([item for item in devices if item.get("state") == STATE_AVAILABLE]),
            "failed_required_count": len(failed_required),
            "devices": devices,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _suppress_release:
    def __init__(self, capture: Any) -> None:
        self.capture = capture

    def __enter__(self) -> "_suppress_release":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return True
