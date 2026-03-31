from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Optional

from .logger import get_logger
from .stream_state import STATE_AVAILABLE, STATE_DEGRADED, STATE_FAILED, StreamState

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaptureSpec:
    device_id: str
    locator: str
    device_path: str
    kind: str
    required: bool = True
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    jpeg_quality: int = 85
    fourcc: str = "MJPG"
    reconnect_interval_seconds: float = 2.0
    startup_frame_timeout_seconds: float = 8.0
    first_frame_retries: int = 20
    fps_log_interval_seconds: float = 5.0
    frame_failure_threshold: int = 15
    open_retries: int = 4
    open_retry_delay_seconds: float = 0.6
    stale_frame_threshold_seconds: float = 5.0


class CaptureManager:
    def __init__(self, spec: CaptureSpec) -> None:
        self.spec = spec
        self.state = StreamState(
            device_id=spec.device_id,
            locator=spec.locator,
            device_path=spec.device_path,
            kind=spec.kind,
            required=spec.required,
        )
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture_lock = threading.Lock()
        self._capture: Optional[Any] = None
        self._cv2: Optional[Any] = None
        self._path_lock = threading.Lock()
        self._device_path = str(spec.device_path)
        self._reconnect_requested = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"capture-{self.spec.device_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._reconnect_requested.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
        self._release_capture()
        self.state.mark_stopped()

    def wait_until_ready(self, timeout: float) -> bool:
        return self._ready_event.wait(timeout=max(0.1, float(timeout)))

    def wait_until_initialized(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.1, float(timeout))
        while time.monotonic() < deadline and not self._stop_event.is_set():
            snapshot = self.snapshot()
            if snapshot.get("initialization_complete"):
                return True
            time.sleep(0.05)
        return self.snapshot().get("initialization_complete", False)

    def latest_frame(self) -> Optional[bytes]:
        return self.state.latest_frame()

    def snapshot(self) -> dict:
        return self.state.snapshot().as_dict()

    def set_device_path(self, device_path: str) -> None:
        with self._path_lock:
            self._device_path = str(device_path)
        self.state.set_device_path(str(device_path))

    def get_device_path(self) -> str:
        with self._path_lock:
            return self._device_path

    def request_reconnect(self, reason: str) -> None:
        self.state.mark_reconnecting(reason)
        self._reconnect_requested.set()

    def startup_state(self) -> str:
        snapshot = self.snapshot()
        if snapshot.get("state") == STATE_AVAILABLE:
            return STATE_AVAILABLE
        if snapshot.get("required"):
            return STATE_FAILED
        return STATE_DEGRADED

    def _run_loop(self) -> None:
        self.state.mark_validating(f"initializing {self.get_device_path()}")
        while not self._stop_event.is_set():
            try:
                capture = self._open_capture()
                if capture is None:
                    self.state.mark_initialization_complete()
                    self._sleep_with_stop(self.spec.reconnect_interval_seconds)
                    continue
                if not self._prime_first_frame(capture):
                    self._release_capture()
                    self.state.mark_initialization_complete()
                    self._sleep_with_stop(self.spec.reconnect_interval_seconds)
                    continue
                self.state.mark_initialization_complete()
                self._stream_frames(capture)
            except Exception as exc:
                self.state.mark_exception(str(exc))
                logger.exception(
                    "capture manager exception",
                    extra={
                        "event": "capture_exception",
                        "device_id": self.spec.device_id,
                        "device_path": self.spec.device_path,
                        "kind": self.spec.kind,
                    },
                )
                self._release_capture()
                self.state.mark_initialization_complete()
                self._sleep_with_stop(self.spec.reconnect_interval_seconds)
        self._release_capture()

    def _open_capture(self) -> Optional[Any]:
        cv2 = _load_cv2()
        self._cv2 = cv2
        device_path = self.get_device_path()
        backend = getattr(cv2, "CAP_V4L2", None)
        max_attempts = max(1, int(self.spec.open_retries))
        for attempt in range(1, max_attempts + 1):
            self.state.mark_open_attempt(attempt, max_attempts, f"opening {device_path}")
            logger.info(
                "device open attempt",
                extra={
                    "event": "device_open_attempt",
                    "device_id": self.spec.device_id,
                    "device_path": device_path,
                    "kind": self.spec.kind,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                },
            )
            capture = cv2.VideoCapture(device_path, backend) if backend is not None else cv2.VideoCapture(device_path)
            if capture and capture.isOpened():
                break
            with _suppress_release(capture):
                if capture is not None:
                    capture.release()
            capture = None
            if attempt < max_attempts:
                time.sleep(max(0.05, float(self.spec.open_retry_delay_seconds)))
        else:
            self.state.mark_open_failure(f"unable to open {device_path}")
            logger.error(
                "device open failed",
                extra={
                    "event": "device_open_failure",
                    "device_id": self.spec.device_id,
                    "device_path": device_path,
                    "kind": self.spec.kind,
                    "open_retries": max_attempts,
                },
            )
            return None

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.spec.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.spec.height))
        capture.set(cv2.CAP_PROP_FPS, float(self.spec.fps))
        if len(self.spec.fourcc) == 4:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec.fourcc))

        with self._capture_lock:
            self._capture = capture
        self.state.mark_open_success()
        logger.info(
            "device opened",
            extra={
                "event": "device_open_success",
                "device_id": self.spec.device_id,
                "device_path": device_path,
                "kind": self.spec.kind,
            },
        )
        return capture

    def _prime_first_frame(self, capture: Any) -> bool:
        deadline = time.monotonic() + self.spec.startup_frame_timeout_seconds
        attempts = 0
        while not self._stop_event.is_set() and time.monotonic() < deadline and attempts < self.spec.first_frame_retries:
            ok, frame = capture.read()
            attempts += 1
            if ok and frame is not None:
                encoded = self._encode_jpeg(frame)
                if encoded is None:
                    continue
                self.state.mark_frame(encoded, 0.0)
                self._ready_event.set()
                logger.info(
                    "first frame received",
                    extra={
                        "event": "first_frame_success",
                        "device_id": self.spec.device_id,
                        "device_path": self.get_device_path(),
                        "kind": self.spec.kind,
                        "attempts": attempts,
                    },
                )
                return True
            time.sleep(0.1)

        error = f"first frame not received from {self.get_device_path()}"
        self.state.mark_first_frame_failure(error)
        logger.error(
            "first frame failed",
            extra={
                "event": "first_frame_failure",
                "device_id": self.spec.device_id,
                "device_path": self.get_device_path(),
                "kind": self.spec.kind,
                "attempts": attempts,
            },
        )
        return False

    def _stream_frames(self, capture: Any) -> None:
        frame_failures = 0
        frame_counter = 0
        interval_started = time.monotonic()
        last_fps_log = interval_started
        while not self._stop_event.is_set():
            if self._reconnect_requested.is_set():
                self._reconnect_requested.clear()
                self._release_capture()
                return
            ok, frame = capture.read()
            if not ok or frame is None:
                frame_failures += 1
                if frame_failures >= self.spec.frame_failure_threshold:
                    error = f"frame reads stalled for {self.get_device_path()}"
                    self.state.mark_reconnecting(error)
                    logger.warning(
                        "reconnecting after frame failures",
                        extra={
                            "event": "reconnect_attempt",
                            "device_id": self.spec.device_id,
                            "device_path": self.get_device_path(),
                            "kind": self.spec.kind,
                            "frame_failures": frame_failures,
                        },
                    )
                    self._release_capture()
                    return
                time.sleep(0.05)
                continue

            frame_failures = 0
            encoded = self._encode_jpeg(frame)
            if encoded is None:
                continue
            frame_counter += 1
            now = time.monotonic()
            elapsed = max(now - interval_started, 1e-6)
            fps = frame_counter / elapsed
            self.state.mark_frame(encoded, fps)
            if now - last_fps_log >= self.spec.fps_log_interval_seconds:
                logger.info(
                    "stream fps",
                    extra={
                        "event": "stream_fps",
                        "device_id": self.spec.device_id,
                        "device_path": self.spec.device_path,
                        "kind": self.spec.kind,
                        "fps": round(fps, 2),
                        "frames_captured": frame_counter,
                    },
                )
                frame_counter = 0
                interval_started = now
                last_fps_log = now

    def _encode_jpeg(self, frame: Any) -> Optional[bytes]:
        cv2 = self._cv2 or _load_cv2()
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, max(30, min(95, int(self.spec.jpeg_quality)))],
        )
        if not ok:
            return None
        return encoded.tobytes()

    def _release_capture(self) -> None:
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            logger.info(
                "releasing capture handle",
                extra={
                    "event": "device_release",
                    "device_id": self.spec.device_id,
                    "device_path": self.get_device_path(),
                    "kind": self.spec.kind,
                },
            )
            try:
                capture.release()
            except Exception:
                logger.exception(
                    "capture release failed",
                    extra={
                        "event": "device_release_failure",
                        "device_id": self.spec.device_id,
                        "device_path": self.get_device_path(),
                        "kind": self.spec.kind,
                    },
                )

    def monitor(self, *, resolved_device_path: Optional[str], stale_after_seconds: float) -> None:
        snapshot = self.snapshot()
        current_path = str(snapshot.get("device_path") or self.get_device_path())
        if resolved_device_path and resolved_device_path != current_path:
            logger.info(
                "device path remapped",
                extra={
                    "event": "device_path_remap",
                    "device_id": self.spec.device_id,
                    "old_device_path": current_path,
                    "new_device_path": resolved_device_path,
                    "locator": self.spec.locator,
                },
            )
            self.set_device_path(resolved_device_path)
            self.request_reconnect(f"device path changed to {resolved_device_path}")
            return
        if not resolved_device_path:
            self.state.mark_degraded(f"device locator unresolved: {self.spec.locator}")
            return
        if not Path(resolved_device_path).exists():
            self.request_reconnect(f"device path disappeared: {resolved_device_path}")
            return
        last_frame_at = snapshot.get("last_frame_at")
        if snapshot.get("state") == STATE_AVAILABLE and last_frame_at:
            try:
                last_frame_ts = _parse_iso(last_frame_at)
                age = time.time() - last_frame_ts
            except Exception:
                age = 0.0
            if age > max(0.5, float(stale_after_seconds)):
                self.request_reconnect(f"stale frame detected ({age:.2f}s)")

    def _sleep_with_stop(self, seconds: float) -> None:
        self._stop_event.wait(timeout=max(0.0, float(seconds)))


def _load_cv2() -> Any:
    import os

    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    import cv2  # type: ignore

    try:
        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        elif hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        pass
    return cv2


class _suppress_release:
    def __init__(self, capture: Any) -> None:
        self.capture = capture

    def __enter__(self) -> "_suppress_release":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return True


def _parse_iso(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(str(value)).timestamp()
