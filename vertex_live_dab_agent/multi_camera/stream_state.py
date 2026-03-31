from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Optional


STATE_AVAILABLE = "AVAILABLE"
STATE_DEGRADED = "DEGRADED"
STATE_FAILED = "FAILED"
STATE_STOPPED = "STOPPED"


@dataclass(frozen=True)
class StreamSnapshot:
    device_id: str
    locator: str
    device_path: str
    kind: str
    required: bool
    state: str
    is_open: bool
    first_frame_received: bool
    frames_captured: int
    fps: float
    reconnect_attempts: int
    last_frame_at: Optional[str]
    last_frame_age_seconds: Optional[float]
    started_at: str
    last_state_change_at: str
    open_success_at: Optional[str]
    first_frame_at: Optional[str]
    last_error: Optional[str]
    initialization_complete: bool
    monitor_message: Optional[str]
    frame_available: bool

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "locator": self.locator,
            "device_path": self.device_path,
            "kind": self.kind,
            "required": self.required,
            "state": self.state,
            "is_open": self.is_open,
            "first_frame_received": self.first_frame_received,
            "frames_captured": self.frames_captured,
            "fps": self.fps,
            "reconnect_attempts": self.reconnect_attempts,
            "last_frame_at": self.last_frame_at,
            "last_frame_age_seconds": self.last_frame_age_seconds,
            "started_at": self.started_at,
            "last_state_change_at": self.last_state_change_at,
            "open_success_at": self.open_success_at,
            "first_frame_at": self.first_frame_at,
            "last_error": self.last_error,
            "initialization_complete": self.initialization_complete,
            "monitor_message": self.monitor_message,
            "frame_available": self.frame_available,
        }


class StreamState:
    def __init__(self, *, device_id: str, locator: str, device_path: str, kind: str, required: bool) -> None:
        self._device_id = device_id
        self._locator = locator
        self._device_path = device_path
        self._kind = kind
        self._required = bool(required)
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._state = STATE_DEGRADED
        self._is_open = False
        self._first_frame_received = False
        self._frames_captured = 0
        self._fps = 0.0
        self._reconnect_attempts = 0
        self._last_frame_at: Optional[str] = None
        self._started_at = _utc_now_iso()
        self._last_state_change_at = self._started_at
        self._open_success_at: Optional[str] = None
        self._first_frame_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._initialization_complete = False
        self._monitor_message: Optional[str] = None

    def set_device_path(self, device_path: str) -> None:
        with self._lock:
            self._device_path = str(device_path)

    def mark_validating(self, message: str) -> None:
        self._set_state(STATE_DEGRADED, message, initialization_complete=False)

    def mark_open_attempt(self, attempt: int, max_attempts: int, message: str) -> None:
        self._set_state(
            STATE_DEGRADED,
            f"{message} (attempt {attempt}/{max_attempts})",
            initialization_complete=False,
        )

    def mark_open_success(self) -> None:
        with self._lock:
            self._is_open = True
            self._state = STATE_DEGRADED if not self._first_frame_received else STATE_AVAILABLE
            self._last_state_change_at = _utc_now_iso()
            self._open_success_at = _utc_now_iso()
            self._last_error = None
            self._monitor_message = None

    def mark_open_failure(self, error: str) -> None:
        self._set_state(STATE_FAILED, error, initialization_complete=False, is_open=False)

    def mark_first_frame_failure(self, error: str) -> None:
        self._set_state(STATE_FAILED, error, initialization_complete=True, is_open=False)

    def mark_frame(self, jpeg_bytes: bytes, fps: float) -> None:
        with self._lock:
            self._latest_jpeg = bytes(jpeg_bytes)
            self._frames_captured += 1
            self._fps = float(fps)
            now = _utc_now_iso()
            self._last_frame_at = now
            self._is_open = True
            self._state = STATE_AVAILABLE
            self._last_state_change_at = now
            self._last_error = None
            self._monitor_message = None
            self._initialization_complete = True
            if not self._first_frame_received:
                self._first_frame_received = True
                self._first_frame_at = now

    def mark_reconnecting(self, error: str) -> None:
        with self._lock:
            self._is_open = False
            self._state = STATE_DEGRADED
            self._last_state_change_at = _utc_now_iso()
            self._last_error = str(error)
            self._monitor_message = str(error)
            self._reconnect_attempts += 1
            self._initialization_complete = True

    def mark_exception(self, error: str) -> None:
        self._set_state(STATE_FAILED, error, initialization_complete=True, is_open=False)

    def mark_degraded(self, error: str) -> None:
        self._set_state(STATE_DEGRADED, error, initialization_complete=True, is_open=False)

    def mark_available(self, message: Optional[str] = None) -> None:
        with self._lock:
            self._state = STATE_AVAILABLE
            self._last_state_change_at = _utc_now_iso()
            self._monitor_message = message
            if message is None:
                self._last_error = None

    def mark_stopped(self) -> None:
        self._set_state(STATE_STOPPED, None, initialization_complete=True, is_open=False)

    def mark_initialization_complete(self) -> None:
        with self._lock:
            self._initialization_complete = True

    def is_ready(self) -> bool:
        with self._lock:
            return self._first_frame_received

    def latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return bytes(self._latest_jpeg) if self._latest_jpeg is not None else None

    def snapshot(self) -> StreamSnapshot:
        with self._lock:
            frame_available = self._latest_jpeg is not None
            return StreamSnapshot(
                device_id=self._device_id,
                locator=self._locator,
                device_path=self._device_path,
                kind=self._kind,
                required=self._required,
                state=self._state,
                is_open=self._is_open,
                first_frame_received=self._first_frame_received,
                frames_captured=self._frames_captured,
                fps=self._fps,
                reconnect_attempts=self._reconnect_attempts,
                last_frame_at=self._last_frame_at,
                last_frame_age_seconds=_age_seconds(self._last_frame_at),
                started_at=self._started_at,
                last_state_change_at=self._last_state_change_at,
                open_success_at=self._open_success_at,
                first_frame_at=self._first_frame_at,
                last_error=self._last_error,
                initialization_complete=self._initialization_complete,
                monitor_message=self._monitor_message,
                frame_available=frame_available,
            )

    def _set_state(
        self,
        state: str,
        error: Optional[str],
        *,
        initialization_complete: bool,
        is_open: bool,
    ) -> None:
        with self._lock:
            self._state = state
            self._last_state_change_at = _utc_now_iso()
            self._last_error = str(error) if error else None
            self._monitor_message = str(error) if error else None
            self._initialization_complete = bool(initialization_complete)
            self._is_open = bool(is_open)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(timestamp: Optional[str]) -> Optional[float]:
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = (now - dt.astimezone(timezone.utc)).total_seconds()
    return round(max(0.0, age), 3)
