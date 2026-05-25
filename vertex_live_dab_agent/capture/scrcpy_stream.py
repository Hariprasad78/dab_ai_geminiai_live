"""Android device UI frame capture for scrcpy-style streaming.

The dashboard already consumes an MJPEG stream from the capture abstraction.
This module provides the same frame surface for Android devices through ADB,
so ADT/Android targets can be streamed without an HDMI capture card.  It uses
the same device id that scrcpy would use (`adb -s <serial>`).
"""
from __future__ import annotations

import base64
import logging
import shutil
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ScrcpyStreamSession:
    """Capture Android UI frames over ADB for browser/Gemini streaming."""

    def __init__(
        self,
        *,
        adb_device_id: str,
        adb_path: str = "adb",
        fps: float = 5.0,
        jpeg_quality: int = 80,
    ) -> None:
        self.adb_device_id = str(adb_device_id or "").strip()
        self.adb_path = str(adb_path or "adb").strip() or "adb"
        self.fps = max(0.5, float(fps or 5.0))
        self.jpeg_quality = max(1, min(100, int(jpeg_quality or 80)))
        self.last_error: str = ""
        self._last_png: Optional[bytes] = None
        self._last_png_ts: float = 0.0
        self._last_jpeg: Optional[bytes] = None
        self._last_jpeg_quality: int = 0
        self._last_jpeg_ts: float = 0.0
        self._frame_lock = threading.Lock()
        self._adb_lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_quality: int = self.jpeg_quality

    def available(self) -> bool:
        if not self.adb_device_id:
            self.last_error = "missing adb device id"
            return False
        if not shutil.which(self.adb_path):
            self.last_error = f"adb binary not found: {self.adb_path}"
            return False
        return True

    def device_info(self) -> dict:
        return {
            "device": self.adb_device_id,
            "adb_path": self.adb_path,
            "fps": self.fps,
            "last_error": self.last_error,
            "low_latency_worker": bool(self._worker_thread and self._worker_thread.is_alive()),
            "last_frame_age_ms": (
                round((time.monotonic() - self._last_jpeg_ts) * 1000.0, 1)
                if self._last_jpeg_ts
                else None
            ),
        }

    def _run_adb(self, args: list[str], timeout: float = 4.0) -> subprocess.CompletedProcess[bytes]:
        cmd = [self.adb_path, "-s", self.adb_device_id, *args]
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)

    def _normalize_png(self, payload: bytes) -> Optional[bytes]:
        if not payload:
            return None
        data = payload
        start = data.find(b"\x89PNG\r\n\x1a\n")
        if start == -1:
            start = data.find(b"\x89PNG\n\x1a\n")
        if start > 0:
            data = data[start:]
        # Some adb shell paths alter only the PNG signature newline. Do not
        # rewrite CRLF globally, because that corrupts compressed IDAT bytes.
        if data.startswith(b"\x89PNG\n\x1a\n"):
            data = b"\x89PNG\r\n\x1a\n" + data[7:]
        iend = data.find(b"IEND")
        if iend != -1 and iend + 8 <= len(data):
            data = data[: iend + 8]
        return data if data.startswith(b"\x89PNG\r\n\x1a\n") else None

    def capture_png_bytes(self, *, force: bool = False) -> Optional[bytes]:
        now = time.monotonic()
        min_interval = 1.0 / self.fps
        with self._frame_lock:
            if not force and self._last_png is not None and now - self._last_png_ts < min_interval:
                return self._last_png
        if not self.available():
            return None
        try:
            with self._adb_lock:
                result = self._run_adb(["exec-out", "screencap", "-p"])
        except subprocess.TimeoutExpired:
            self.last_error = f"adb screencap timed out for {self.adb_device_id}"
            return None
        except Exception as exc:
            self.last_error = str(exc)
            return None
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            self.last_error = stderr or f"adb screencap failed with code {result.returncode}"
            return None
        png = self._normalize_png(result.stdout)
        if not png:
            self.last_error = "adb screencap did not return a PNG frame"
            return None
        self.last_error = ""
        with self._frame_lock:
            self._last_png = png
            self._last_png_ts = now
        return png

    def capture_png_base64(self) -> Optional[str]:
        png = self.capture_png_bytes()
        if not png:
            return None
        return base64.b64encode(png).decode("ascii")

    def _decode_jpeg_once(self, jpeg_quality: int, *, force: bool = False) -> Optional[bytes]:
        now = time.monotonic()
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            frame = None
            for attempt in range(2):
                png = self.capture_png_bytes(force=force or attempt > 0)
                if not png:
                    return None
                arr = np.frombuffer(png, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    break
                with self._frame_lock:
                    self._last_png = None
                    self._last_png_ts = 0.0
            if frame is None:
                self.last_error = "OpenCV could not decode adb screencap PNG"
                return None
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                self.last_error = "OpenCV could not encode scrcpy frame as JPEG"
                return None
            data = encoded.tobytes()
        except Exception as exc:
            self.last_error = f"JPEG encoding unavailable for scrcpy frame: {exc}"
            logger.warning(self.last_error)
            return None
        with self._frame_lock:
            self._last_jpeg = data
            self._last_jpeg_quality = jpeg_quality
            self._last_jpeg_ts = now
        self.last_error = ""
        return data

    def _capture_worker(self, jpeg_quality: int) -> None:
        interval = max(0.04, 1.0 / max(1.0, self.fps))
        while not self._worker_stop.is_set():
            started = time.monotonic()
            self._decode_jpeg_once(jpeg_quality, force=True)
            elapsed = time.monotonic() - started
            self._worker_stop.wait(max(0.01, interval - elapsed))

    def _ensure_worker(self, jpeg_quality: int) -> None:
        with self._worker_lock:
            if self._worker_thread and self._worker_thread.is_alive() and self._worker_quality == jpeg_quality:
                return
            self._worker_stop.set()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=0.2)
            self._worker_stop = threading.Event()
            self._worker_quality = jpeg_quality
            self._worker_thread = threading.Thread(
                target=self._capture_worker,
                args=(jpeg_quality,),
                name=f"scrcpy-capture-{self.adb_device_id}",
                daemon=True,
            )
            self._worker_thread.start()

    def capture_jpeg_bytes(self, quality: Optional[int] = None) -> Optional[bytes]:
        jpeg_quality = max(1, min(100, int(quality or self.jpeg_quality)))
        self._ensure_worker(jpeg_quality)
        now = time.monotonic()
        with self._frame_lock:
            if self._last_jpeg is not None and self._last_jpeg_quality == jpeg_quality:
                return self._last_jpeg
        # First frame only: capture synchronously so the stream can start.
        frame = self._decode_jpeg_once(jpeg_quality, force=True)
        if frame is not None:
            return frame
        with self._frame_lock:
            if self._last_jpeg is not None and now - self._last_jpeg_ts < 2.0:
                return self._last_jpeg
        return None

    def close(self) -> None:
        self._worker_stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
        self._worker_thread = None
        with self._frame_lock:
            self._last_png = None
            self._last_jpeg = None
