"""Android device UI frame capture for scrcpy-style streaming.

The dashboard already consumes an MJPEG stream from the capture abstraction.
This module provides the same frame surface for Android devices through ADB,
so ADT/Android targets can be streamed without an HDMI capture card.  It uses
the same device id that scrcpy would use (`adb -s <serial>`).
"""
from __future__ import annotations

import base64
import contextlib
import logging
import os
import select
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_SESSION_POOL: Dict[Tuple[str, str], "ScrcpyStreamSession"] = {}
_SESSION_POOL_LOCK = threading.Lock()


def get_pooled_scrcpy_session(
    *,
    adb_device_id: str,
    adb_path: str = "adb",
    fps: float = 5.0,
    jpeg_quality: int = 80,
) -> "ScrcpyStreamSession":
    key = (str(adb_path or "adb").strip() or "adb", str(adb_device_id or "").strip())
    with _SESSION_POOL_LOCK:
        session = _SESSION_POOL.get(key)
        if session is None:
            session = ScrcpyStreamSession(
                adb_device_id=key[1],
                adb_path=key[0],
                fps=fps,
                jpeg_quality=jpeg_quality,
            )
            _SESSION_POOL[key] = session
        else:
            session.fps = max(0.5, float(fps or session.fps or 5.0))
            session.jpeg_quality = max(1, min(100, int(jpeg_quality or session.jpeg_quality or 80)))
        return session


def close_pooled_scrcpy_sessions() -> None:
    with _SESSION_POOL_LOCK:
        sessions = list(_SESSION_POOL.values())
        _SESSION_POOL.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            pass



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
        self._max_cached_jpeg_age_s: float = max(1.0, float(os.environ.get("SCRCPY_MAX_CACHED_JPEG_AGE_SECONDS", "5.0")))
        self._worker_processes: list[subprocess.Popen] = []
        # Some Android TV builds buffer `screenrecord --output-format=h264 -`
        # until the process exits. Keep the reliable screencap worker as the
        # default, while allowing screenrecord on devices that stream live.
        self._use_screenrecord = os.environ.get("SCRCPY_USE_SCREENRECORD", "false").strip().lower() in {"1", "true", "yes", "on"}

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

    def _store_jpeg(self, data: bytes, jpeg_quality: int) -> None:
        if not data:
            return
        with self._frame_lock:
            self._last_jpeg = data
            self._last_jpeg_quality = jpeg_quality
            self._last_jpeg_ts = time.monotonic()
        self.last_error = ""

    def _track_worker_process(self, process: subprocess.Popen) -> None:
        self._worker_processes.append(process)

    def _terminate_worker_processes(self) -> None:
        processes = list(self._worker_processes)
        self._worker_processes = []
        for process in processes:
            with contextlib.suppress(Exception):
                if process.poll() is None:
                    process.terminate()
            with contextlib.suppress(Exception):
                process.wait(timeout=0.5)
            with contextlib.suppress(Exception):
                if process.poll() is None:
                    process.kill()

    def _screenrecord_worker(self, jpeg_quality: int) -> bool:
        if not self._use_screenrecord or not shutil.which("ffmpeg") or not self.available():
            return False
        fps = max(1.0, min(15.0, float(self.fps or 12.0)))
        qscale = max(2, min(31, int(round(31 - (jpeg_quality / 100.0) * 25))))
        adb_cmd = [self.adb_path, "-s", self.adb_device_id, "exec-out", "screenrecord", "--output-format=h264", "-"]
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-vf",
            f"fps={fps}",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            str(qscale),
            "pipe:1",
        ]
        adb_proc = None
        ffmpeg_proc = None
        try:
            adb_proc = subprocess.Popen(adb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._track_worker_process(adb_proc)
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=adb_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._track_worker_process(ffmpeg_proc)
            if adb_proc.stdout is not None:
                with contextlib.suppress(Exception):
                    adb_proc.stdout.close()
            buffer = bytearray()
            frames = 0
            started_at = time.monotonic()
            last_output = started_at
            stdout_fd = ffmpeg_proc.stdout.fileno() if ffmpeg_proc.stdout is not None else -1
            while not self._worker_stop.is_set():
                if stdout_fd < 0:
                    break
                ready, _, _ = select.select([stdout_fd], [], [], 0.5)
                if not ready:
                    if frames == 0 and time.monotonic() - started_at > 3.0:
                        self.last_error = f"screenrecord produced no live frames for {self.adb_device_id}; falling back to screencap"
                        break
                    continue
                try:
                    chunk = os.read(stdout_fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                last_output = time.monotonic()
                buffer.extend(chunk)
                if frames == 0 and time.monotonic() - started_at > 3.0:
                    self.last_error = f"screenrecord produced no complete JPEG frames for {self.adb_device_id}; falling back to screencap"
                    break
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 1024 * 1024:
                            del buffer[:-2]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start > 0:
                            del buffer[:start]
                        break
                    frame = bytes(buffer[start : end + 2])
                    del buffer[: end + 2]
                    self._store_jpeg(frame, jpeg_quality)
                    frames += 1
            if frames == 0:
                self.last_error = self.last_error or f"screenrecord produced no frames for {self.adb_device_id}"
                return False
            return True
        except Exception as exc:
            self.last_error = f"screenrecord stream failed for {self.adb_device_id}: {exc}"
            return False
        finally:
            self._terminate_worker_processes()

    def _capture_worker(self, jpeg_quality: int) -> None:
        if self._screenrecord_worker(jpeg_quality):
            return
        interval = max(0.04, 1.0 / max(1.0, self.fps))
        while not self._worker_stop.is_set():
            started = time.monotonic()
            self._decode_jpeg_once(jpeg_quality, force=True)
            elapsed = time.monotonic() - started
            self._worker_stop.wait(max(0.01, interval - elapsed))

    def _ensure_worker(self, jpeg_quality: int) -> None:
        with self._worker_lock:
            if self._worker_thread and self._worker_thread.is_alive():
                # Keep the warm worker alive. Restarting it on a viewer quality
                # change makes the first stream request pay ADB startup cost.
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
        
        # Wait up to 0.45s for the worker to provide a fresh frame without blocking indefinitely
        for _ in range(9):
            now = time.monotonic()
            with self._frame_lock:
                cached_age = now - self._last_jpeg_ts if self._last_jpeg_ts else 999.0
                if self._last_jpeg is not None and cached_age <= self._max_cached_jpeg_age_s:
                    return self._last_jpeg
            time.sleep(0.05)
            
        with self._frame_lock:
            cached_age = time.monotonic() - self._last_jpeg_ts if self._last_jpeg_ts else 999.0
            if self._last_jpeg is not None and cached_age <= self._max_cached_jpeg_age_s:
                return self._last_jpeg
        return None

    def close(self) -> None:
        self._worker_stop.set()
        self._terminate_worker_processes()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
        self._worker_thread = None
        with self._frame_lock:
            self._last_png = None
            self._last_jpeg = None
