"""HDMI/V4L2 capture helpers used by the agent and web preview."""

from __future__ import annotations

import base64
import contextlib
import glob
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_warned_missing_opencv = False
_CAPTURE_WIDTH_720P = 1280
_CAPTURE_HEIGHT_720P = 720
_DEVICE_OPERATION_LOCKS: Dict[str, threading.RLock] = {}
_DEVICE_OPERATION_LOCKS_GUARD = threading.Lock()
_DEFAULT_HOLDER_PRIORITIES = {
    "ffmpeg": 10,
    "python": 40,
    "python3": 40,
}


def _device_operation_lock(device: str) -> threading.RLock:
    key = str(device or "").strip()
    with contextlib.suppress(Exception):
        key = os.path.realpath(key)
    if not key:
        key = str(device or "")
    with _DEVICE_OPERATION_LOCKS_GUARD:
        lock = _DEVICE_OPERATION_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _DEVICE_OPERATION_LOCKS[key] = lock
        return lock


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_priority_map(value: str) -> Dict[str, int]:
    priorities: Dict[str, int] = dict(_DEFAULT_HOLDER_PRIORITIES)
    for raw_item in str(value or "").split(","):
        item = raw_item.strip()
        if not item or "=" not in item:
            continue
        name, raw_priority = item.split("=", 1)
        name = name.strip()
        if not name:
            continue
        try:
            priorities[name] = int(raw_priority.strip())
        except Exception:
            logger.warning("Ignoring invalid camera holder priority entry: %s", item)
    return priorities


def _process_command(pid_name: str) -> str:
    command = ""
    with contextlib.suppress(Exception):
        with open(f"/proc/{pid_name}/comm", "r", encoding="utf-8") as fh:
            command = fh.read().strip()
    return command


def _process_cmdline(pid_name: str) -> str:
    with contextlib.suppress(Exception):
        with open(f"/proc/{pid_name}/cmdline", "rb") as fh:
            raw = fh.read().replace(b"\x00", b" ").strip()
        return raw.decode("utf-8", errors="replace")
    return ""


def _device_holders(device: str) -> List[Dict[str, Any]]:
    """Return processes currently holding a video device."""
    dev = str(device or "").strip()
    if not dev:
        return []
    with contextlib.suppress(Exception):
        dev = os.path.realpath(dev)
    try:
        dev_stat = os.stat(dev)
    except Exception:
        return []

    holders: List[Dict[str, Any]] = []
    current_pid = os.getpid()
    for pid_name in os.listdir("/proc"):
        if not pid_name.isdigit():
            continue
        pid = int(pid_name)
        if pid == current_pid:
            continue
        fd_dir = f"/proc/{pid_name}/fd"
        try:
            fd_names = os.listdir(fd_dir)
        except Exception:
            continue
        holds_device = False
        for fd_name in fd_names:
            fd_path = os.path.join(fd_dir, fd_name)
            try:
                fd_stat = os.stat(fd_path)
            except Exception:
                continue
            if fd_stat.st_rdev == dev_stat.st_rdev:
                holds_device = True
                break
        if not holds_device:
            continue
        holders.append(
            {
                "pid": pid,
                "pid_name": pid_name,
                "command": _process_command(pid_name),
                "cmdline": _process_cmdline(pid_name),
            }
        )
    return holders


def _holder_priority(holder: Dict[str, Any]) -> int:
    priority_map = _parse_priority_map(os.environ.get("CAMERA_HOLDER_PRIORITIES", ""))
    command = str(holder.get("command") or "").strip()
    cmdline = str(holder.get("cmdline") or "")
    if command in priority_map:
        return priority_map[command]
    for name, priority in priority_map.items():
        if name and name in cmdline:
            return priority
    try:
        return int(os.environ.get("CAMERA_UNKNOWN_HOLDER_PRIORITY", "50"))
    except Exception:
        return 50


def _device_holder_summary(device: str) -> str:
    """Return a short summary of processes currently holding a video device."""
    holders = _device_holders(device)
    if not holders:
        return ""
    bits = []
    for holder in holders[:3]:
        command = str(holder.get("command") or "process")
        bits.append(f"{command} pid={holder.get('pid')}")
    return "Camera device is already in use by " + ", ".join(bits)


def _kill_device_holders(device: str, *, priority_aware: bool = True) -> int:
    """Kill lower-priority processes holding one video device."""
    whitelist_env = os.environ.get("CAMERA_HOLDER_WHITELIST", "")
    whitelist = {x.strip() for x in whitelist_env.split(",") if x.strip()}
    try:
        owner_priority = int(os.environ.get("CAMERA_DEVICE_OWNER_PRIORITY", "60"))
    except Exception:
        owner_priority = 60

    killed_count = 0
    for holder in _device_holders(device):
        pid = int(holder.get("pid") or 0)
        command = str(holder.get("command") or "")
        if not pid:
            continue
        if str(pid) in whitelist:
            logger.info("Ignoring whitelisted process pid=%s holding device %s", pid, device)
            continue

        if command and command in whitelist:
            logger.info("Ignoring whitelisted application '%s' (pid=%s) holding device %s", command, pid, device)
            continue

        holder_priority = _holder_priority(holder)
        if priority_aware and holder_priority >= owner_priority:
            logger.warning(
                "Leaving higher/equal priority camera holder alive: pid=%s command=%s priority=%s owner_priority=%s device=%s",
                pid,
                command or "process",
                holder_priority,
                owner_priority,
                device,
            )
            continue

        try:
            os.kill(pid, signal.SIGTERM)
            logger.warning(
                "Sent SIGTERM to lower-priority process pid=%s (%s, priority=%s) holding device %s",
                pid,
                command or "process",
                holder_priority,
                device,
            )

            for _ in range(15):  # Wait up to 1.5 seconds for graceful cleanup
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
            else:
                os.kill(pid, signal.SIGKILL)
                logger.warning("Process pid=%s (%s) failed to exit cleanly; escalated to SIGKILL", pid, command)
            killed_count += 1
        except Exception as exc:
            logger.error("Failed to kill process pid=%s holding device %s: %s", pid, device, exc)
    return killed_count


def _recover_busy_device(device: str) -> int:
    policy = os.environ.get("CAMERA_BUSY_RECOVERY_POLICY", "priority").strip().lower()
    if not _parse_bool_env("FORCE_KILL_CAMERA_HOLDERS", False):
        return 0
    if policy in {"off", "false", "none", "disabled"}:
        return 0
    return _kill_device_holders(device, priority_aware=policy != "force")


class HdmiCaptureError(Exception):
    """Raised for HDMI capture open/read/encode errors."""


def _import_cv2() -> Any:
    try:
        # Reduce noisy OpenCV backend warnings in production logs.
        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
        # Best-effort reduction of FFmpeg/OpenCV backend verbosity.
        os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
        os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
        import cv2  # type: ignore

        try:
            # OpenCV logging API differs by build/version.
            if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            elif hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        except Exception:
            pass

        return cv2
    except Exception as exc:  # pragma: no cover - optional dependency
        raise HdmiCaptureError(
            "OpenCV is required for HDMI capture. Install: pip install opencv-python-headless"
        ) from exc


class HdmiCaptureSession:
    """Small wrapper around OpenCV VideoCapture for HDMI-to-USB cards."""

    def __init__(
        self,
        device: str,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        fourcc: str = "MJPG",
        rotation_degrees: int = 0,
    ) -> None:
        self.device = device
        requested_width = int(width)
        requested_height = int(height)
        self.width = _CAPTURE_WIDTH_720P
        self.height = _CAPTURE_HEIGHT_720P
        if requested_width != self.width or requested_height != self.height:
            logger.info(
                "Forcing capture resolution to 720p: requested=%sx%s effective=%sx%s",
                requested_width,
                requested_height,
                self.width,
                self.height,
            )
        self.fps = float(fps)
        self.fourcc = (fourcc or "MJPG").upper()
        self.rotation_degrees = self.normalize_rotation_degrees(rotation_degrees)

        self._cv2: Optional[Any] = None
        self._cap: Optional[Any] = None
        self._ffmpeg_fallback = False
        self._lock = threading.Lock()
        self._capture_io_lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._last_frame: Optional[Any] = None
        self._last_frame_ts: float = 0.0
        self._opened_at: float = 0.0
        self._reader_stop = threading.Event()
        self._frame_ready = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._device_lock = _device_operation_lock(device)

    @staticmethod
    def normalize_rotation_degrees(rotation_degrees: int) -> int:
        value = int(rotation_degrees)
        if value == 360:
            return 0
        if value not in {0, 90, 180, 270}:
            raise ValueError("rotation_degrees must be one of: 0, 90, 180, 270, 360")
        return value

    def _rotate_frame(self, frame: Any) -> Any:
        if self.rotation_degrees == 0:
            return frame
        cv2 = self._cv2 or _import_cv2()
        if self.rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def open(self) -> bool:
        """Open the configured V4L2 device and apply capture settings."""
        with self._device_lock, self._lock:
            if self._cap is not None:
                return True

            try:
                cv2 = _import_cv2()
                self._cv2 = cv2

                cap = self._open_capture_with_fallbacks(cv2)
                if not cap or not cap.isOpened():
                    if self._open_ffmpeg_fallback():
                        return True
                    fallback_error = str(self._last_error or "").strip()
                    holder_summary = _device_holder_summary(self.device)

                    if holder_summary and _parse_bool_env("FORCE_KILL_CAMERA_HOLDERS", False):
                        logger.warning("Camera %s is busy. Attempting to gracefully free the device...", self.device)
                        killed = _recover_busy_device(self.device)
                        if killed > 0:
                            time.sleep(0.5)  # Give the OS a moment to clean up file descriptors
                            cap = self._open_capture_with_fallbacks(cv2)
                            if cap and cap.isOpened():
                                logger.info("Successfully recovered camera %s after freeing resources", self.device)
                                holder_summary = ""
                            elif self._open_ffmpeg_fallback():
                                logger.info("Successfully recovered camera %s via ffmpeg fallback after freeing resources", self.device)
                                return True

                    self._last_error = f"Unable to open capture device: {self.device}"
                    if fallback_error:
                        self._last_error = f"{self._last_error}. {fallback_error}"
                    if holder_summary:
                        self._last_error = f"{self._last_error}. {holder_summary}"
                    return False

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                with contextlib.suppress(Exception):
                    # Keep capture buffers shallow so camera switches reflect quickly.
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if len(self.fourcc) == 4:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

                self._cap = cap
                self._last_error = None
                self._opened_at = time.monotonic()
                self._reader_stop.clear()
                self._frame_ready.clear()
                self._reader_thread = threading.Thread(
                    target=self._reader_loop,
                    name=f"hdmi-capture-{os.path.basename(str(self.device))}",
                    daemon=True,
                )
                self._reader_thread.start()
                return True
            except Exception as exc:
                self._last_error = str(exc)
                if isinstance(exc, HdmiCaptureError):
                    global _warned_missing_opencv
                    if not _warned_missing_opencv:
                        logger.warning("HDMI open failed: %s", exc)
                        _warned_missing_opencv = True
                    else:
                        logger.debug("HDMI open skipped: %s", exc)
                else:
                    logger.warning("HDMI open failed: %s", exc)
                return False

    def _open_capture_with_fallbacks(self, cv2: Any) -> Optional[Any]:
        """Try path/index + backend combinations for better Linux OpenCV compatibility."""
        cap_v4l2 = getattr(cv2, "CAP_V4L2", None)

        candidates: List[tuple[Any, Optional[int]]] = []
        seen_candidates: set[tuple[str, Optional[int]]] = set()

        def add_candidate(source: Any, backend: Optional[int]) -> None:
            key = (str(source), backend)
            if key in seen_candidates:
                return
            seen_candidates.add(key)
            candidates.append((source, backend))

        # Prefer exact persistent paths, but also try their resolved
        # /dev/videoN target because some OpenCV V4L2 builds cannot open
        # /dev/v4l/by-id symlinks by name.
        dev = str(self.device).strip()
        if re.match(r"^/dev/video\d+$", dev):
            add_candidate(dev, cap_v4l2)
            add_candidate(dev, None)
        else:
            # Explicit numeric sources are still supported.
            try:
                idx = int(dev)
                add_candidate(idx, cap_v4l2)
                add_candidate(idx, None)
            except Exception:
                add_candidate(dev, cap_v4l2)
                add_candidate(dev, None)
                with contextlib.suppress(Exception):
                    resolved = os.path.realpath(dev)
                    if resolved and resolved != dev and re.match(r"^/dev/video\d+$", resolved):
                        add_candidate(resolved, cap_v4l2)
                        add_candidate(resolved, None)

        for source, backend in candidates:
            cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
            if cap and cap.isOpened():
                return cap
            try:
                cap.release()
            except Exception:
                pass

        return None

    def _ffmpeg_input_formats(self) -> List[str]:
        formats: List[str] = []
        if self.fourcc in {"MJPG", "MJPEG"}:
            formats.append("mjpeg")
        elif self.fourcc == "YUYV":
            formats.append("yuyv422")
        formats.append("")
        return list(dict.fromkeys(formats))

    def _ffmpeg_sources(self) -> List[str]:
        sources = [str(self.device)]
        with contextlib.suppress(Exception):
            resolved = os.path.realpath(str(self.device))
            if resolved and resolved not in sources:
                sources.append(resolved)
        return sources

    def _ffmpeg_sizes(self) -> List[tuple[int, int]]:
        sizes = [
            (int(self.width), int(self.height)),
            (1280, 720),
            (640, 480),
            (640, 360),
        ]
        return list(dict.fromkeys(sizes))

    def _capture_ffmpeg_image_bytes(self, codec: str = "mjpeg") -> Optional[bytes]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._last_error = "ffmpeg not found; cannot use V4L2 fallback capture"
            return None
        last_error = ""
        fps_value = max(1, int(round(float(self.fps or 30.0))))
        for source in self._ffmpeg_sources():
            for width, height in self._ffmpeg_sizes():
                for input_format in self._ffmpeg_input_formats():
                    cmd = [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "v4l2",
                    ]
                    if input_format:
                        cmd.extend(["-input_format", input_format])
                    cmd.extend([
                        "-video_size",
                        f"{width}x{height}",
                        "-framerate",
                        str(fps_value),
                        "-i",
                        source,
                        "-frames:v",
                        "1",
                        "-an",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        codec,
                        "pipe:1",
                    ])
                    try:
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=4.0,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        last_error = f"ffmpeg fallback timed out for {source} at {width}x{height}"
                        continue
                    except Exception as exc:
                        last_error = f"ffmpeg fallback failed for {source}: {exc}"
                        continue
                    if result.returncode == 0 and result.stdout:
                        if width != self.width or height != self.height:
                            logger.info(
                                "ffmpeg fallback using alternate capture size: device=%s size=%sx%s",
                                source,
                                width,
                                height,
                            )
                        self._last_error = None
                        return result.stdout
                    stderr = result.stderr.decode("utf-8", errors="replace").strip()
                    last_error = stderr or f"ffmpeg exited with status {result.returncode}"
        self._last_error = f"ffmpeg fallback could not capture {self.device}: {last_error}"
        return None

    def _open_ffmpeg_fallback(self) -> bool:
        if not shutil.which("ffmpeg"):
            return False
        probe = self._capture_ffmpeg_image_bytes("mjpeg")
        if not probe:
            return False
        self._ffmpeg_fallback = True
        self._opened_at = time.monotonic()
        self._last_error = None
        logger.info("HDMI/camera capture using ffmpeg V4L2 fallback: device=%s", self.device)
        return True

    def close(self) -> None:
        """Release the capture device."""
        self._reader_stop.set()
        cap: Optional[Any] = None
        reader: Optional[threading.Thread] = None
        with self._device_lock:
            with self._lock:
                reader = self._reader_thread
                self._reader_thread = None
                self._frame_ready.clear()
                self._ffmpeg_fallback = False
                self._last_frame = None
                self._last_frame_ts = 0.0
                if self._cap is not None:
                    cap = self._cap
                    self._cap = None
            if cap is not None:
                with self._capture_io_lock:
                    with contextlib.suppress(Exception):
                        cap.release()
            if reader is not None and reader.is_alive() and reader is not threading.current_thread():
                reader.join(timeout=1.0)
            if cap is not None:
                # V4L2 devices can stay busy briefly after release; give the
                # kernel/driver a small handoff window before another open.
                with contextlib.suppress(Exception):
                    time.sleep(0.15)

    def _copy_frame(self, frame: Any) -> Any:
        try:
            return frame.copy()
        except Exception:
            return frame

    def _reader_loop(self) -> None:
        consecutive_failures = 0
        while not self._reader_stop.is_set():
            with self._lock:
                cap = self._cap
                if cap is None:
                    break
                cv2 = self._cv2
            # Never hold the shared lock during a potentially blocking read.
            with self._capture_io_lock:
                if self._reader_stop.is_set():
                    break
                ok, frame = cap.read()
            if ok and frame is not None:
                try:
                    rotated = self._rotate_frame(frame)
                except Exception as exc:
                    self._last_error = str(exc)
                    time.sleep(0.05)
                    continue
                with self._lock:
                    self._last_frame = rotated
                    self._last_frame_ts = time.monotonic()
                    self._last_error = None
                    self._frame_ready.set()
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            if consecutive_failures >= 12:
                self._last_error = "Failed to read frame"
            time.sleep(0.04 if cv2 is not None else 0.08)

    def read_frame(self) -> Optional[Any]:
        """Read one frame from the HDMI input."""
        if self._cap is None and not self._ffmpeg_fallback and not self.open():
            return None
        if self._ffmpeg_fallback:
            data = self._capture_ffmpeg_image_bytes("mjpeg")
            if not data:
                return None
            try:
                import numpy as np  # type: ignore

                cv2 = self._cv2 or _import_cv2()
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    self._last_error = "ffmpeg fallback produced an undecodable frame"
                    return None
                return self._rotate_frame(frame)
            except Exception as exc:
                self._last_error = f"ffmpeg fallback decode failed: {exc}"
                return None

        if self._last_frame is None:
            self._frame_ready.wait(timeout=0.45)
        frame: Optional[Any] = None
        with self._lock:
            if self._last_frame is not None:
                frame = self._copy_frame(self._last_frame)
            elif (time.monotonic() - self._opened_at) > 1.5:
                self._last_error = self._last_error or "Failed to read frame"
        return frame

    def capture_png_base64(self) -> Optional[str]:
        """Capture one frame and return as base64 PNG."""
        if self._ffmpeg_fallback or (self._cap is None and self.open() and self._ffmpeg_fallback):
            data = self._capture_ffmpeg_image_bytes("png")
            return base64.b64encode(data).decode("ascii") if data else None

        frame = self.read_frame()
        if frame is None:
            return None

        cv2 = self._cv2 or _import_cv2()
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            self._last_error = "Failed to encode frame as PNG"
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def capture_jpeg_bytes(self, quality: int = 80) -> Optional[bytes]:
        """Capture one frame and return as JPEG bytes."""
        if self._ffmpeg_fallback or (self._cap is None and self.open() and self._ffmpeg_fallback):
            return self._capture_ffmpeg_image_bytes("mjpeg")

        frame = self.read_frame()
        if frame is None:
            return None

        cv2 = self._cv2 or _import_cv2()
        quality = max(30, min(95, int(quality)))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            self._last_error = "Failed to encode frame as JPEG"
            return None
        return encoded.tobytes()

    def device_info(self) -> Dict[str, float]:
        """Return best-effort information about the active device."""
        with self._lock:
            if self._ffmpeg_fallback:
                return {
                    "width": float(self.width),
                    "height": float(self.height),
                    "fps": float(self.fps),
                    "rotation_degrees": float(self.rotation_degrees),
                }
            if self._cap is None:
                return {}
            cv2 = self._cv2 or _import_cv2()
            return {
                "width": float(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": float(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(self._cap.get(cv2.CAP_PROP_FPS)),
                "rotation_degrees": float(self.rotation_degrees),
            }

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error


def list_hdmi_devices(
    fourcc: str = "MJPG",
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
) -> List[Dict[str, float | str]]:
    """Probe /dev/video* and return devices that can deliver at least one frame."""
    devices: List[Dict[str, float | str]] = []
    for dev in sorted(glob.glob("/dev/video*")):
        sess = HdmiCaptureSession(
            dev,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )
        try:
            if not sess.open():
                continue
            frame = sess.read_frame()
            if frame is None:
                continue
            info = sess.device_info()
            devices.append(
                {
                    "device": dev,
                    "width": float(info.get("width", width)),
                    "height": float(info.get("height", height)),
                    "fps": float(info.get("fps", fps)),
                }
            )
        finally:
            sess.close()
    return devices
