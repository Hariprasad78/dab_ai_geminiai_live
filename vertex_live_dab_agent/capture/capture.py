"""Screenshot capture using HDMI/camera/DAB image sources."""
import asyncio
import glob
import grp
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.client import DABClientBase
from vertex_live_dab_agent.capture.camera_devices import (
    camera_label,
    get_camera_path,
    validate_camera_devices,
)
from vertex_live_dab_agent.capture.hdmi_capture import HdmiCaptureSession

logger = logging.getLogger(__name__)


def extract_output_image_b64(payload: dict) -> Optional[str]:
    """Extract a normalized base64 PNG string from a DAB output/image payload.

    Supports both legacy and compliance-suite field names:
    - ``image``
    - ``outputImage``

    Also accepts ``data:image/...;base64,`` URIs and normalizes missing base64
    padding.
    """
    if not isinstance(payload, dict):
        return None

    raw = payload.get("image") or payload.get("outputImage")
    if not isinstance(raw, str) or not raw.strip():
        return None

    value = raw.strip()
    if value.startswith("data:image/"):
        comma = value.find(",")
        if comma != -1:
            value = value[comma + 1 :]

    # Remove whitespace/newlines and fix base64 padding.
    value = re.sub(r"\s+", "", value)
    remainder = len(value) % 4
    if remainder:
        value += "=" * (4 - remainder)

    return value or None


class CaptureResult:
    """Result from a capture operation."""

    def __init__(self, image_b64: Optional[str], ocr_text: Optional[str], source: str):
        self.image_b64 = image_b64
        self.ocr_text = ocr_text
        self.source = source


class ScreenCapture:
    """Handles screenshot capture from HDMI/camera/DAB."""

    def __init__(self, dab_client: DABClientBase, hdmi_session: Optional[HdmiCaptureSession] = None) -> None:
        self._config = get_config()
        validate_camera_devices()
        self._dab = dab_client
        self._image_source = self._normalize_source(self._config.image_source)
        self._selected_video_device = (self._config.hdmi_capture_device or "").strip() or None
        self._preferred_video_kind = "auto"
        self._rotation_degrees = self._normalize_rotation_degrees(
            getattr(self._config, "hdmi_capture_rotation", 0)
        )
        self._capture_pref_path = os.path.join(self._config.artifacts_base_dir, "capture_preference.json")
        self._load_capture_preference()
        self._capture_lock = asyncio.Lock()
        self._session_lock = threading.RLock()
        self._hdmi_reprobe_interval_s = 3.0
        self._next_hdmi_probe_ts = 0.0
        self._dab_capture_cooldown_s = 2.5
        self._next_dab_capture_ts = 0.0
        self._warned_dab_cooldown = False
        self._warned_no_hdmi = False
        self._last_hdmi_error: Optional[str] = None
        self._hdmi = hdmi_session

    def _normalize_rotation_degrees(self, rotation_degrees: Optional[int]) -> int:
        try:
            return HdmiCaptureSession.normalize_rotation_degrees(int(rotation_degrees or 0))
        except Exception:
            return 0

    def _normalize_source(self, source: str) -> str:
        s = (source or "auto").strip().lower()
        if s in {"hdmi", "hdmi-capture", "capture-card"}:
            return "hdmi-capture"
        if s in {"camera", "camera-capture", "webcam"}:
            return "camera-capture"
        if s in {"auto", "dab"}:
            return s
        return "auto"

    def _load_capture_preference(self) -> None:
        try:
            if not os.path.exists(self._capture_pref_path):
                return
            with open(self._capture_pref_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return
            source = data.get("source")
            if isinstance(source, str):
                self._image_source = self._normalize_source(source)
            device = data.get("device")
            if isinstance(device, str) and device.strip():
                selected = device.strip()
                if os.path.exists(selected):
                    if self._is_capture_capable_device(selected):
                        self._selected_video_device = selected
                    else:
                        logger.warning(
                            "Saved capture device is not capture-capable (index != 0), clearing preference: %s",
                            selected,
                        )
                        self._selected_video_device = None
                else:
                    logger.warning("Saved capture device is missing, clearing preference: %s", selected)
                    self._selected_video_device = None
            kind = str(data.get("preferred_kind") or "auto").strip().lower()
            if kind in {"auto", "hdmi", "camera"}:
                self._preferred_video_kind = kind
            rotation_degrees = data.get("rotation_degrees")
            if rotation_degrees is not None:
                self._rotation_degrees = self._normalize_rotation_degrees(rotation_degrees)
        except Exception as exc:
            logger.warning("Failed to load capture preference: %s", exc)

    def _save_capture_preference(self) -> None:
        try:
            os.makedirs(self._config.artifacts_base_dir, exist_ok=True)
            payload = {
                "source": self._image_source,
                "device": self._selected_video_device,
                "preferred_kind": self._preferred_video_kind,
                "rotation_degrees": self._rotation_degrees,
            }
            with open(self._capture_pref_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to persist capture preference: %s", exc)

    def _video_device_name(self, device: str) -> str:
        dev = str(device).strip()
        if not dev:
            return ""
        try:
            resolved = os.path.realpath(dev)
        except Exception:
            resolved = dev
        m = re.match(r"^/dev/video(\d+)$", resolved)
        if not m:
            return ""
        sys_name = f"/sys/class/video4linux/video{m.group(1)}/name"
        try:
            with open(sys_name, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except Exception:
            return ""

    def _video_device_index(self, device: str) -> Optional[int]:
        dev = str(device).strip()
        if not dev:
            return None
        try:
            resolved = os.path.realpath(dev)
        except Exception:
            resolved = dev
        m = re.match(r"^/dev/video(\d+)$", resolved)
        if not m:
            return None
        sys_index = f"/sys/class/video4linux/video{m.group(1)}/index"
        try:
            with open(sys_index, "r", encoding="utf-8") as fh:
                return int((fh.read() or "").strip())
        except Exception:
            return None

    def _is_capture_capable_device(self, device: str) -> bool:
        idx = self._video_device_index(device)
        if idx is not None:
            # Prefer the primary capture endpoint and avoid metadata/sideband
            # companion nodes (often index 1+ on UVC capture cards).
            return idx == 0
        return True

    def _classify_video_device_kind(self, name: str) -> str:
        n = str(name or "").lower()
        if any(k in n for k in ["hdmi", "capture", "cam link", "loop", "elgato", "u3"]):
            return "hdmi"
        if any(k in n for k in ["camera", "webcam", "integrated", "uvc"]):
            return "camera"
        return "unknown"

    def _list_video_device_details(self) -> list[dict]:
        details: list[dict] = []
        known = set(sorted(glob.glob("/dev/video*")))
        for key in ("adt4", "sonytv", "kirkwood"):
            path = get_camera_path(key)
            if path:
                known.add(path)

        for dev in sorted(known):
            if not os.path.exists(dev):
                continue
            name = self._video_device_name(dev)
            dev_index = self._video_device_index(dev)
            details.append(
                {
                    "device": dev,
                    "name": name,
                    "kind": self._classify_video_device_kind(name),
                    "index": dev_index,
                    "capture_capable": self._is_capture_capable_device(dev),
                    "readable": bool(os.access(dev, os.R_OK)),
                }
            )
        return details

    def _effective_kind_preference(self) -> str:
        kind_pref = self._preferred_video_kind
        if kind_pref == "auto":
            if self._image_source == "hdmi-capture":
                kind_pref = "hdmi"
            elif self._image_source == "camera-capture":
                kind_pref = "camera"
        return kind_pref

    def _is_kind_enabled(self, kind: str) -> bool:
        k = str(kind or "").lower()
        if k == "hdmi":
            return bool(self._config.enable_hdmi_capture)
        if k == "camera":
            return bool(self._config.enable_camera_capture)
        return True

    def set_capture_preference(
        self,
        *,
        source: Optional[str] = None,
        device: Optional[str] = None,
        preferred_kind: Optional[str] = None,
        rotation_degrees: Optional[int] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Set capture source/device preference and re-open capture session."""
        with self._session_lock:
            previous_selected_device = self._selected_video_device
            previous_source = self._image_source
            previous_kind = self._preferred_video_kind
            previous_rotation = self._rotation_degrees

            if source is not None:
                raw = str(source).strip().lower()
                allowed = {
                    "auto",
                    "dab",
                    "hdmi",
                    "hdmi-capture",
                    "capture-card",
                    "camera",
                    "camera-capture",
                    "webcam",
                }
                if raw not in allowed:
                    raise ValueError("Unsupported capture source")
                normalized = self._normalize_source(source)
                if normalized not in {"auto", "dab", "hdmi-capture", "camera-capture"}:
                    raise ValueError("Unsupported capture source")
                if normalized == "hdmi-capture" and not self._config.enable_hdmi_capture:
                    raise ValueError("HDMI capture is disabled by ENABLE_HDMI_CAPTURE=false")
                if normalized == "camera-capture" and not self._config.enable_camera_capture:
                    raise ValueError("Camera capture is disabled by ENABLE_CAMERA_CAPTURE=false")
                self._image_source = normalized

            if preferred_kind is not None:
                kind = str(preferred_kind).strip().lower()
                if kind not in {"auto", "hdmi", "camera"}:
                    raise ValueError("preferred_kind must be one of: auto, hdmi, camera")
                self._preferred_video_kind = kind

            if device is not None:
                dev = str(device or "").strip()
                if dev and (not dev.startswith("/dev/") or not os.path.exists(dev)):
                    raise ValueError("device must be an existing /dev/* path")
                if dev and not self._is_capture_capable_device(dev):
                    raise ValueError("device is not capture-capable (requires /dev/video* index0)")
                self._selected_video_device = dev or None
            elif source is not None and previous_source != self._image_source and self._selected_video_device:
                # If caller switches capture source without specifying a device,
                # avoid carrying over a stale explicit selection from the
                # opposite class (camera vs HDMI capture card).
                selected_kind = self._classify_video_device_kind(
                    self._video_device_name(self._selected_video_device)
                )
                target_kind = self._effective_kind_preference()
                if (
                    target_kind in {"hdmi", "camera"}
                    and selected_kind in {"hdmi", "camera"}
                    and selected_kind != target_kind
                ):
                    logger.info(
                        "Clearing stale selected video device %s (kind=%s) after source switch to %s",
                        self._selected_video_device,
                        selected_kind,
                        self._image_source,
                    )
                    self._selected_video_device = None

            if rotation_degrees is not None:
                try:
                    self._rotation_degrees = HdmiCaptureSession.normalize_rotation_degrees(int(rotation_degrees))
                except Exception:
                    raise ValueError("rotation_degrees must be one of: 0, 90, 180, 270, 360")

            selection_changed = (
                previous_selected_device != self._selected_video_device
                or previous_source != self._image_source
                or previous_kind != self._preferred_video_kind
                or previous_rotation != self._rotation_degrees
            )

            self.close()
            self._next_hdmi_probe_ts = 0.0
            self._warned_no_hdmi = False

            # Make camera switching return quickly; stream endpoints will lazily
            # open and stabilize the session on demand.
            self._hdmi = None

            if selection_changed:
                logger.info(
                    "Capture selection changed: source=%s preferred_kind=%s device=%s",
                    self._image_source,
                    self._preferred_video_kind,
                    self._selected_video_device or "auto",
                )

            if persist:
                self._save_capture_preference()

            return self.capture_source_status()

    def _init_hdmi_session(self) -> Optional[HdmiCaptureSession]:
        if self._image_source not in {"auto", "hdmi-capture", "camera-capture"}:
            return None

        if self._image_source == "hdmi-capture" and not self._config.enable_hdmi_capture:
            if not self._warned_no_hdmi:
                logger.info("HDMI capture is disabled by ENABLE_HDMI_CAPTURE=false")
                self._warned_no_hdmi = True
            return None

        if self._image_source == "camera-capture" and not self._config.enable_camera_capture:
            if not self._warned_no_hdmi:
                logger.info("Camera capture is disabled by ENABLE_CAMERA_CAPTURE=false")
                self._warned_no_hdmi = True
            return None

        configured = (self._selected_video_device or self._config.hdmi_capture_device or "").strip()
        if configured and not self._is_capture_capable_device(configured):
            logger.warning("Ignoring non-capture configured device (index != 0): %s", configured)
            configured = ""
        kind_pref = self._effective_kind_preference()
        explicit_device_requested = bool(configured)
        explicit_from_selection = bool((self._selected_video_device or "").strip())

        device_details = self._list_video_device_details()
        if self._image_source == "hdmi-capture":
            device_details = [d for d in device_details if str(d.get("kind")) != "camera"]
        elif self._image_source == "camera-capture":
            device_details = [d for d in device_details if str(d.get("kind")) != "hdmi"]

        device_details = [
            d for d in device_details
            if self._is_kind_enabled(str(d.get("kind") or "unknown"))
            and bool(d.get("capture_capable", True))
        ]
        devs = [d["device"] for d in device_details]
        if kind_pref in {"hdmi", "camera"}:
            preferred = [d["device"] for d in device_details if d.get("kind") == kind_pref]
            others = [d for d in devs if d not in preferred]
            devs = preferred + others

        candidates: list[str] = []
        if configured:
            candidates.append(configured)
        # Recovery path: if a previously selected /dev/videoN disappears or
        # stops producing frames after switching sources, fall back to
        # auto-discovered peers of the requested kind.
        if not explicit_device_requested or explicit_from_selection:
            candidates.extend([d for d in devs if d not in candidates])

        candidate_errors = []
        for device in candidates:
            if not os.path.exists(device):
                continue
            # Read permission is sufficient for VideoCapture; requiring write
            # causes false negatives on systems with strict v4l2 ACLs.
            if not os.access(device, os.R_OK):
                logger.info("Skipping HDMI device %s (missing read permission)", device)
                continue

            session = HdmiCaptureSession(
                device=device,
                width=self._config.hdmi_capture_width,
                height=self._config.hdmi_capture_height,
                fps=self._config.hdmi_capture_fps,
                fourcc=self._config.hdmi_capture_fourcc,
                rotation_degrees=self._rotation_degrees,
            )

            if device == get_camera_path("adt4"):
                logger.info("[INFO] Opening %s camera from %s", camera_label("adt4"), device)
            elif device == get_camera_path("sonytv"):
                logger.info("[INFO] Opening %s camera from %s", camera_label("sonytv"), device)
            elif device == get_camera_path("kirkwood"):
                logger.info("[INFO] Opening %s camera from %s", camera_label("kirkwood"), device)

            if not session.open():
                if session.last_error:
                    candidate_errors.append(f"{device}: {session.last_error}")
                continue

            logger.info("HDMI capture ready: device=%s", device)
            self._warned_no_hdmi = False
            self._last_hdmi_error = None
            return session

        if not self._warned_no_hdmi:
            video_devices = sorted(glob.glob("/dev/video*"))
            unreadable = [d for d in video_devices if not os.access(d, os.R_OK)]
            try:
                video_gid = grp.getgrnam("video").gr_gid
                user_in_video_group = video_gid in os.getgroups()
            except Exception:
                user_in_video_group = None

            if self._image_source == "auto":
                logger.info("No HDMI capture device detected; using DAB screenshot fallback")
            else:
                logger.info("No HDMI capture device detected")
            if explicit_device_requested and not explicit_from_selection:
                logger.warning(
                    "Configured capture device could not be opened; refusing fallback to a different device: %s",
                    configured,
                )
            if video_devices and unreadable:
                logger.warning(
                    "HDMI permission issue: detected video nodes but unreadable: %s",
                    ", ".join(unreadable),
                )
            if user_in_video_group is False:
                logger.warning(
                    "User is not in 'video' group. Add user to video group and re-login to enable HDMI capture."
                )
            if candidate_errors:
                logger.info("HDMI probe errors: %s", " | ".join(candidate_errors[:4]))
                self._last_hdmi_error = candidate_errors[0]
            elif explicit_device_requested and not explicit_from_selection:
                self._last_hdmi_error = f"Configured device could not be opened: {configured}"
            elif video_devices and unreadable:
                self._last_hdmi_error = (
                    "Video device permission issue: at least one /dev/video* node is unreadable"
                )
            else:
                self._last_hdmi_error = "No readable HDMI/camera capture device detected"
            self._warned_no_hdmi = True
        return None

    async def capture(self) -> CaptureResult:
        """Capture screenshot using configured source (HDMI or DAB)."""
        async with self._capture_lock:
            image_b64: Optional[str] = None
            source = "error"

            if self._image_source == "hdmi-capture":
                image_b64 = self._capture_from_hdmi()
                source = "hdmi-capture" if image_b64 else "error"
            elif self._image_source == "auto":
                image_b64 = self._capture_from_hdmi()
                source = "hdmi-capture" if image_b64 else "dab"
                if image_b64 is None:
                    image_b64 = await self._capture_from_dab()
            else:
                image_b64 = await self._capture_from_dab()
                source = "dab" if image_b64 else "error"

            return CaptureResult(image_b64=image_b64, ocr_text=None, source=source)

    async def capture_live_stream_frame(self) -> CaptureResult:
        """Capture one frame from the active live video session only.

        This method never falls back to DAB screenshots. It is intended for
        YTS/Gemini operator flows that must rely exclusively on the same live
        HDMI/camera stream shown in the dashboard when a capture session is
        healthy.
        """
        async with self._capture_lock:
            image_b64 = self._capture_from_hdmi()
            source = "hdmi-capture" if image_b64 else "error"
            return CaptureResult(image_b64=image_b64, ocr_text=None, source=source)

    async def _capture_from_dab(self) -> Optional[str]:
        """Capture one frame via DAB screenshot topic."""
        now = time.monotonic()
        if now < self._next_dab_capture_ts:
            if not self._warned_dab_cooldown:
                remaining = max(0.0, self._next_dab_capture_ts - now)
                logger.warning("Skipping DAB screenshot while in cooldown (%.1fs remaining)", remaining)
                self._warned_dab_cooldown = True
            return None
        try:
            resp = await self._dab.capture_screenshot()
            self._warned_dab_cooldown = False
            return extract_output_image_b64(resp.data) if resp.success else None
        except Exception as exc:
            logger.error("Screenshot capture failed: %s", exc)
            self._next_dab_capture_ts = time.monotonic() + self._dab_capture_cooldown_s
            self._warned_dab_cooldown = False
            return None

    def _capture_from_hdmi(self) -> Optional[str]:
        """Capture one frame from HDMI capture card as PNG base64."""
        with self._session_lock:
            if self._hdmi is None:
                now = time.monotonic()
                if now < self._next_hdmi_probe_ts:
                    return None
                self._next_hdmi_probe_ts = now + self._hdmi_reprobe_interval_s
                self._hdmi = self._init_hdmi_session()
            if self._hdmi is None:
                return None

            image_b64 = self._hdmi.capture_png_base64()
            if image_b64 is None and self._hdmi.last_error:
                self._last_hdmi_error = self._hdmi.last_error
                logger.warning("HDMI capture failed: %s", self._hdmi.last_error)
                # A transient read miss should not immediately tear down the
                # shared session for every consumer. Retry once with a fresh
                # open before giving up.
                failed_session = self._hdmi
                failed_session.close()
                self._hdmi = self._init_hdmi_session()
                if self._hdmi is not None:
                    image_b64 = self._hdmi.capture_png_base64()
                if image_b64 is None:
                    if self._hdmi is not None:
                        self._hdmi.close()
                    self._hdmi = None
            elif image_b64 is not None:
                self._last_hdmi_error = None
            return image_b64

    def ensure_hdmi_session(self, force: bool = False) -> bool:
        """Best-effort ensure the shared HDMI/camera session is open."""
        with self._session_lock:
            if self._hdmi is not None:
                return True
            now = time.monotonic()
            if not force and now < self._next_hdmi_probe_ts:
                return False
            self._next_hdmi_probe_ts = now + self._hdmi_reprobe_interval_s
            self._hdmi = self._init_hdmi_session()
            return self._hdmi is not None

    def capture_source_status(self) -> Dict[str, Any]:
        """Return capture source state for API/UI diagnostics."""
        with self._session_lock:
            hdmi_available = self._hdmi is not None
            hdmi_info = self._hdmi.device_info() if self._hdmi else {}
            hdmi_device = self._hdmi.device if self._hdmi else None
            configured_source = self._image_source
            selected_video_device = self._selected_video_device
            preferred_video_kind = self._preferred_video_kind
            effective_preferred_kind = self._effective_kind_preference()
            rotation_degrees = self._rotation_degrees
        video_details = self._list_video_device_details()
        video_devices = [d["device"] for d in video_details]
        device_readable = {dev: bool(os.access(dev, os.R_OK)) for dev in video_devices}
        try:
            video_gid = grp.getgrnam("video").gr_gid
            user_in_video_group = video_gid in os.getgroups()
        except Exception:
            user_in_video_group = None

        return {
            "configured_source": configured_source,
            "hdmi_configured": configured_source in {"auto", "hdmi-capture", "camera-capture"},
            "hdmi_available": hdmi_available,
            "hdmi_device": hdmi_device,
            "hdmi_info": hdmi_info,
            "hdmi_last_error": self._last_hdmi_error,
            "rotation_degrees": rotation_degrees,
            "enable_hdmi_capture": bool(self._config.enable_hdmi_capture),
            "enable_camera_capture": bool(self._config.enable_camera_capture),
            "selected_video_device": selected_video_device,
            "preferred_video_kind": preferred_video_kind,
            "effective_preferred_kind": effective_preferred_kind,
            "video_devices": video_devices,
            "video_device_details": video_details,
            "device_readable": device_readable,
            "user_in_video_group": user_in_video_group,
        }

    def get_hdmi_stream_frame_jpeg(self, quality: Optional[int] = None) -> Optional[bytes]:
        """Capture one HDMI frame encoded as JPEG for MJPEG web streaming."""
        with self._session_lock:
            if self._hdmi is None:
                now = time.monotonic()
                if now < self._next_hdmi_probe_ts:
                    return None
                self._next_hdmi_probe_ts = now + self._hdmi_reprobe_interval_s
                self._hdmi = self._init_hdmi_session()
            if self._hdmi is None:
                return None

            jpeg_quality = (
                int(quality)
                if quality is not None
                else int(getattr(self._config, "hdmi_stream_jpeg_quality", 80))
            )
            frame = self._hdmi.capture_jpeg_bytes(quality=jpeg_quality)
            if frame is None:
                failed_session = self._hdmi
                failed_session.close()
                self._hdmi = self._init_hdmi_session()
                if self._hdmi is not None:
                    frame = self._hdmi.capture_jpeg_bytes(quality=jpeg_quality)
                if frame is None:
                    if self._hdmi is not None:
                        self._hdmi.close()
                    self._hdmi = None
            return frame

    def get_hdmi_stream_frame_raw(self) -> Optional[Any]:
        """Capture one HDMI frame as a raw ndarray for server-side WebRTC tracks."""
        with self._session_lock:
            if self._hdmi is None:
                now = time.monotonic()
                if now < self._next_hdmi_probe_ts:
                    return None
                self._next_hdmi_probe_ts = now + self._hdmi_reprobe_interval_s
                self._hdmi = self._init_hdmi_session()
            if self._hdmi is None:
                return None

            frame = self._hdmi.read_frame()
            if frame is None:
                failed_session = self._hdmi
                failed_session.close()
                self._hdmi = self._init_hdmi_session()
                if self._hdmi is not None:
                    frame = self._hdmi.read_frame()
                if frame is None:
                    if self._hdmi is not None:
                        self._hdmi.close()
                    self._hdmi = None
            return frame

    def close(self) -> None:
        """Release optional capture resources."""
        with self._session_lock:
            if self._hdmi is not None:
                logger.info("Closing active HDMI/camera capture session: device=%s", self._hdmi.device)
                self._hdmi.close()
                self._hdmi = None
