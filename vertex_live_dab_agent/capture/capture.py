"""Screenshot capture and OCR integration."""
import base64
import glob
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.client import DABClientBase
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
    """Handles screenshot capture and OCR."""

    def __init__(self, dab_client: DABClientBase, hdmi_session: Optional[HdmiCaptureSession] = None) -> None:
        self._config = get_config()
        self._dab = dab_client
        self._ocr_available = self._check_ocr()
        self._image_source = self._normalize_source(self._config.image_source)
        self._hdmi_reprobe_interval_s = 3.0
        self._next_hdmi_probe_ts = 0.0
        self._warned_no_hdmi = False
        self._hdmi = hdmi_session or self._init_hdmi_session()

    def _normalize_source(self, source: str) -> str:
        s = (source or "auto").strip().lower()
        if s in {"hdmi", "hdmi-capture", "capture-card"}:
            return "hdmi-capture"
        if s in {"auto", "dab"}:
            return s
        return "auto"

    def _init_hdmi_session(self) -> Optional[HdmiCaptureSession]:
        if self._image_source not in {"auto", "hdmi-capture"}:
            return None

        configured = (self._config.hdmi_capture_device or "").strip()

        candidates = []
        if configured:
            candidates.append(configured)
        else:
            devs = sorted(glob.glob("/dev/video*"))
            preferred = ["/dev/video0", "/dev/video1"]
            candidates.extend([d for d in preferred if d in devs])
            candidates.extend([d for d in devs if d not in candidates])

        for device in candidates:
            if not os.path.exists(device):
                continue
            if not os.access(device, os.R_OK | os.W_OK):
                logger.info("Skipping HDMI device %s (insufficient permissions)", device)
                continue

            session = HdmiCaptureSession(
                device=device,
                width=self._config.hdmi_capture_width,
                height=self._config.hdmi_capture_height,
                fps=self._config.hdmi_capture_fps,
                fourcc=self._config.hdmi_capture_fourcc,
            )
            if not session.open():
                continue

            # Validate that we can obtain at least one frame.
            if session.read_frame() is None:
                session.close()
                continue

            logger.info("HDMI capture ready: device=%s", device)
            self._warned_no_hdmi = False
            return session

        if not self._warned_no_hdmi:
            logger.info("No HDMI capture device detected")
            self._warned_no_hdmi = True
        return None

    def _check_ocr(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            logger.info("pytesseract not available - OCR disabled")
            return False

    async def capture(self) -> CaptureResult:
        """Capture screenshot using configured source (HDMI or DAB)."""
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

        ocr_text = None
        if image_b64 and self._ocr_available:
            ocr_text = self._run_ocr(image_b64)

        return CaptureResult(image_b64=image_b64, ocr_text=ocr_text, source=source)

    async def _capture_from_dab(self) -> Optional[str]:
        """Capture one frame via DAB screenshot topic."""
        try:
            resp = await self._dab.capture_screenshot()
            return extract_output_image_b64(resp.data) if resp.success else None
        except Exception as exc:
            logger.error("Screenshot capture failed: %s", exc)
            return None

    def _capture_from_hdmi(self) -> Optional[str]:
        """Capture one frame from HDMI capture card as PNG base64."""
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
            logger.warning("HDMI capture failed: %s", self._hdmi.last_error)
            self._hdmi.close()
            self._hdmi = None
        return image_b64

    def capture_source_status(self) -> Dict[str, Any]:
        """Return capture source state for API/UI diagnostics."""
        hdmi_available = self._hdmi is not None
        hdmi_info = self._hdmi.device_info() if self._hdmi else {}
        return {
            "configured_source": self._image_source,
            "hdmi_configured": self._image_source in {"auto", "hdmi-capture"},
            "hdmi_available": hdmi_available,
            "hdmi_device": self._hdmi.device if self._hdmi else None,
            "hdmi_info": hdmi_info,
        }

    def get_hdmi_stream_frame_jpeg(self, quality: Optional[int] = None) -> Optional[bytes]:
        """Capture one HDMI frame encoded as JPEG for MJPEG web streaming."""
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
            self._hdmi.close()
            self._hdmi = None
        return frame

    def close(self) -> None:
        """Release optional capture resources."""
        if self._hdmi is not None:
            self._hdmi.close()
            self._hdmi = None

    def _run_ocr(self, image_b64: str) -> Optional[str]:
        """Run OCR on base64 image."""
        try:
            import io
            import pytesseract
            from PIL import Image
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip() if text.strip() else None
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return None
