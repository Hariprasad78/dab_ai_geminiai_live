"""Screenshot capture and OCR integration."""
import base64
import logging
import re
from typing import Optional

from vertex_live_dab_agent.dab.client import DABClientBase

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

    def __init__(self, dab_client: DABClientBase) -> None:
        self._dab = dab_client
        self._ocr_available = self._check_ocr()

    def _check_ocr(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            return True
        except ImportError:
            logger.info("pytesseract not available - OCR disabled")
            return False

    async def capture(self) -> CaptureResult:
        """Capture screenshot via DAB."""
        try:
            resp = await self._dab.capture_screenshot()
            image_b64 = extract_output_image_b64(resp.data) if resp.success else None
            ocr_text = None
            if image_b64 and self._ocr_available:
                ocr_text = self._run_ocr(image_b64)
            return CaptureResult(image_b64=image_b64, ocr_text=ocr_text, source="dab")
        except Exception as exc:
            logger.error("Screenshot capture failed: %s", exc)
            return CaptureResult(image_b64=None, ocr_text=None, source="error")

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
