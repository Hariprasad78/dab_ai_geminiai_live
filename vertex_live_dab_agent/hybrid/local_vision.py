"""Local visual feature extraction for TV screenshots and OCR summaries."""

from __future__ import annotations

import base64
import binascii
import hashlib
import struct
from typing import Any, Dict, List, Optional


def _safe_b64decode(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=False)
    except (binascii.Error, ValueError):
        return b""


def _read_png_dimensions(image_bytes: bytes) -> tuple[Optional[int], Optional[int]]:
    if len(image_bytes) < 24 or image_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    try:
        width, height = struct.unpack(">II", image_bytes[16:24])
        return int(width), int(height)
    except struct.error:
        return None, None


def _contains_any(text: str, tokens: List[str]) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in tokens)


def extract_local_visual_features(
    *,
    image_b64: Optional[str],
    ocr_text: Optional[str],
) -> Dict[str, Any]:
    """Extract lightweight local features without external model dependencies."""

    image_bytes = _safe_b64decode(str(image_b64 or ""))
    width, height = _read_png_dimensions(image_bytes)
    text = str(ocr_text or "").strip()
    lowered = text.lower()
    tokens = [token for token in lowered.replace("/", " ").replace("_", " ").split() if token]

    screen_labels: List[str] = []
    if _contains_any(lowered, ["settings", "device preferences", "system"]):
        screen_labels.append("settings")
    if _contains_any(lowered, ["language", "english", "spanish", "locale"]):
        screen_labels.append("language")
    if _contains_any(lowered, ["time zone", "timezone", "date & time", "date and time"]):
        screen_labels.append("timezone")
    if _contains_any(lowered, ["youtube", "shorts", "subscriptions"]):
        screen_labels.append("youtube_home")
    if _contains_any(lowered, ["pause", "up next", "play", "gear"]):
        screen_labels.append("video_player")
    if _contains_any(lowered, ["talkback", "accessibility"]):
        screen_labels.append("accessibility")
    if _contains_any(lowered, ["wifi", "network", "ethernet"]):
        screen_labels.append("network")
    if not screen_labels:
        screen_labels.append("unknown")

    focus_hints: List[str] = []
    if _contains_any(lowered, ["selected", "highlighted", "focused"]):
        focus_hints.append("focus_text_present")
    if _contains_any(lowered, ["ok", "continue", "confirm", "apply"]):
        focus_hints.append("commit_option_visible")
    if _contains_any(lowered, ["back", "cancel", "skip"]):
        focus_hints.append("escape_option_visible")

    return {
        "image_sha1": hashlib.sha1(image_bytes).hexdigest() if image_bytes else "",
        "image_bytes": len(image_bytes),
        "width": width,
        "height": height,
        "aspect_ratio": round((float(width) / float(height)), 3) if width and height else None,
        "ocr_length": len(text),
        "ocr_token_count": len(tokens),
        "screen_labels": screen_labels,
        "focus_hints": focus_hints,
        "settings_like": "settings" in screen_labels,
        "player_like": "video_player" in screen_labels,
        "language_visible": "language" in screen_labels,
        "timezone_visible": "timezone" in screen_labels,
        "accessibility_visible": "accessibility" in screen_labels,
        "network_visible": "network" in screen_labels,
    }
