from __future__ import annotations

from fractions import Fraction
import asyncio
import time

from aiortc import MediaStreamTrack
from av import VideoFrame
import numpy as np

from .capture_manager import CaptureManager
from .logger import get_logger

VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = Fraction(1, VIDEO_CLOCK_RATE)

logger = get_logger(__name__)


class OpenCVCaptureTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, manager: CaptureManager, *, stale_after_seconds: float = 5.0) -> None:
        super().__init__()
        self._manager = manager
        self._stale_after_seconds = max(0.5, float(stale_after_seconds))
        self._start_time: float | None = None
        self._timestamp = 0
        self._frame_period = 1.0 / max(1.0, float(manager.spec.fps))
        self._placeholder = np.zeros((manager.spec.height, manager.spec.width, 3), dtype=np.uint8)
        self._last_sequence: int | None = None

    async def recv(self) -> VideoFrame:
        pts, time_base = await self._next_timestamp()
        latest = self._manager.latest_frame()
        use_placeholder = latest is None
        if latest is not None:
            frame_age = max(0.0, time.monotonic() - latest.captured_monotonic)
            use_placeholder = frame_age > self._stale_after_seconds
        image = self._placeholder if use_placeholder else latest.image
        if latest is not None and latest.sequence != self._last_sequence:
            self._last_sequence = latest.sequence
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    async def _next_timestamp(self) -> tuple[int, Fraction]:
        if self.readyState != "live":
            raise RuntimeError("track is not live")
        if self._start_time is None:
            self._start_time = time.time()
            self._timestamp = 0
        else:
            self._timestamp += int(self._frame_period * VIDEO_CLOCK_RATE)
            target = self._start_time + (self._timestamp / VIDEO_CLOCK_RATE)
            delay = target - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
        return self._timestamp, VIDEO_TIME_BASE
