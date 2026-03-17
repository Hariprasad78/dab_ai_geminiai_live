"""HDMI/ALSA audio capture and streaming helpers."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    """Return True when ffmpeg binary is available in PATH."""
    return shutil.which("ffmpeg") is not None


def arecord_available() -> bool:
    """Return True when arecord binary is available in PATH."""
    return shutil.which("arecord") is not None


def ffmpeg_has_input_format(name: str) -> bool:
    """Return True if ffmpeg supports input format (demuxer) `name`."""
    if not ffmpeg_available():
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-demuxers"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    pattern = re.compile(rf"\b{name}\b", re.IGNORECASE)
    return bool(pattern.search(out))


def list_alsa_capture_devices() -> List[Dict[str, str]]:
    """Return ALSA capture devices from `arecord -l` output."""
    arecord = shutil.which("arecord")
    if not arecord:
        return []

    try:
        proc = subprocess.run(
            [arecord, "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    out = proc.stdout or ""
    devices: List[Dict[str, str]] = []

    # Example line:
    # card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
    pattern = re.compile(
        r"^card\s+(?P<card>\d+):\s*(?P<card_name>[^\[]+)\[(?P<card_desc>[^\]]+)\],\s*"
        r"device\s+(?P<device>\d+):\s*(?P<dev_name>[^\[]+)\[(?P<dev_desc>[^\]]+)\]",
        re.IGNORECASE,
    )

    for line in out.splitlines():
        m = pattern.search(line.strip())
        if not m:
            continue
        card = m.group("card")
        dev = m.group("device")
        devices.append(
            {
                "alsa_device": f"hw:{card},{dev}",
                "card": card,
                "device": dev,
                "card_name": m.group("card_name").strip(),
                "device_name": m.group("dev_name").strip(),
                "description": f"{m.group('card_desc').strip()} / {m.group('dev_desc').strip()}",
            }
        )

    return devices


def list_alsa_pcm_names(limit: int = 80) -> List[str]:
    """Return ALSA PCM endpoint names from `arecord -L` output."""
    arecord = shutil.which("arecord")
    if not arecord:
        return []
    try:
        proc = subprocess.run(
            [arecord, "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    names: List[str] = []
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line[:1].isspace():
            continue
        # first token on non-indented lines is PCM name
        token = stripped.split()[0]
        if token and token not in names:
            names.append(token)
        if len(names) >= int(limit):
            break
    return names


class HdmiAudioStreamSession:
    """Streams HDMI-capture-card audio via ffmpeg -> stdout (mp3)."""

    def __init__(
        self,
        device: str,
        input_format: str = "alsa",
        sample_rate: int = 48000,
        channels: int = 2,
        bitrate: str = "128k",
    ) -> None:
        self.device = device
        self.input_format = input_format
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.bitrate = bitrate
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._aux_proc: Optional[subprocess.Popen[bytes]] = None
        self._last_error: Optional[str] = None

    def start(self) -> bool:
        """Start ffmpeg process for audio streaming."""
        if self._proc is not None:
            return True

        if self.input_format == "arecord":
            return self._start_arecord_stream()

        if not ffmpeg_available():
            return False

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            self.input_format,
            "-i",
            self.device,
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "libmp3lame",
            "-b:a",
            self.bitrate,
            "-f",
            "mp3",
            "pipe:1",
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            return self._verify_process_started(self._proc)
        except Exception as exc:
            logger.error("Failed to start ffmpeg audio stream: %s", exc)
            self._last_error = str(exc)
            self._proc = None
            return False

    def _start_arecord_stream(self) -> bool:
        """Start arecord -> ffmpeg(mp3) pipeline for browser-friendly stream."""
        arecord = shutil.which("arecord")
        if not arecord or not ffmpeg_available():
            return False

        arecord_cmd = [
            arecord,
            "-q",
            "-D",
            self.device,
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-r",
            str(self.sample_rate),
            "-t",
            "raw",
            "-",
        ]

        ffmpeg_cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            self.bitrate,
            "-f",
            "mp3",
            "pipe:1",
        ]

        try:
            arec = subprocess.Popen(
                arecord_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            ffm = subprocess.Popen(
                ffmpeg_cmd,
                stdin=arec.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            if arec.stdout is not None:
                arec.stdout.close()

            self._aux_proc = arec
            self._proc = ffm
            ok = self._verify_process_started(arec) and self._verify_process_started(ffm)
            if not ok:
                self.close()
            return ok
        except Exception as exc:
            logger.error("Failed to start arecord audio stream: %s", exc)
            self._last_error = str(exc)
            self._proc = None
            self._aux_proc = None
            return False

    def _verify_process_started(self, proc: subprocess.Popen[bytes]) -> bool:
        """Detect immediate process failure and keep readable error text."""
        time.sleep(0.15)
        rc = proc.poll()
        if rc is None:
            return True
        err = b""
        try:
            if proc.stderr is not None:
                err = proc.stderr.read(2048)
        except Exception:
            pass
        msg = (err.decode("utf-8", errors="ignore").strip() or f"process exited with code {rc}")
        self._last_error = msg
        return False

    def read_chunk(self, size: int = 4096) -> bytes:
        """Read one audio chunk from ffmpeg stdout."""
        if self._proc is None or self._proc.stdout is None:
            return b""
        try:
            return self._proc.stdout.read(size)
        except Exception:
            return b""

    def close(self) -> None:
        """Terminate ffmpeg process."""
        p = self._proc
        a = self._aux_proc
        self._proc = None
        self._aux_proc = None
        if p is None:
            if a is None:
                return
        for proc in [p, a]:
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error


def resolve_audio_input(preferred_format: str = "auto", configured_device: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Resolve `(format, device)` pair compatible with this host ffmpeg build."""
    pf = (preferred_format or "auto").strip().lower()
    configured_device = (configured_device or "").strip()

    # Explicit format mode first.
    if pf in {"alsa", "pulse", "arecord"}:
        if pf == "arecord":
            if not arecord_available():
                return None, None
            if configured_device:
                return "arecord", configured_device
            devs = list_alsa_capture_devices()
            if devs:
                return "arecord", str(devs[0].get("alsa_device") or "")
            pcm_names = set(list_alsa_pcm_names())
            for fallback in ("default", "pulse"):
                if fallback in pcm_names:
                    return "arecord", fallback
            return None, None
        if not ffmpeg_has_input_format(pf):
            return None, None
        if configured_device:
            return pf, configured_device
        if pf == "pulse":
            return "pulse", "default"
        devs = list_alsa_capture_devices()
        if devs:
            return "alsa", str(devs[0].get("alsa_device") or "")
        return None, None

    # auto mode: ALSA first, then PulseAudio.
    if ffmpeg_has_input_format("alsa"):
        if configured_device:
            return "alsa", configured_device
        devs = list_alsa_capture_devices()
        if devs:
            return "alsa", str(devs[0].get("alsa_device") or "")

    if ffmpeg_has_input_format("pulse"):
        return "pulse", configured_device or "default"

    # Last fallback: arecord WAV stream/capture
    if arecord_available():
        if configured_device:
            return "arecord", configured_device
        devs = list_alsa_capture_devices()
        if devs:
            return "arecord", str(devs[0].get("alsa_device") or "")
        pcm_names = set(list_alsa_pcm_names())
        for fallback in ("default", "pulse"):
            if fallback in pcm_names:
                return "arecord", fallback

    return None, None
