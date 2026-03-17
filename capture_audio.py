#!/usr/bin/env python3
"""Standalone HDMI-audio capture helper for ALSA capture-card input.

Examples:
  python capture_audio.py --list
  python capture_audio.py --device hw:1,0 --seconds 10 --out hdmi_audio.wav
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from vertex_live_dab_agent.capture.hdmi_audio import list_alsa_capture_devices, resolve_audio_input


def _cmd_list() -> int:
    devices = list_alsa_capture_devices()
    if not devices:
        print("No ALSA capture devices found (arecord -l)")
        return 1
    print("ALSA capture devices:")
    for d in devices:
        print(f"  {d['alsa_device']:<10}  {d['card_name']} / {d['device_name']}  ({d['description']})")
    return 0


def _cmd_record(
    device: str,
    seconds: int,
    out_file: str,
    sample_rate: int,
    channels: int,
    input_format: str,
) -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Install ffmpeg and try again.")
        return 1

    resolved_format, resolved_device = resolve_audio_input(
        preferred_format=input_format,
        configured_device=device,
    )
    if not resolved_format or not resolved_device:
        print("[ERROR] No supported ffmpeg input format found (alsa/pulse) or no capture device available.")
        print("        Try: python capture_audio.py --list")
        print("        Or force pulse: python capture_audio.py --format pulse --device default")
        print("        If `sudo arecord` works but user mode fails, add user to audio group:")
        print("        sudo usermod -aG audio $USER && newgrp audio")
        print("        Then use: python capture_audio.py --format arecord --device hw:<card>,<device>")
        return 1

    print(f"[INFO] Recording {seconds}s from {resolved_format}:{resolved_device} -> {out_file}")

    if resolved_format == "arecord":
        arecord = shutil.which("arecord")
        if not arecord:
            print("[ERROR] arecord not found")
            return 1
        cmd = [
            arecord,
            "-q",
            "-D",
            resolved_device,
            "-f",
            "S16_LE",
            "-c",
            str(channels),
            "-r",
            str(sample_rate),
            "-d",
            str(seconds),
            "-t",
            "wav",
            out_file,
        ]
        try:
            subprocess.run(cmd, check=True)
            print("[INFO] Done")
            return 0
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] arecord failed: {exc}")
            print("        Check device value from: arecord -l")
            print("        If only sudo works, run: sudo usermod -aG audio $USER && newgrp audio")
            return 1

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        resolved_format,
        "-i",
        resolved_device,
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-t",
        str(seconds),
        out_file,
    ]
    try:
        subprocess.run(cmd, check=True)
        print("[INFO] Done")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] ffmpeg failed: {exc}")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="HDMI/ALSA audio capture utility")
    ap.add_argument("--list", action="store_true", help="List ALSA capture devices and exit")
    ap.add_argument("--device", default="", help="ALSA device (e.g. hw:1,0)")
    ap.add_argument("--seconds", type=int, default=10, help="Record duration in seconds")
    ap.add_argument("--out", default="hdmi_audio.wav", help="Output audio file")
    ap.add_argument("--format", default="auto", choices=["auto", "alsa", "pulse", "arecord"], help="Audio input format")
    ap.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")
    ap.add_argument("--channels", type=int, default=2, help="Channels")
    args = ap.parse_args()

    if args.list:
        return _cmd_list()

    device = args.device.strip()
    if not device:
        fmt, dev = resolve_audio_input(preferred_format=args.format, configured_device="")
        if not fmt or not dev:
            print("No usable audio input found. Run with --list or set --format pulse --device default")
            return 1
        device = dev
        print(f"[INFO] Auto-selected {fmt}:{device}")

    return _cmd_record(device, args.seconds, args.out, args.sample_rate, args.channels, args.format)


if __name__ == "__main__":
    sys.exit(main())
