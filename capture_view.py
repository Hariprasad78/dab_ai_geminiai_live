#!/usr/bin/env python3
"""Standalone HDMI-to-USB capture card viewer for DAB Android Automation.

Displays a live preview of the HDMI signal received through a V4L2 USB capture
card directly on the host machine screen.

Usage examples::

    python capture_view.py --list
    python capture_view.py --device /dev/adt4_cam
    python capture_view.py --device /dev/adt4_cam --width 1280 --height 720 --fps 60
    python capture_view.py --device /dev/adt4_cam --fourcc YUYV

Press **q** or **ESC** in the preview window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import cv2  # type: ignore
except ImportError as exc:
    sys.exit(
        f"OpenCV is required but not installed: {exc}\n"
        "Install it with: pip install opencv-python"
    )

from vertex_live_dab_agent.capture.hdmi_capture import HdmiCaptureSession, list_hdmi_devices
from vertex_live_dab_agent.capture.camera_devices import get_camera_path


def _cmd_list(args: argparse.Namespace) -> int:
    print("Probing video devices (this may take a few seconds)...")
    devices = list_hdmi_devices(
        fourcc=args.fourcc,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    if not devices:
        print(
            "No working capture devices found.\n"
            "Check that:\n"
            "  - The HDMI capture card is plugged in.\n"
            "  - The HDMI source is powered ON and outputting a signal.\n"
            "  - Your user has read/write access to camera nodes (add to 'video' group)."
        )
        return 1

    print("Working capture devices:")
    for d in devices:
        print(f"  {d['device']}  ({d['width']:.0f}x{d['height']:.0f} @ {d['fps']:.2f} fps)")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    device = args.device
    if device is None:
        configured = get_camera_path("adt4")
        if configured:
            device = configured
            print(f"No --device specified; using configured ADT-4 camera: {device}")

    if device is None:
        print("No --device specified; auto-detecting...")
        found = list_hdmi_devices(
            fourcc=args.fourcc,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        if not found:
            print(
                "Auto-detect failed: no working capture devices found.\n"
                "Try: python capture_view.py --list"
            )
            return 1
        device = str(found[0]["device"])
        print(f"[INFO] Auto-selected: {device}")

    sess = HdmiCaptureSession(
        device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
    )

    print(
        f"[INFO] Opening {device} "
        f"({args.width}x{args.height} @ {args.fps:.0f}fps, FOURCC={args.fourcc})..."
    )
    if not sess.open():
        print(
            f"Failed to open capture device: {device}\n"
            "Tips:\n"
            "  - Make sure the HDMI source is powered ON.\n"
            "  - Try --fourcc YUYV if MJPG does not work.\n"
            "  - Run with --list to see available devices."
        )
        return 1

    info = sess.device_info()
    if info:
        print(f"[INFO] Opened {device}: {info['width']:.0f}x{info['height']:.0f} @ {info['fps']:.2f} fps")
    else:
        print(f"[INFO] Opened {device}")
    print("[INFO] Preview window is open. Press 'q' or ESC inside the window to quit.")

    try:
        while True:
            frame = sess.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            cv2.imshow("HDMI Capture Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        sess.close()
        cv2.destroyAllWindows()
        print("[INFO] Capture session closed.")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HDMI-to-USB capture card live viewer for DAB Android Automation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List video devices that are delivering frames and exit.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="V4L2 device path (e.g. /dev/adt4_cam). Auto-detected when omitted.",
    )
    ap.add_argument("--width", type=int, default=1280, help="Requested capture width  (default 1280).")
    ap.add_argument("--height", type=int, default=720, help="Requested capture height (default 720).")
    ap.add_argument("--fps", type=float, default=30.0, help="Requested capture FPS    (default 30).")
    ap.add_argument("--fourcc", default="MJPG", help="FOURCC codec: MJPG (default) or YUYV.")
    args = ap.parse_args()

    if args.list:
        return _cmd_list(args)
    return _cmd_view(args)


if __name__ == "__main__":
    sys.exit(main())
