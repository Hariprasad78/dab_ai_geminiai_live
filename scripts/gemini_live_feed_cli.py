#!/usr/bin/env python3
"""Interactive Gemini live-feed experiment CLI.

This is a lightweight experiment script for asking Gemini questions about the
current HDMI/camera live feed.

It does not start a true bidirectional Gemini Live session. Instead, it
captures a fresh live frame for each CLI question and sends that frame to the
configured Gemini model using the repo's existing client wiring.

Usage:
  python3 scripts/gemini_live_feed_cli.py
  python3 scripts/gemini_live_feed_cli.py --model gemini-3.1-flash-live-preview
  python3 scripts/gemini_live_feed_cli.py --source hdmi-capture --device /dev/video4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import os
import sys
import time
from typing import Optional

import httpx

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.client import create_dab_client
from vertex_live_dab_agent.capture.capture import ScreenCapture
from vertex_live_dab_agent.capture.hdmi_capture import list_hdmi_devices
from vertex_live_dab_agent.planner.vertex_client import VertexPlannerClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Gemini about the current live HDMI/camera feed")
    parser.add_argument(
        "--model",
        default="",
        help="Gemini model to use. Defaults to VERTEX_LIVE_MODEL, then VERTEX_PLANNER_MODEL.",
    )
    parser.add_argument(
        "--project",
        default="",
        help="Google Cloud project override. Defaults to GOOGLE_CLOUD_PROJECT / repo config.",
    )
    parser.add_argument(
        "--location",
        default="",
        help="Google Cloud location override. Defaults to GOOGLE_CLOUD_LOCATION / repo config.",
    )
    parser.add_argument(
        "--source",
        default="hdmi-capture",
        help="Capture source: auto, hdmi-capture, camera-capture, or dab.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional explicit /dev/video* device to use.",
    )
    parser.add_argument(
        "--preferred-kind",
        default="hdmi",
        help="Capture kind preference: auto, hdmi, or camera.",
    )
    parser.add_argument(
        "--system",
        default=(
            "You are observing a live TV/camera feed for debugging and QA. "
            "Answer only from the attached frame. Be concrete, visual, and concise. "
            "If something is uncertain, say that it is uncertain."
        ),
        help="System instruction prepended to each question.",
    )
    parser.add_argument(
        "--capture-timeout",
        type=float,
        default=4.0,
        help="How long to keep retrying for a fresh live frame before giving up.",
    )
    parser.add_argument(
        "--transport",
        default="direct",
        choices=["direct", "webrtc"],
        help="Frame source transport. 'direct' reads the local capture device, 'webrtc' receives one frame from the backend WebRTC stream.",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="Backend API base URL used for WebRTC mode.",
    )
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> VertexPlannerClient:
    config = get_config()
    project = str(args.project or config.google_cloud_project or "").strip()
    location = str(args.location or config.google_cloud_location or "").strip()
    model = str(args.model or config.vertex_live_model or config.vertex_planner_model or "").strip()
    api_key = str(getattr(config, "google_api_key", "") or "").strip() or None
    return VertexPlannerClient(
        project=project,
        location=location,
        model=model,
        api_key=api_key,
    )


def build_capture(args: argparse.Namespace) -> ScreenCapture:
    capture = ScreenCapture(create_dab_client())
    capture.set_capture_preference(
        source=str(args.source or "hdmi-capture").strip(),
        device=str(args.device or "").strip() or None,
        preferred_kind=str(args.preferred_kind or "hdmi").strip(),
        persist=False,
    )
    capture.ensure_hdmi_session(force=True)
    return capture


async def capture_live_frame_b64(capture: ScreenCapture) -> tuple[Optional[str], str]:
    result = await capture.capture_live_stream_frame()
    return result.image_b64, str(result.source or "unknown")


async def capture_webrtc_frame_b64(api_base: str, timeout_s: float) -> tuple[Optional[str], str]:
    try:
        from aiortc import RTCPeerConnection, RTCSessionDescription
    except Exception as exc:
        raise RuntimeError(f"aiortc is required for --transport webrtc: {exc}") from exc

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"opencv-python-headless is required for --transport webrtc: {exc}") from exc

    pc = RTCPeerConnection()
    frame_future: asyncio.Future = asyncio.get_running_loop().create_future()

    @pc.on("track")
    def _on_track(track) -> None:
        if getattr(track, "kind", "") != "video":
            return

        async def _recv_one() -> None:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=max(1.0, float(timeout_s)))
                array = frame.to_ndarray(format="bgr24")
                ok, encoded = cv2.imencode(".png", array)
                if not ok:
                    raise RuntimeError("Failed to encode WebRTC frame as PNG")
                image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                if not frame_future.done():
                    frame_future.set_result(image_b64)
            except Exception as exc:
                if not frame_future.done():
                    frame_future.set_exception(exc)

        asyncio.create_task(_recv_one())

    peer_id = ""
    try:
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        async with httpx.AsyncClient(timeout=max(10.0, float(timeout_s) + 6.0)) as client:
            response = await client.post(
                f"{api_base.rstrip('/')}/webrtc/offer",
                json={
                    "sdp": str(pc.localDescription.sdp or ""),
                    "type": str(pc.localDescription.type or "offer"),
                },
            )
            response.raise_for_status()
            answer = response.json()
            peer_id = str(answer.get("peer_id") or "").strip()

        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=str(answer.get("sdp") or ""),
                type=str(answer.get("type") or "answer"),
            )
        )
        image_b64 = await asyncio.wait_for(frame_future, timeout=max(1.0, float(timeout_s) + 2.0))
        return str(image_b64 or ""), "webrtc"
    finally:
        if peer_id:
            with contextlib.suppress(Exception):
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(f"{api_base.rstrip('/')}/webrtc/close/{peer_id}")
        with contextlib.suppress(Exception):
            await pc.close()


def render_capture_status(capture: ScreenCapture) -> str:
    status = capture.capture_source_status()
    configured_source = str(status.get("configured_source") or "unknown")
    hdmi_available = bool(status.get("hdmi_available"))
    hdmi_device = str(status.get("hdmi_device") or "").strip() or "none"
    selected = str(status.get("selected_video_device") or "").strip() or "auto"
    preferred_kind = str(status.get("preferred_video_kind") or "auto")
    last_error = str(status.get("hdmi_last_error") or "").strip() or "none"
    return (
        f"configured_source={configured_source} "
        f"preferred_kind={preferred_kind} "
        f"selected_video_device={selected} "
        f"active_hdmi_device={hdmi_device} "
        f"hdmi_available={'yes' if hdmi_available else 'no'} "
        f"hdmi_last_error={last_error}"
    )


def probe_working_video_devices() -> list[str]:
    config = get_config()
    try:
        devices = list_hdmi_devices(
            fourcc=str(config.hdmi_capture_fourcc or "MJPG"),
            width=int(config.hdmi_capture_width or 1280),
            height=int(config.hdmi_capture_height or 720),
            fps=float(config.hdmi_capture_fps or 30.0),
        )
    except Exception:
        return []
    return [str(item.get("device") or "").strip() for item in devices if str(item.get("device") or "").strip()]


def try_switch_capture_device(capture: ScreenCapture, device: str, preferred_kind: str) -> bool:
    try:
        capture.set_capture_preference(
            source="hdmi-capture",
            device=device,
            preferred_kind=preferred_kind,
            persist=False,
        )
        capture.ensure_hdmi_session(force=True)
        return True
    except Exception:
        return False


async def capture_frame_with_retries(
    capture: ScreenCapture,
    *,
    timeout_s: float,
    preferred_kind: str,
    transport: str,
    api_base: str,
) -> tuple[Optional[str], str]:
    if transport == "webrtc":
        try:
            return await capture_webrtc_frame_b64(api_base, timeout_s)
        except Exception:
            return None, "webrtc-error"

    deadline = time.monotonic() + max(0.5, float(timeout_s or 4.0))
    last_source = "error"
    while time.monotonic() < deadline:
        image_b64, source = await capture_live_frame_b64(capture)
        last_source = source
        if image_b64:
            return image_b64, source
        await asyncio.sleep(0.25)
        capture.ensure_hdmi_session(force=True)
    for device in probe_working_video_devices():
        if not try_switch_capture_device(capture, device, preferred_kind):
            continue
        image_b64, source = await capture_live_frame_b64(capture)
        last_source = source
        if image_b64:
            print(f"[OK] Auto-switched capture to working device: {device}")
            return image_b64, source
    return None, last_source


def compose_prompt(system_prompt: str, user_prompt: str, source: str) -> str:
    return "\n\n".join(
        [
            system_prompt.strip(),
            f"Live visual source: {source or 'unknown'}",
            "Task: answer the user's question about what is happening in this live frame.",
            f"User question: {user_prompt.strip()}",
        ]
    )


async def interactive_loop(args: argparse.Namespace) -> int:
    try:
        client = build_client(args)
    except Exception as exc:
        print(f"[FAIL] Gemini client setup failed: {exc}")
        return 1

    try:
        capture = build_capture(args)
    except Exception as exc:
        print(f"[FAIL] Capture setup failed: {exc}")
        return 1

    model_name = str(args.model or get_config().vertex_live_model or get_config().vertex_planner_model or "").strip()
    print("=== Gemini Live Feed CLI ===")
    print(f"model={model_name or '(unknown)'}")
    print(f"source={args.source} preferred_kind={args.preferred_kind} device={args.device or 'auto'}")
    print(f"transport={args.transport} api_base={args.api_base}")
    print("Type a question and press Enter.")
    print("Commands: /refresh, /status, /probe, /quit")
    print(f"capture_status: {render_capture_status(capture)}")

    session_id = "gemini-live-feed-cli"
    while True:
        try:
            question = input("\nask> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not question:
            continue
        if question.lower() in {"/quit", "quit", "exit"}:
            break
        if question.lower() == "/status":
            print(render_capture_status(capture))
            continue
        if question.lower() == "/probe":
            devices = probe_working_video_devices()
            if devices:
                print("working_devices:")
                for device in devices:
                    print(f"  - {device}")
            else:
                print("[WARN] No working video devices probed")
            continue
        if question.lower() == "/refresh":
            image_b64, source = await capture_frame_with_retries(
                capture,
                timeout_s=args.capture_timeout,
                preferred_kind=args.preferred_kind,
                transport=args.transport,
                api_base=args.api_base,
            )
            if image_b64:
                print(f"[OK] Captured fresh frame from {source}")
            else:
                print(f"[WARN] No live frame available from {source}")
                print(f"        {render_capture_status(capture)}")
            continue

        image_b64, source = await capture_frame_with_retries(
            capture,
            timeout_s=args.capture_timeout,
            preferred_kind=args.preferred_kind,
            transport=args.transport,
            api_base=args.api_base,
        )
        if not image_b64:
            print(f"[WARN] No live frame available from {source}. Check HDMI/camera capture.")
            print(f"        {render_capture_status(capture)}")
            continue

        prompt = compose_prompt(args.system, question, source)
        try:
            answer = await client.generate_content(
                prompt,
                screenshot_b64=image_b64,
                session_id=session_id,
            )
        except Exception as exc:
            print(f"[FAIL] Gemini request failed: {exc}")
            continue

        print(f"\nGemini> {str(answer or '').strip() or '(empty response)'}")

    print("bye")
    return 0


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    return asyncio.run(interactive_loop(args))


if __name__ == "__main__":
    sys.exit(main())
