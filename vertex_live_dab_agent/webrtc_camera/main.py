from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .capture_manager import CaptureRegistry
from .config import AppSettings, load_settings
from .logger import configure_logging
from .signaling import PeerConnectionManager, build_signaling_router

_BASE_DIR = Path(__file__).parent
_STATIC_DIR = _BASE_DIR / "static"
_TEMPLATE_DIR = _BASE_DIR / "templates"


class WebRTCAppState:
    def __init__(self, settings: AppSettings, registry: CaptureRegistry, peer_manager: PeerConnectionManager) -> None:
        self.settings = settings
        self.registry = registry
        self.peer_manager = peer_manager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state: WebRTCAppState = app.state.webrtc_camera_state
    state.registry.start()
    try:
        yield
    finally:
        await state.peer_manager.close_all()
        state.registry.stop()


def create_app(settings: AppSettings | None = None) -> FastAPI:
    resolved_settings = settings or load_settings()
    configure_logging(resolved_settings.log_level)
    registry = CaptureRegistry(resolved_settings)
    peer_manager = PeerConnectionManager(registry=registry, settings=resolved_settings)
    app = FastAPI(title="WebRTC Camera Preview Service", version="1.0.0", lifespan=_lifespan)
    app.state.webrtc_camera_state = WebRTCAppState(
        settings=resolved_settings,
        registry=registry,
        peer_manager=peer_manager,
    )

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    app.include_router(build_signaling_router(registry=registry, peer_manager=peer_manager, settings=resolved_settings), prefix="/api")

    def _status_payload() -> dict:
        payload = registry.status()
        payload["peer_count"] = peer_manager.peer_count()
        payload["service"] = "webrtc_camera"
        payload["http_port"] = resolved_settings.port
        return payload

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "page_title": "WebRTC Camera Preview",
                "status_interval_ms": int(max(0.2, resolved_settings.status_interval_seconds) * 1000.0),
            },
        )

    @app.head("/", include_in_schema=False)
    async def index_head() -> Response:
        return Response(status_code=200)

    @app.get("/api/status")
    async def status() -> dict:
        return _status_payload()

    @app.get("/api/devices")
    async def devices() -> dict:
        return {"devices": registry.list_devices()}

    @app.get("/health")
    async def health() -> dict:
        payload = _status_payload()
        return {
            "status": payload["status"],
            "service": payload["service"],
            "http_port": payload["http_port"],
            "peer_count": payload["peer_count"],
            "device_count": payload["device_count"],
            "available_device_count": payload["available_device_count"],
            "failed_required_count": payload["failed_required_count"],
        }

    @app.get("/stream/status")
    async def stream_status() -> dict:
        return _status_payload()

    @app.get("/capture/devices")
    async def capture_devices() -> dict:
        return {
            "service": "webrtc_camera",
            "devices": registry.list_devices(),
        }

    @app.get("/capture/source")
    async def capture_source() -> dict:
        return {
            "service": "webrtc_camera",
            "transport": "webrtc",
            "http_port": resolved_settings.port,
            "devices": [
                {
                    "device_id": item["device_id"],
                    "kind": item["kind"],
                    "locator": item["locator"],
                    "device_path": item["device_path"],
                    "state": item["state"],
                    "frame_available": item["frame_available"],
                }
                for item in registry.list_devices()
            ],
        }

    @app.websocket("/ws/status")
    async def status_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(_status_payload())
                await asyncio.sleep(max(0.2, resolved_settings.status_interval_seconds))
        except WebSocketDisconnect:
            return

    return app


app = create_app()
