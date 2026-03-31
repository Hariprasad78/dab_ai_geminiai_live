from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from .device_registry import DeviceRegistry
from .logger import configure_logging

_STATIC_DIR = Path(__file__).parent / "static"
_PLACEHOLDER_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDxmiiigD//2Q=="
)


class MultiCameraAppState:
    def __init__(self, registry: DeviceRegistry) -> None:
        self.registry = registry


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state: MultiCameraAppState = app.state.multi_camera_state
    state.registry.start()
    try:
        yield
    finally:
        state.registry.stop()


def create_app(registry: Optional[DeviceRegistry] = None) -> FastAPI:
    configure_logging(os.environ.get("MULTI_CAMERA_LOG_LEVEL", "INFO"))
    resolved_registry = registry or DeviceRegistry.from_env()
    app = FastAPI(title="Multi Camera Preview Service", version="1.0.0", lifespan=_lifespan)
    app.state.multi_camera_state = MultiCameraAppState(resolved_registry)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/health")
    async def health() -> dict:
        return app.state.multi_camera_state.registry.health()

    @app.get("/devices")
    async def devices() -> dict:
        return {"devices": app.state.multi_camera_state.registry.list_devices()}

    @app.get("/stream-status")
    async def stream_status() -> dict:
        return app.state.multi_camera_state.registry.stream_status()

    @app.get("/snapshot/{device_id}")
    async def snapshot(device_id: str) -> Response:
        try:
            payload = app.state.multi_camera_state.registry.latest_frame(device_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown device_id: {device_id}") from exc
        is_placeholder = payload is None
        return Response(
            content=payload if payload is not None else _PLACEHOLDER_JPEG,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Placeholder-Frame": "true" if is_placeholder else "false",
                "X-Device-Id": device_id,
            },
        )

    @app.websocket("/ws/status")
    async def ws_status(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(app.state.multi_camera_state.registry.stream_status())
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
