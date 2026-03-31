from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "vertex_live_dab_agent.webrtc_camera.main:app",
        host=os.environ.get("WEBRTC_CAMERA_HOST", "0.0.0.0"),
        port=int(os.environ.get("WEBRTC_CAMERA_PORT", "8080")),
        reload=False,
        log_level=os.environ.get("WEBRTC_CAMERA_LOG_LEVEL", "info").lower(),
    )
