from __future__ import annotations

import uuid

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .capture_manager import CaptureRegistry
from .config import AppSettings
from .logger import get_logger
from .webrtc_tracks import OpenCVCaptureTrack

logger = get_logger(__name__)


class OfferRequest(BaseModel):
    device_id: str
    sdp: str
    type: str


class OfferResponse(BaseModel):
    peer_id: str
    sdp: str
    type: str


class PeerConnectionManager:
    def __init__(self, registry: CaptureRegistry, settings: AppSettings) -> None:
        self._registry = registry
        self._settings = settings
        self._pcs: dict[str, RTCPeerConnection] = {}

    async def create_answer(self, offer: OfferRequest) -> OfferResponse:
        try:
            capture_manager = self._registry.get_manager(offer.device_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown device_id: {offer.device_id}") from exc

        peer_id = str(uuid.uuid4())
        peer_connection = RTCPeerConnection(configuration=self._rtc_configuration())
        self._pcs[peer_id] = peer_connection

        @peer_connection.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info(
                "peer connection state change",
                extra={
                    "event": "peer_connection_state_change",
                    "peer_id": peer_id,
                    "device_id": offer.device_id,
                    "connection_state": peer_connection.connectionState,
                },
            )
            if peer_connection.connectionState in {"failed", "closed", "disconnected"}:
                await self.close(peer_id)

        track = OpenCVCaptureTrack(capture_manager, stale_after_seconds=self._settings.frame_timeout_seconds)
        peer_connection.addTrack(track)
        await peer_connection.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)
        local_description = peer_connection.localDescription
        logger.info(
            "peer connection created",
            extra={
                "event": "peer_connection_created",
                "peer_id": peer_id,
                "device_id": offer.device_id,
            },
        )
        return OfferResponse(peer_id=peer_id, sdp=local_description.sdp, type=local_description.type)

    async def close(self, peer_id: str) -> None:
        peer_connection = self._pcs.pop(peer_id, None)
        if peer_connection is None:
            return
        await peer_connection.close()
        logger.info("peer connection closed", extra={"event": "peer_connection_closed", "peer_id": peer_id})

    async def close_all(self) -> None:
        for peer_id in list(self._pcs):
            await self.close(peer_id)

    def peer_count(self) -> int:
        return len(self._pcs)

    def _rtc_configuration(self) -> RTCConfiguration | None:
        if not self._settings.stun_urls:
            return None
        return RTCConfiguration(iceServers=[RTCIceServer(urls=self._settings.stun_urls)])


def build_signaling_router(*, registry: CaptureRegistry, peer_manager: PeerConnectionManager, settings: AppSettings) -> APIRouter:
    router = APIRouter(prefix="/signaling", tags=["webrtc-signaling"])

    @router.post("/offer", response_model=OfferResponse)
    async def offer(payload: OfferRequest) -> OfferResponse:
        return await peer_manager.create_answer(payload)

    @router.post("/close/{peer_id}")
    async def close(peer_id: str) -> dict:
        await peer_manager.close(peer_id)
        return {"closed": True, "peer_id": peer_id}

    @router.get("/peers")
    async def peers() -> dict:
        return {"peer_count": peer_manager.peer_count(), "status_interval_seconds": settings.status_interval_seconds}

    return router
