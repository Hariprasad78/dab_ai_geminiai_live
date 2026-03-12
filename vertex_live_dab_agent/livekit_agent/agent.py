"""LiveKit agent entrypoint for Vertex AI Live integration."""
import asyncio
import logging

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.session.manager import SessionManager

logger = logging.getLogger(__name__)


async def run_agent() -> None:
    """
    Main LiveKit agent entrypoint.

    TODO: Replace with real LiveKit agent SDK integration.
    This is an adapter interface. Integrate with livekit-agents SDK
    when LiveKit infrastructure is available.

    Required env vars:
        LIVEKIT_URL
        LIVEKIT_API_KEY
        LIVEKIT_API_SECRET
        GOOGLE_APPLICATION_CREDENTIALS
        GOOGLE_CLOUD_PROJECT
        GOOGLE_CLOUD_LOCATION
    """
    config = get_config()

    if not config.livekit_url:
        logger.warning("LIVEKIT_URL not set - running in stub mode")

    session_manager = SessionManager()

    logger.info("LiveKit agent starting (stub mode)")
    logger.info("Vertex model: %s", config.vertex_live_model)
    logger.info("Project: %s, Location: %s", config.google_cloud_project, config.google_cloud_location)

    # TODO: Initialize LiveKit worker
    # from livekit.agents import WorkerOptions, cli
    # from livekit.plugins import google
    #
    # async def entrypoint(ctx):
    #     session = session_manager.start_session(ctx.room.name)
    #     model = google.beta.realtime.RealtimeModel(
    #         model=config.vertex_live_model,
    #         project=config.google_cloud_project,
    #         location=config.google_cloud_location,
    #     )
    #     agent = multimodal.MultimodalAgent(model=model)
    #     agent.start(ctx.room)

    logger.info("LiveKit agent stub running. Waiting...")
    try:
        while True:
            await asyncio.sleep(60)
            session_manager.cleanup_expired()
    except asyncio.CancelledError:
        logger.info("LiveKit agent stopped")


if __name__ == "__main__":
    from vertex_live_dab_agent.artifacts.logger import setup_logging
    config = get_config()
    setup_logging(config.log_level)
    asyncio.run(run_agent())
