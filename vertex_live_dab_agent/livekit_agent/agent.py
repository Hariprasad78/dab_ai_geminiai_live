"""LiveKit + Vertex AI Live agent entrypoint.

Architecture
------------
The agent separates two concerns:

1. **Operator / live interaction layer** (this module) — handles LiveKit room
   lifecycle, connects to the real-time multimodal session, manages
   session state, and routes operator instructions to the planner.

2. **Deterministic planner / tool execution layer**
   (:mod:`vertex_live_dab_agent.planner`) — takes structured observations and
   produces exactly one validated :class:`~.planner.schemas.PlannedAction` per
   step. Never receives raw audio/video; always works in text mode.

Media pipeline note
-------------------
To enable full audio/video streaming via LiveKit you will need to add the
optional ``livekit`` extra (``pip install -e ".[livekit]"``) and the frontend
must publish a video track so the agent can observe the screen in real-time.
See comments marked ``# MEDIA_PIPELINE`` below for each integration point.

The module is designed so that the **text-mode path always works** even when
the full audio/video pipeline is not enabled — useful for CI, testing, and
headless operation.

Required environment variables
-------------------------------
Vertex AI (mandatory for AI planning):
    GOOGLE_CLOUD_PROJECT
    GOOGLE_CLOUD_LOCATION
    GOOGLE_APPLICATION_CREDENTIALS  (path to service-account JSON, or use
                                     Application Default Credentials)

LiveKit (optional — agent runs in stub mode without these):
    LIVEKIT_URL
    LIVEKIT_API_KEY
    LIVEKIT_API_SECRET
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from vertex_live_dab_agent.artifacts.logger import setup_logging
from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.planner.planner import Planner
from vertex_live_dab_agent.planner.schemas import ActionType, PlannedAction
from vertex_live_dab_agent.session.manager import SessionManager, SessionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credential / config validation
# ---------------------------------------------------------------------------


class AgentConfigError(RuntimeError):
    """Raised on startup when required credentials are missing."""


def _validate_config() -> None:
    """Check for required credentials and raise :class:`AgentConfigError` if absent.

    This is called early in :func:`run_agent` so failures are loud and clear
    rather than showing up as a cryptic runtime error deep inside an async loop.
    """
    config = get_config()
    missing: List[str] = []

    if not config.google_cloud_project:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if not config.google_cloud_location:
        missing.append("GOOGLE_CLOUD_LOCATION")

    if missing:
        raise AgentConfigError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them before starting the agent."
        )


# ---------------------------------------------------------------------------
# Operator session — wraps a single LiveKit room connection
# ---------------------------------------------------------------------------


@dataclass
class OperatorSession:
    """State for a single operator interaction session.

    The *operator* is the human (or automated script) communicating with the
    agent in real-time via LiveKit.  Their instructions drive the planner.

    Attributes
    ----------
    session_state:
        Shared :class:`~.session.manager.SessionState` from the session
        manager, used to record conversation history and check for expiry.
    room_name:
        LiveKit room name (empty string in stub/text mode).
    pending_goal:
        Most-recent goal text received from the operator that has not yet been
        dispatched to the planner.
    planned_actions:
        Ordered list of all :class:`~.planner.schemas.PlannedAction` objects
        produced by the planner during this session.
    """

    session_state: SessionState
    room_name: str = ""
    pending_goal: Optional[str] = None
    planned_actions: List[PlannedAction] = field(default_factory=list)

    def receive_operator_message(self, text: str) -> None:
        """Record an inbound operator instruction and set it as the pending goal."""
        self.session_state.record_message("operator", text)
        self.pending_goal = text
        logger.info("Operator message received: %r (session=%s)", text, self.session_state.session_id)

    def record_agent_response(self, text: str) -> None:
        """Record an outbound agent message in the conversation history."""
        self.session_state.record_message("agent", text)
        logger.info("Agent response: %r (session=%s)", text, self.session_state.session_id)

    def add_planned_action(self, action: PlannedAction) -> None:
        """Store a planned action and log it."""
        self.planned_actions.append(action)
        logger.info(
            "Action planned: action=%s confidence=%.2f reason=%r (session=%s)",
            action.action, action.confidence, action.reason,
            self.session_state.session_id,
        )


# ---------------------------------------------------------------------------
# LiveKit worker factory
# ---------------------------------------------------------------------------


def _build_livekit_worker(config, session_manager: SessionManager, planner: Planner):
    """Return a configured LiveKit WorkerOptions, or None if livekit-agents is unavailable.

    This function attempts to import ``livekit.agents`` at runtime so that the
    module can be imported (and tested) even when the ``livekit`` optional
    dependency is not installed.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from livekit.agents import WorkerOptions, cli  # type: ignore[import]
        from livekit.agents import multimodal           # type: ignore[import]
    except ImportError:
        logger.warning(
            "livekit-agents not installed. Install with: pip install -e '.[livekit]'. "
            "Running in stub/text mode."
        )
        return None

    try:
        # Vertex AI Gemini live model — requires google-cloud-aiplatform + auth
        # MEDIA_PIPELINE: The RealtimeModel will stream audio/video to/from the
        # room when the frontend publishes a track.  Without a video track the
        # model operates on text only.
        from livekit.plugins import google as lk_google  # type: ignore[import]

        async def entrypoint(ctx):  # type: ignore[no-untyped-def]
            """Called by the LiveKit worker for every new room connection."""
            room_name = ctx.room.name
            session_state = session_manager.start_session(room_name)
            op_session = OperatorSession(session_state=session_state, room_name=room_name)

            logger.info("LiveKit room connected: %s", room_name)

            # MEDIA_PIPELINE: Subscribe to the operator's video/audio tracks here.
            # Example (add when frontend sends tracks):
            #   @ctx.room.on("track_subscribed")
            #   async def on_track(track, publication, participant):
            #       if track.kind == "video":
            #           # forward frames to the model or screenshot buffer
            #           pass

            # Vertex AI Gemini live session
            model = lk_google.beta.realtime.RealtimeModel(  # type: ignore[attr-defined]
                model=config.vertex_live_model,
                project=config.google_cloud_project,
                location=config.google_cloud_location,
            )
            # MEDIA_PIPELINE: MultimodalAgent handles audio/video I/O with LiveKit.
            # When only text mode is needed, use model.generate_content() directly
            # (see _run_text_mode_loop below).
            agent = multimodal.MultimodalAgent(model=model)
            agent.start(ctx.room)

            # Route operator text messages to the planner
            @ctx.room.on("data_received")  # type: ignore[misc]
            async def on_data(data: bytes, participant, kind):  # type: ignore[no-untyped-def]
                try:
                    text = data.decode("utf-8").strip()
                    if text:
                        op_session.receive_operator_message(text)
                        action = await planner.plan(
                            goal=text,
                            current_app=None,
                            retry_count=0,
                        )
                        op_session.add_planned_action(action)
                        await ctx.room.local_participant.publish_data(
                            f"ACTION:{action.action}:{action.reason}".encode()
                        )
                except Exception as exc:
                    logger.error("Error handling operator data: %s", exc)

        return WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=cli.WorkerType.ROOM,
        )
    except Exception as exc:
        logger.error("Failed to build LiveKit worker: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Text-mode fallback loop
# ---------------------------------------------------------------------------


async def _run_text_mode_loop(
    session_manager: SessionManager,
    planner: Planner,
    cleanup_interval: float = 60.0,
) -> None:
    """Stub / text-mode event loop used when LiveKit is not available.

    This loop keeps the process alive and periodically cleans up expired
    sessions. It is a valid operational mode for backend-only deployments
    where the planner is invoked via the REST API (``/planner/debug``).
    """
    logger.info(
        "Running in text/stub mode. Planner accessible via REST API at /planner/debug."
    )
    try:
        while True:
            await asyncio.sleep(cleanup_interval)
            removed = session_manager.cleanup_expired()
            if removed:
                logger.info("Cleaned up %d expired session(s)", removed)
    except asyncio.CancelledError:
        logger.info("Text-mode loop cancelled")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def run_agent(*, skip_config_validation: bool = False) -> None:
    """Start the LiveKit + Vertex AI live agent.

    Args:
        skip_config_validation: If ``True``, skip the credential check (used
            in unit tests to avoid requiring real credentials).

    Raises:
        AgentConfigError: If required environment variables are missing and
            ``skip_config_validation`` is ``False``.
    """
    if not skip_config_validation:
        _validate_config()

    config = get_config()
    session_manager = SessionManager()
    planner = Planner()  # heuristic mode; pass vertex_client= for AI planning

    logger.info("Agent starting")
    logger.info("Vertex model: %s | Project: %s | Location: %s",
                config.vertex_live_model, config.google_cloud_project, config.google_cloud_location)

    if config.livekit_url and config.livekit_api_key and config.livekit_api_secret:
        logger.info("LiveKit configured: url=%s", config.livekit_url)
        worker = _build_livekit_worker(config, session_manager, planner)
        if worker is not None:
            # MEDIA_PIPELINE: cli.run_app() blocks and handles worker lifecycle,
            # reconnection, and graceful shutdown.  In production this is the
            # main blocking call.
            try:
                from livekit.agents import cli  # type: ignore[import]
                logger.info("Starting LiveKit worker (blocking)")
                # cli.run_app(worker) is synchronous; wrap in executor for async context
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, cli.run_app, worker)
                return
            except Exception as exc:
                logger.error("LiveKit worker failed: %s — falling back to text mode", exc)
        else:
            logger.warning("LiveKit SDK unavailable — falling back to text mode")
    else:
        logger.warning(
            "LiveKit credentials not set (LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET). "
            "Running in text/stub mode. REST API still available."
        )

    await _run_text_mode_loop(session_manager, planner)


if __name__ == "__main__":
    _cfg = get_config()
    setup_logging(_cfg.log_level)
    asyncio.run(run_agent())

