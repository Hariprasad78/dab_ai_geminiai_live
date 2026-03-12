"""Session manager for Vertex AI / LiveKit sessions."""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from vertex_live_dab_agent.config import get_config

logger = logging.getLogger(__name__)


class SessionState:
    """Represents a single session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.last_activity = datetime.now(timezone.utc)
        self.conversation_history: List[Dict[str, str]] = []
        self.is_active = True

    def record_message(self, role: str, content: str) -> None:
        self.conversation_history.append(
            {"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()}
        )
        self.last_activity = datetime.now(timezone.utc)

    def is_expired(self, timeout_seconds: int) -> bool:
        elapsed = (datetime.now(timezone.utc) - self.last_activity).total_seconds()
        return elapsed > timeout_seconds


class SessionManager:
    """Manages Vertex AI / LiveKit sessions."""

    def __init__(self) -> None:
        self._config = get_config()
        self._sessions: Dict[str, SessionState] = {}

    def start_session(self, session_id: str) -> SessionState:
        """Start a new session."""
        session = SessionState(session_id)
        self._sessions[session_id] = session
        logger.info("Session started: %s", session_id)
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        """End a session."""
        if session_id in self._sessions:
            self._sessions[session_id].is_active = False
            logger.info("Session ended: %s", session_id)

    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count removed."""
        to_remove = [
            sid for sid, s in self._sessions.items()
            if s.is_expired(self._config.session_timeout_seconds)
        ]
        for sid in to_remove:
            del self._sessions[sid]
            logger.info("Session expired and cleaned up: %s", sid)
        return len(to_remove)

    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """List all active sessions."""
        return {
            sid: {
                "session_id": s.session_id,
                "started_at": s.started_at,
                "is_active": s.is_active,
                "message_count": len(s.conversation_history),
            }
            for sid, s in self._sessions.items()
        }
