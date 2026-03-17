"""Runtime Vertex/Gemini client for planner.

This module is optional at runtime. If Vertex SDK/auth is unavailable,
construction fails with a clear exception and callers should fall back to
heuristics.
"""

from __future__ import annotations

import asyncio
import base64
import warnings
from typing import Any, Optional


class VertexPlannerClient:
    """Async wrapper around Vertex GenerativeModel for planner usage."""

    def __init__(self, *, project: str, location: str, model: str) -> None:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required for Vertex planner")
        if not location:
            raise ValueError("GOOGLE_CLOUD_LOCATION is required for Vertex planner")
        if not model:
            raise ValueError("VERTEX_PLANNER_MODEL is required for Vertex planner")

        warnings.filterwarnings(
            "ignore",
            message=r"This feature is deprecated as of June 24, 2025",
            category=UserWarning,
        )
        vertexai.init(project=project, location=location)
        self._model = GenerativeModel(model)
        self._chat_sessions: dict[str, Any] = {}

    async def generate_content(
        self,
        prompt: str,
        screenshot_b64: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Generate text response with optional screenshot context."""

        def _call() -> Any:
            content: Any = prompt
            if screenshot_b64:
                from vertexai.generative_models import Part

                image_bytes = base64.b64decode(screenshot_b64)
                image_part = Part.from_data(data=image_bytes, mime_type="image/png")
                content = [prompt, image_part]

            if session_id:
                chat = self._chat_sessions.get(session_id)
                if chat is None:
                    chat = self._model.start_chat()
                    self._chat_sessions[session_id] = chat
                return chat.send_message(content)

            return self._model.generate_content(content)

        response = await asyncio.to_thread(_call)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        return str(response)
