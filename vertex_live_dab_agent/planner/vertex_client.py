"""Runtime Vertex/Gemini client for planner.

This module is optional at runtime. If Vertex SDK/auth is unavailable,
construction fails with a clear exception and callers should fall back to
heuristics.
"""

from __future__ import annotations

import asyncio
import base64
import httpx
import warnings
from typing import Any, Optional


class VertexPlannerClient:
    """Async wrapper around Vertex GenerativeModel for planner usage."""

    def __init__(self, *, project: str, location: str, model: str, api_key: Optional[str] = None) -> None:
        self._api_key = str(api_key or "").strip()
        self._use_api_key = bool(self._api_key)
        self._project = str(project or "").strip()
        self._location = str(location or "").strip()
        self._model_name = str(model or "").strip()
        if not self._model_name:
            raise ValueError("VERTEX_PLANNER_MODEL is required for planner")

        if not self._use_api_key:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            if not self._project:
                raise ValueError("GOOGLE_CLOUD_PROJECT is required when GOOGLE_API_KEY/GEMINI_API_KEY is not set")
            if not self._location:
                raise ValueError("GOOGLE_CLOUD_LOCATION is required when GOOGLE_API_KEY/GEMINI_API_KEY is not set")

            warnings.filterwarnings(
                "ignore",
                message=r"This feature is deprecated as of June 24, 2025",
                category=UserWarning,
            )
            vertexai.init(project=self._project, location=self._location)
            self._model = GenerativeModel(self._model_name)
        else:
            self._model = None

        self._chat_sessions: dict[str, Any] = {}

    @staticmethod
    def _is_model_not_found_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return (
            "publisher model" in text
            and ("not found" in text or "does not have access" in text)
        ) or ("404" in text and "model" in text)

    @staticmethod
    def _fallback_models(preferred: str) -> list[str]:
        ordered = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-live-preview",
            "gemini-3-flash-preview",
            "gemini-2.0-flash-001",
            "gemini-1.5-flash-002",
        ]
        p = str(preferred or "").strip()
        return [m for m in ordered if m and m != p]

    def _switch_model(self, model_name: str) -> None:
        if not model_name or model_name == self._model_name:
            return
        if not self._use_api_key:
            from vertexai.generative_models import GenerativeModel

            self._model = GenerativeModel(model_name)
        self._model_name = model_name
        self._chat_sessions.clear()

    async def _generate_with_api_key(
        self,
        *,
        prompt: str,
        screenshot_b64: Optional[str],
        session_id: Optional[str],
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent"
        params = {"key": self._api_key}

        parts: list[dict[str, Any]] = [{"text": prompt}]
        if screenshot_b64:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": screenshot_b64,
                    }
                }
            )

        user_turn = {"role": "user", "parts": parts}
        contents: list[dict[str, Any]] = []
        if session_id:
            prior = self._chat_sessions.get(session_id)
            if isinstance(prior, list):
                contents.extend(prior)
        contents.append(user_turn)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, params=params, json={"contents": contents})
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}

        text_out = ""
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if isinstance(candidates, list) and candidates:
            first = candidates[0] if isinstance(candidates[0], dict) else {}
            content = first.get("content") if isinstance(first, dict) else {}
            parts_out = (content or {}).get("parts") if isinstance(content, dict) else []
            if isinstance(parts_out, list):
                texts = [
                    str(p.get("text", "")).strip()
                    for p in parts_out
                    if isinstance(p, dict) and str(p.get("text", "")).strip()
                ]
                text_out = "\n".join(texts).strip()

        if session_id:
            history = list(contents)
            if text_out:
                history.append({"role": "model", "parts": [{"text": text_out}]})
            self._chat_sessions[session_id] = history

        return text_out or str(payload)

    async def generate_content(
        self,
        prompt: str,
        screenshot_b64: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Generate text response with optional screenshot context."""

        if self._use_api_key:
            try:
                return await self._generate_with_api_key(
                    prompt=prompt,
                    screenshot_b64=screenshot_b64,
                    session_id=session_id,
                )
            except Exception as exc:
                if not self._is_model_not_found_error(exc):
                    raise
                last_exc: Exception = exc
                for fallback in self._fallback_models(self._model_name):
                    try:
                        self._switch_model(fallback)
                        return await self._generate_with_api_key(
                            prompt=prompt,
                            screenshot_b64=screenshot_b64,
                            session_id=session_id,
                        )
                    except Exception as inner:
                        last_exc = inner
                        continue
                raise last_exc

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

        try:
            response = await asyncio.to_thread(_call)
        except Exception as exc:
            if not self._is_model_not_found_error(exc):
                raise

            last_exc: Exception = exc
            for fallback in self._fallback_models(self._model_name):
                try:
                    self._switch_model(fallback)
                    response = await asyncio.to_thread(_call)
                    break
                except Exception as inner:
                    last_exc = inner
                    continue
            else:
                raise last_exc

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        return str(response)
