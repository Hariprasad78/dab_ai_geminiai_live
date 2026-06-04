"""Runtime Vertex/Gemini client for planner.

This module is optional at runtime. If Vertex SDK/auth is unavailable,
construction fails with a clear exception and callers should fall back to
heuristics.
"""

from __future__ import annotations

import asyncio
import base64
import io
import warnings
from typing import Any, Optional


GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"


class VertexPlannerClient:
    """Async wrapper around Vertex GenerativeModel for planner usage."""

    def __init__(self, *, project: str, location: str, model: str, api_key: Optional[str] = None) -> None:
        self._api_key = str(api_key or "").strip()
        self._use_api_key = bool(self._api_key)
        self._project = str(project or "").strip()
        self._location = str(location or "").strip()
        self._model_name = str(model or GEMINI_LIVE_MODEL).strip() or GEMINI_LIVE_MODEL

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
        self._genai_client: Any = None
        self._live_sessions: dict[str, dict[str, Any]] = {}
        self._live_session_creation_lock = asyncio.Lock()

    @staticmethod
    def _is_model_not_found_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return (
            "publisher model" in text
            and ("not found" in text or "does not have access" in text)
        ) or ("404" in text and "model" in text)

    async def _generate_with_api_key(
        self,
        *,
        prompt: str,
        screenshot_b64: Optional[str],
        session_id: Optional[str],
        keep_live_session: bool,
    ) -> str:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError(
                "google-genai is required for Gemini Live model calls. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        model_name = self._model_name
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        if self._genai_client is None:
            self._genai_client = genai.Client(
                http_options={"api_version": "v1beta"},
                api_key=self._api_key,
            )
        client = self._genai_client

        if "live" not in self._model_name.lower():
            contents: Any = prompt or "."
            if screenshot_b64:
                image_bytes = base64.b64decode(screenshot_b64)
                contents = [
                    prompt or ".",
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ]
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            response_text = str(getattr(response, "text", "") or response).strip()
            if session_id:
                self._chat_sessions[session_id] = [{"role": "model", "parts": [{"text": response_text}]}]
            return response_text

        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
                )
            ),
            generation_config=types.GenerationConfig(temperature=0.1),
        )

        async def _run_live_turn(session: Any) -> str:
            if screenshot_b64:
                image_bytes = base64.b64decode(screenshot_b64)
                try:
                    from PIL import Image

                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    jpeg_buffer = io.BytesIO()
                    image.save(jpeg_buffer, format="JPEG", quality=92)
                    image_bytes = jpeg_buffer.getvalue()
                except Exception:
                    pass
                await session.send_realtime_input(
                    video=types.Blob(data=image_bytes, mime_type="image/jpeg")
                )
            await session.send_realtime_input(text=prompt or ".")
            await session.send_realtime_input(activity_end={})

            text_parts: list[str] = []
            latest_transcription_text = ""
            async for response in session.receive():
                if text := getattr(response, "text", None):
                    text_parts.append(str(text))
                server_content = getattr(response, "server_content", None)
                if server_content:
                    output_transcription = getattr(server_content, "output_transcription", None)
                    if output_transcription and getattr(output_transcription, "text", None):
                        latest_transcription_text = str(output_transcription.text)
            return "".join(text_parts).strip() or latest_transcription_text.strip()

        if keep_live_session and session_id:
            async with self._live_session_creation_lock:
                entry = self._live_sessions.get(session_id)
                if entry is None:
                    context = client.aio.live.connect(model=model_name, config=live_config)
                    session = await context.__aenter__()
                    entry = {"context": context, "session": session, "lock": asyncio.Lock()}
                    self._live_sessions[session_id] = entry
            try:
                async with entry["lock"]:
                    response_text = await asyncio.wait_for(_run_live_turn(entry["session"]), timeout=90.0)
            except Exception:
                await self.close_live_session(session_id)
                raise
        else:
            async with client.aio.live.connect(model=model_name, config=live_config) as session:
                response_text = await asyncio.wait_for(_run_live_turn(session), timeout=90.0)
        if session_id:
            self._chat_sessions[session_id] = [{"role": "model", "parts": [{"text": response_text}]}]
        return response_text

    async def generate_content(
        self,
        prompt: str,
        screenshot_b64: Optional[str] = None,
        session_id: Optional[str] = None,
        keep_live_session: bool = False,
    ) -> str:
        """Generate text response with optional screenshot context."""

        if self._use_api_key:
            return await self._generate_with_api_key(
                prompt=prompt,
                screenshot_b64=screenshot_b64,
                session_id=session_id,
                keep_live_session=keep_live_session,
            )

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
        except Exception:
            raise

        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        return str(response)

    async def close_live_session(self, session_id: str) -> None:
        """Close one retained Gemini Live session."""

        entry = self._live_sessions.pop(str(session_id or ""), None)
        if entry is None:
            return
        try:
            await entry["context"].__aexit__(None, None, None)
        except Exception:
            pass
