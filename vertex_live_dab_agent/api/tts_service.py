"""Narration mapping + Google Cloud TTS synthesis service."""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from vertex_live_dab_agent.config import get_config


def should_speak_event(event: Dict[str, Any], last_text: str = "") -> bool:
    text = str(event.get("tts_text", "")).strip()
    if not text:
        return False
    if text == last_text:
        return False
    return bool(event.get("tts_should_play", True))


def build_tts_text(event: Dict[str, Any], run_state: Optional[Dict[str, Any]] = None) -> Optional[str]:
    category = str(event.get("tts_category", "")).upper()
    text = str(event.get("tts_text", "")).strip()
    if text:
        return text
    if category == "GOAL":
        return f"Goal: {str((run_state or {}).get('goal', '')).strip()}"
    return None


def event_to_narration(action: str, reason: str, goal: str, category: str = "STEP_START") -> Dict[str, Any]:
    action_u = str(action or "").upper()
    msg = {
        "LAUNCH_APP": "Trying to open the app.",
        "OPEN_CONTENT": "Trying to open the content.",
        "WAIT": "Waiting briefly for the screen to load.",
        "GET_STATE": "Checking whether the app is really open.",
        "CAPTURE_SCREENSHOT": "Taking a screenshot to understand the current screen.",
        "PRESS_BACK": "Pressing Back.",
        "PRESS_HOME": "Pressing Home.",
        "PRESS_OK": "Pressing OK.",
        "NEED_PLAYER_CONTROLS_VISIBLE": "The player controls are not clearly visible yet.",
        "NEED_VIDEO_PLAYBACK_CONFIRMED": "Trying to confirm video playback.",
        "NEED_SETTINGS_GEAR_LOCATION": "Looking for the settings gear in the player controls.",
        "NEED_PLAYER_MENU_CONFIRMATION": "Trying to confirm the player settings menu.",
        "NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION": "Trying to confirm the Stats for Nerds toggle.",
        "FAILED": "The test has stopped because recovery did not work.",
        "DONE": "The test is complete.",
    }.get(action_u, f"Running action {action_u.lower().replace('_', ' ')}.")

    return {
        "tts_text": msg,
        "tts_priority": 50 if category in {"FAILURE", "WARNING"} else 20,
        "tts_category": category,
        "tts_should_play": True,
        "tts_interruptible": category not in {"FAILURE", "SUCCESS"},
        "goal": goal,
        "reason": reason,
    }


class GoogleTTSService:
    def __init__(self) -> None:
        self._config = get_config()
        self._client = None
        self._tts_mod = None
        self._init_error: Optional[str] = None
        try:
            from google.cloud import texttospeech  # type: ignore

            self._tts_mod = texttospeech
            self._client = texttospeech.TextToSpeechClient()
        except Exception as exc:  # pragma: no cover
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self._client is not None and self._tts_mod is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def _choose_voice_name(self) -> str:
        model = str(self._config.tts_model or "").lower()
        # Preference order: Gemini-TTS (if configured) -> Chirp 3 HD -> fallback
        if "gemini" in model and self._config.tts_voice_name:
            return self._config.tts_voice_name
        if self._config.tts_voice_name:
            return self._config.tts_voice_name
        return self._config.tts_fallback_voice_name

    def synthesize_tts(self, text: str, use_ssml: Optional[bool] = None) -> Dict[str, Any]:
        if not self._config.tts_enabled:
            return {"success": False, "error": "TTS disabled", "audio_b64": None}
        if not self.available:
            return {"success": False, "error": self._init_error or "Google TTS unavailable", "audio_b64": None}

        tts = self._tts_mod
        assert tts is not None
        client = self._client
        assert client is not None

        raw_text = str(text or "").strip()
        if not raw_text:
            return {"success": False, "error": "Empty text", "audio_b64": None}

        use_ssml_final = self._config.tts_use_ssml if use_ssml is None else bool(use_ssml)
        if use_ssml_final:
            input_obj = tts.SynthesisInput(ssml=f"<speak>{raw_text}</speak>")
        else:
            input_obj = tts.SynthesisInput(text=raw_text)

        voice = tts.VoiceSelectionParams(
            language_code=self._config.tts_language_code,
            name=self._choose_voice_name(),
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MP3,
            speaking_rate=float(self._config.tts_speaking_rate),
            pitch=float(self._config.tts_pitch),
        )

        try:
            response = client.synthesize_speech(
                request={"input": input_obj, "voice": voice, "audio_config": audio_config}
            )
            audio_b64 = base64.b64encode(response.audio_content).decode("ascii")
            return {
                "success": True,
                "audio_b64": audio_b64,
                "voice_name": voice.name,
                "language_code": voice.language_code,
            }
        except Exception as exc:  # pragma: no cover
            # graceful fallback voice
            try:
                fallback_voice = tts.VoiceSelectionParams(
                    language_code=self._config.tts_language_code,
                    name=self._config.tts_fallback_voice_name,
                )
                response = client.synthesize_speech(
                    request={"input": input_obj, "voice": fallback_voice, "audio_config": audio_config}
                )
                audio_b64 = base64.b64encode(response.audio_content).decode("ascii")
                return {
                    "success": True,
                    "audio_b64": audio_b64,
                    "voice_name": fallback_voice.name,
                    "language_code": fallback_voice.language_code,
                    "fallback_used": True,
                }
            except Exception as exc2:
                return {"success": False, "error": f"{exc}; fallback failed: {exc2}", "audio_b64": None}
