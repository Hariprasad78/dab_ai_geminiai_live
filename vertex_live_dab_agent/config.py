"""Centralized configuration loaded from environment variables.

If ``python-dotenv`` is installed, a local ``.env`` file is loaded
automatically once at import time.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


if load_dotenv is not None:
    # Keep existing exported env vars as source of truth.
    load_dotenv(override=False)


@dataclass
class Config:
    # Vertex AI
    google_cloud_project: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    google_cloud_location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1"))
    google_application_credentials: str = field(default_factory=lambda: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    vertex_planner_model: str = field(
        default_factory=lambda: (
            os.environ.get("VERTEX_PLANNER_MODEL")
            or os.environ.get("VERTEX_MODEL")
            or "gemini-2.5-flash"
        )
    )
    vertex_429_cooldown_seconds: float = field(
        default_factory=lambda: float(os.environ.get("VERTEX_429_COOLDOWN_SECONDS", "8.0"))
    )
    vertex_429_wait_seconds: float = field(
        default_factory=lambda: float(os.environ.get("VERTEX_429_WAIT_SECONDS", "1.5"))
    )
    vertex_live_model: str = field(default_factory=lambda: os.environ.get("VERTEX_LIVE_MODEL", "gemini-2.0-flash-live-preview-04-09"))
    enable_vertex_planner: bool = field(default_factory=lambda: os.environ.get("ENABLE_VERTEX_PLANNER", "false").lower() == "true")

    # LiveKit
    livekit_url: str = field(default_factory=lambda: os.environ.get("LIVEKIT_URL", ""))
    livekit_api_key: str = field(default_factory=lambda: os.environ.get("LIVEKIT_API_KEY", ""))
    livekit_api_secret: str = field(default_factory=lambda: os.environ.get("LIVEKIT_API_SECRET", ""))
    enable_livekit_agent: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_LIVEKIT_AGENT", "false").lower() == "true"
    )

    # DAB
    dab_mock_mode: bool = field(default_factory=lambda: os.environ.get("DAB_MOCK_MODE", "true").lower() == "true")
    dab_mqtt_broker: str = field(default_factory=lambda: os.environ.get("DAB_MQTT_BROKER", "localhost"))
    dab_mqtt_port: int = field(default_factory=lambda: int(os.environ.get("DAB_MQTT_PORT", "1883")))
    dab_device_id: str = field(default_factory=lambda: os.environ.get("DAB_DEVICE_ID", "mock-device"))
    dab_request_timeout: float = field(default_factory=lambda: float(os.environ.get("DAB_REQUEST_TIMEOUT", "10.0")))
    dab_max_retries: int = field(default_factory=lambda: int(os.environ.get("DAB_MAX_RETRIES", "3")))
    youtube_app_id: str = field(default_factory=lambda: os.environ.get("YOUTUBE_APP_ID", "youtube"))

    # Capture
    image_source: str = field(default_factory=lambda: os.environ.get("IMAGE_SOURCE", "auto"))
    enable_hdmi_capture: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_HDMI_CAPTURE", "true").lower() == "true"
    )
    enable_camera_capture: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_CAMERA_CAPTURE", "true").lower() == "true"
    )
    hdmi_capture_device: str = field(default_factory=lambda: os.environ.get("HDMI_CAPTURE_DEVICE", ""))
    hdmi_capture_width: int = field(default_factory=lambda: int(os.environ.get("HDMI_CAPTURE_WIDTH", "1920")))
    hdmi_capture_height: int = field(default_factory=lambda: int(os.environ.get("HDMI_CAPTURE_HEIGHT", "1080")))
    hdmi_capture_fps: float = field(default_factory=lambda: float(os.environ.get("HDMI_CAPTURE_FPS", "30.0")))
    hdmi_capture_fourcc: str = field(default_factory=lambda: os.environ.get("HDMI_CAPTURE_FOURCC", "MJPG"))
    hdmi_stream_jpeg_quality: int = field(default_factory=lambda: int(os.environ.get("HDMI_STREAM_JPEG_QUALITY", "80")))
    hdmi_audio_enabled: bool = field(default_factory=lambda: os.environ.get("HDMI_AUDIO_ENABLED", "false").lower() == "true")
    hdmi_audio_input_format: str = field(default_factory=lambda: os.environ.get("HDMI_AUDIO_INPUT_FORMAT", "auto"))
    hdmi_audio_device: str = field(default_factory=lambda: os.environ.get("HDMI_AUDIO_DEVICE", ""))
    hdmi_audio_sample_rate: int = field(default_factory=lambda: int(os.environ.get("HDMI_AUDIO_SAMPLE_RATE", "48000")))
    hdmi_audio_channels: int = field(default_factory=lambda: int(os.environ.get("HDMI_AUDIO_CHANNELS", "2")))
    hdmi_audio_bitrate: str = field(default_factory=lambda: os.environ.get("HDMI_AUDIO_BITRATE", "128k"))
    hdmi_audio_chunk_bytes: int = field(default_factory=lambda: int(os.environ.get("HDMI_AUDIO_CHUNK_BYTES", "4096")))

    # Planner/Orchestrator cadence
    planner_capture_interval_steps: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_CAPTURE_INTERVAL_STEPS", "3"))
    )
    planner_step_delay_seconds: float = field(
        default_factory=lambda: float(os.environ.get("PLANNER_STEP_DELAY_SECONDS", "0.2"))
    )
    planner_enable_subplans: bool = field(
        default_factory=lambda: os.environ.get("PLANNER_ENABLE_SUBPLANS", "true").lower() == "true"
    )
    planner_subplan_max_actions: int = field(
        default_factory=lambda: int(os.environ.get("PLANNER_SUBPLAN_MAX_ACTIONS", "4"))
    )
    planner_prompt_path: str = field(
        default_factory=lambda: os.environ.get("PLANNER_PROMPT_PATH", "")
    )
    planner_prompt_override: str = field(
        default_factory=lambda: os.environ.get("PLANNER_PROMPT_OVERRIDE", "")
    )

    # Session
    session_timeout_seconds: int = field(default_factory=lambda: int(os.environ.get("SESSION_TIMEOUT_SECONDS", "300")))
    max_steps_per_run: int = field(default_factory=lambda: int(os.environ.get("MAX_STEPS_PER_RUN", "50")))

    # Artifacts
    artifacts_base_dir: str = field(default_factory=lambda: os.environ.get("ARTIFACTS_BASE_DIR", "./artifacts"))
    device_profiles_dir: str = field(default_factory=lambda: os.environ.get("DEVICE_PROFILES_DIR", ""))
    trajectory_memory_path: str = field(default_factory=lambda: os.environ.get("TRAJECTORY_MEMORY_PATH", ""))
    hybrid_policy_mode: str = field(default_factory=lambda: os.environ.get("HYBRID_POLICY_MODE", "auto"))
    local_ranker_model_path: str = field(default_factory=lambda: os.environ.get("LOCAL_RANKER_MODEL_PATH", ""))

    # API
    api_host: str = field(default_factory=lambda: os.environ.get("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.environ.get("PORT") or os.environ.get("API_PORT", "8000")))

    # Logging
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))

    # TTS / narration
    tts_enabled: bool = field(default_factory=lambda: os.environ.get("TTS_ENABLED", "false").lower() == "true")
    tts_voice_provider: str = field(default_factory=lambda: os.environ.get("TTS_VOICE_PROVIDER", "google"))
    tts_model: str = field(default_factory=lambda: os.environ.get("TTS_MODEL", ""))
    tts_voice_name: str = field(default_factory=lambda: os.environ.get("TTS_VOICE_NAME", "en-US-Chirp3-HD-Aoede"))
    tts_fallback_voice_name: str = field(default_factory=lambda: os.environ.get("TTS_FALLBACK_VOICE_NAME", "en-US-Neural2-F"))
    tts_language_code: str = field(default_factory=lambda: os.environ.get("TTS_LANGUAGE_CODE", "en-US"))
    tts_speaking_rate: float = field(default_factory=lambda: float(os.environ.get("TTS_SPEAKING_RATE", "1.0")))
    tts_pitch: float = field(default_factory=lambda: float(os.environ.get("TTS_PITCH", "0.0")))
    tts_use_ssml: bool = field(default_factory=lambda: os.environ.get("TTS_USE_SSML", "true").lower() == "true")
    tts_play_goal_at_start: bool = field(default_factory=lambda: os.environ.get("TTS_PLAY_GOAL_AT_START", "true").lower() == "true")
    tts_play_step_updates: bool = field(default_factory=lambda: os.environ.get("TTS_PLAY_STEP_UPDATES", "true").lower() == "true")
    tts_play_recovery_updates: bool = field(default_factory=lambda: os.environ.get("TTS_PLAY_RECOVERY_UPDATES", "true").lower() == "true")
    tts_play_final_summary: bool = field(default_factory=lambda: os.environ.get("TTS_PLAY_FINAL_SUMMARY", "true").lower() == "true")


_config: Optional[Config] = None


def get_config() -> Config:
    """Return singleton config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the singleton (useful in tests when env vars change)."""
    global _config
    _config = None
