"""Centralized configuration loaded from environment variables."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Vertex AI
    google_cloud_project: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    google_cloud_location: str = field(default_factory=lambda: os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1"))
    google_application_credentials: str = field(default_factory=lambda: os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""))
    vertex_live_model: str = field(default_factory=lambda: os.environ.get("VERTEX_LIVE_MODEL", "gemini-2.0-flash-live-preview-04-09"))

    # LiveKit
    livekit_url: str = field(default_factory=lambda: os.environ.get("LIVEKIT_URL", ""))
    livekit_api_key: str = field(default_factory=lambda: os.environ.get("LIVEKIT_API_KEY", ""))
    livekit_api_secret: str = field(default_factory=lambda: os.environ.get("LIVEKIT_API_SECRET", ""))

    # DAB
    dab_mock_mode: bool = field(default_factory=lambda: os.environ.get("DAB_MOCK_MODE", "true").lower() == "true")
    dab_mqtt_broker: str = field(default_factory=lambda: os.environ.get("DAB_MQTT_BROKER", "localhost"))
    dab_mqtt_port: int = field(default_factory=lambda: int(os.environ.get("DAB_MQTT_PORT", "1883")))
    dab_device_id: str = field(default_factory=lambda: os.environ.get("DAB_DEVICE_ID", "mock-device"))
    dab_request_timeout: float = field(default_factory=lambda: float(os.environ.get("DAB_REQUEST_TIMEOUT", "10.0")))
    dab_max_retries: int = field(default_factory=lambda: int(os.environ.get("DAB_MAX_RETRIES", "3")))

    # Session
    session_timeout_seconds: int = field(default_factory=lambda: int(os.environ.get("SESSION_TIMEOUT_SECONDS", "300")))
    max_steps_per_run: int = field(default_factory=lambda: int(os.environ.get("MAX_STEPS_PER_RUN", "50")))

    # Artifacts
    artifacts_base_dir: str = field(default_factory=lambda: os.environ.get("ARTIFACTS_BASE_DIR", "./artifacts"))

    # API
    api_host: str = field(default_factory=lambda: os.environ.get("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.environ.get("API_PORT", "8000")))

    # Logging
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))


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
