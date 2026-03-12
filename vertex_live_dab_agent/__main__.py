"""Package entrypoint: ``python -m vertex_live_dab_agent``

Starts the FastAPI server using settings from environment variables (or
``.env`` if you load it with python-dotenv).

Usage::

    python -m vertex_live_dab_agent
    # or with uvicorn directly:
    uvicorn vertex_live_dab_agent.api.api:app --reload --port 8000
"""
import uvicorn

from vertex_live_dab_agent.artifacts.logger import setup_logging
from vertex_live_dab_agent.config import get_config

_VALID_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}

if __name__ == "__main__":
    config = get_config()
    setup_logging(config.log_level)
    uvicorn_log_level = config.log_level.lower()
    if uvicorn_log_level not in _VALID_LOG_LEVELS:
        uvicorn_log_level = "info"
    uvicorn.run(
        "vertex_live_dab_agent.api.api:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level=uvicorn_log_level,
    )
