"""Artifact store and logging setup for DAB AI agent runs.

Each run gets its own directory under ``artifacts_base_dir`` / ``run_id``.
Subdirectories ``screenshots/`` and ``planner_traces/`` are created on
instantiation.  All writes are best-effort: errors are logged, never raised.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from vertex_live_dab_agent.config import get_config

logger = logging.getLogger(__name__)


class ArtifactStore:
    """Persists run artifacts (screenshots, action logs, planner traces).

    Parameters
    ----------
    run_id:
        Unique run identifier.  Used as the leaf directory name.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        cfg = get_config()
        base = Path(cfg.artifacts_base_dir).expanduser()
        self._run_dir = base / run_id
        (self._run_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (self._run_dir / "planner_traces").mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        """Root directory for this run's artifacts."""
        return self._run_dir

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def save_metadata(self, data: Dict[str, Any]) -> None:
        """Persist run metadata as ``metadata.json``."""
        self._write_json(self._run_dir / "metadata.json", data)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def save_action(self, data: Dict[str, Any]) -> None:
        """Append a single action record to ``actions.jsonl``."""
        path = self._run_dir / "actions.jsonl"
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("ArtifactStore.save_action failed: %s", exc)

    # ------------------------------------------------------------------
    # Screenshots
    # ------------------------------------------------------------------

    def save_screenshot(self, image_b64: str, step: int) -> Optional[Path]:
        """Decode *image_b64* and write it to ``screenshots/step_NNNN.png``.

        Returns the :class:`~pathlib.Path` of the saved file, or *None* when
        the input is not valid base-64 or any write error occurs.
        """
        dest = self._run_dir / "screenshots" / f"step_{step:04d}.png"
        try:
            raw = base64.b64decode(image_b64)
        except Exception as exc:
            logger.warning("ArtifactStore.save_screenshot: invalid base64 at step %d: %s", step, exc)
            return None
        try:
            dest.write_bytes(raw)
            return dest
        except Exception as exc:
            logger.warning("ArtifactStore.save_screenshot: write failed at step %d: %s", step, exc)
            return None

    # ------------------------------------------------------------------
    # Planner traces
    # ------------------------------------------------------------------

    def save_planner_trace(self, data: Dict[str, Any], step: int) -> None:
        """Write a planner trace for *step* to ``planner_traces/step_NNNN.json``."""
        dest = self._run_dir / "planner_traces" / f"step_{step:04d}.json"
        self._write_json(dest, data)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def save_final_summary(self, data: Dict[str, Any]) -> None:
        """Persist end-of-run summary as ``final_summary.json``."""
        self._write_json(self._run_dir / "final_summary.json", data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("ArtifactStore._write_json failed for %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def setup_logging(level: str = "INFO") -> None:
    """Configure root-logger level and a sensible console handler.

    Unknown level strings are silently replaced with ``"INFO"``.
    Safe to call multiple times; duplicate handlers are avoided.
    """
    normalized = str(level or "INFO").strip().upper()
    if normalized not in _VALID_LEVELS:
        normalized = "INFO"
    numeric = getattr(logging, normalized, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric)

    # Avoid adding duplicate StreamHandlers when called repeatedly.
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        handler = logging.StreamHandler()
        handler.setLevel(numeric)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
