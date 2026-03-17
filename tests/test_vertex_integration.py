"""Optional live Vertex AI integration smoke tests.

These tests call the real Vertex API.

Behavior:
- Auto mode (default): runs when required env is present, otherwise skips.
- Force run: RUN_VERTEX_INTEGRATION_TESTS=1
- Force skip: RUN_VERTEX_INTEGRATION_TESTS=0

Required env vars:
- GOOGLE_CLOUD_PROJECT
- GOOGLE_CLOUD_LOCATION
- GOOGLE_APPLICATION_CREDENTIALS (or ADC)

Optional env var:
- VERTEX_TEST_MODEL (default: gemini-2.5-flash)
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import warnings
from typing import Any

import pytest

try:
    from google.api_core import exceptions as gexc
except ImportError:  # pragma: no cover
    gexc = None  # type: ignore[assignment]

from vertex_live_dab_agent.capture.validator import ValidationResult, Validator
from vertex_live_dab_agent.planner.planner import Planner
from vertex_live_dab_agent.planner.schemas import ActionType


_REQUIRED_ENV = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")


class AsyncVertexTextClient:
    """Tiny async wrapper around Vertex GenerativeModel for tests."""

    def __init__(self, model_name: str) -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"This feature is deprecated as of June 24, 2025",
            category=UserWarning,
        )
        vertexai = importlib.import_module("vertexai")
        gm = importlib.import_module("vertexai.generative_models")
        GenerativeModel = getattr(gm, "GenerativeModel")

        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "")
        vertexai.init(project=project, location=location)
        self._model = GenerativeModel(model_name)

    async def generate_content(self, prompt: str) -> str:
        def _call() -> Any:
            return self._model.generate_content(prompt)

        response = await asyncio.to_thread(_call)
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        # Fallback for SDK variants where text may be absent.
        return str(response)


def _resolve_test_model_name() -> str:
    """Choose non-live model name from explicit test env or default."""
    return (
        os.getenv("VERTEX_TEST_MODEL")
        or "gemini-2.5-flash"
    )


@pytest.fixture(scope="module")
def vertex_client() -> AsyncVertexTextClient:
    run_mode = os.getenv("RUN_VERTEX_INTEGRATION_TESTS", "auto").strip().lower()
    if run_mode in {"0", "false", "no"}:
        pytest.skip("RUN_VERTEX_INTEGRATION_TESTS=0 (live Vertex tests disabled)")

    missing = [k for k in _REQUIRED_ENV if not os.getenv(k)]
    if missing:
        if run_mode in {"1", "true", "yes"}:
            pytest.skip(
                f"RUN_VERTEX_INTEGRATION_TESTS is enabled but required env is missing: {', '.join(missing)}"
            )
        pytest.skip(
            f"Auto mode: missing required env vars for live Vertex tests: {', '.join(missing)}"
        )

    model_name = _resolve_test_model_name()
    client = AsyncVertexTextClient(model_name)

    async def _probe() -> None:
        await client.generate_content('Return only JSON: {"ok": true}')

    _catch = (gexc.NotFound, gexc.PermissionDenied, gexc.Unauthenticated) if gexc is not None else ()
    try:
        asyncio.run(_probe())
    except _catch as exc:
        pytest.skip(
            f"Model not accessible in this project/region: {model_name}. "
            "Check credentials/service-account state and set VERTEX_TEST_MODEL "
            f"to an allowed model. Details: {exc}"
        )

    return client


@pytest.mark.asyncio
async def test_vertex_generate_content_smoke(vertex_client: AsyncVertexTextClient):
    """Connectivity smoke test: model should return a short deterministic token."""
    prompt = "Reply with exactly this JSON: {\"ok\": true}"
    text = await vertex_client.generate_content(prompt)
    assert isinstance(text, str)
    assert text.strip()
    assert "ok" in text.lower()


@pytest.mark.asyncio
async def test_vertex_planner_path_smoke(vertex_client: AsyncVertexTextClient):
    """Planner should invoke Vertex path and produce a parseable PlannedAction."""
    planner = Planner(vertex_client=vertex_client)
    result = await planner.plan(
        goal="Open YouTube",
        ocr_text="Home YouTube Apps",
        last_actions=[ActionType.CAPTURE_SCREENSHOT.value],
        retry_count=0,
    )

    # Require a non-terminal result to avoid false positives from failure fallback.
    assert result.action in {a.value for a in ActionType}
    assert result.action != ActionType.FAILED.value
    assert isinstance(result.reason, str)
    assert result.reason.strip()


@pytest.mark.asyncio
async def test_vertex_semantic_validator_smoke(vertex_client: AsyncVertexTextClient):
    """Validator semantic path should return a valid ValidationResult enum."""
    validator = Validator(vertex_client=vertex_client)
    res = await validator.validate_semantic(
        goal="Is home screen visible?",
        screenshot_b64=None,
        ocr_text="Home Movies Apps",
    )
    assert res.result in {ValidationResult.PASS, ValidationResult.FAIL}


@pytest.mark.asyncio
async def test_vertex_json_response_parseability(vertex_client: AsyncVertexTextClient):
    """Model should be able to return machine-parseable JSON when asked directly."""
    prompt = (
        "Return only JSON with keys action, confidence, reason, params. "
        "Use action=GET_STATE, confidence=0.8, reason='check state', params={}."
    )
    text = await vertex_client.generate_content(prompt)
    # tolerate markdown fences from some models
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1])
    # best effort parse
    data = json.loads(cleaned)
    assert isinstance(data, dict)
    assert "action" in data
