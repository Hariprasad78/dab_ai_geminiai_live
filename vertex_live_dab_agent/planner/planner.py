"""Planning layer: takes observations and produces the next single action.

The planner is the only component allowed to decide what the agent does next.
It accepts all available context (goal, OCR text, screenshot flag, current
app/screen, last actions, retry count) and returns exactly one
:class:`~vertex_live_dab_agent.planner.schemas.PlannedAction`.

Two modes are supported:

* **Heuristic mode** (default, no ``vertex_client``) — rule-based fallback
  that is fully deterministic and requires no external services. Suitable for
  CI and mock tests.
* **Vertex AI mode** — sends a structured prompt to a Gemini model and parses
  the JSON response, with heuristic fall-through on any failure.

Safe fallbacks are always enforced:
* Unclear screen → ``NEED_BETTER_VIEW``
* Repeated failures → ``FAILED``
* Goal reached → ``DONE``
"""
import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.planner.schemas import ActionType, PlannedAction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported actions exposed in the context so the model knows its vocabulary
# ---------------------------------------------------------------------------
_SUPPORTED_ACTIONS: List[str] = [a.value for a in ActionType]

# ---------------------------------------------------------------------------
# System prompt — kept concise and deterministic
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """You are an AI agent controlling an Android TV / Google TV device via DAB protocol.

OUTPUT FORMAT — respond with ONLY a single JSON object, no prose, no markdown:
{"action": "<ACTION>", "confidence": <0.0-1.0>, "reason": "<short explanation>", "params": {}}

ALLOWED ACTIONS (use exactly these strings):
PRESS_UP, PRESS_DOWN, PRESS_LEFT, PRESS_RIGHT, PRESS_OK,
PRESS_BACK, PRESS_HOME, LAUNCH_APP, GET_STATE,
CAPTURE_SCREENSHOT, WAIT, DONE, FAILED, NEED_BETTER_VIEW

MANDATORY RULES:
1. Return exactly one action.
2. reason must be a non-empty string (max 120 chars).
3. confidence must be a float in [0.0, 1.0].
4. LAUNCH_APP requires params: {"app_id": "<package_name>"}.
5. WAIT requires params: {"seconds": <int>}.
6. If goal is achieved: action = DONE, confidence >= 0.9.
7. If screen is unclear or no context: action = NEED_BETTER_VIEW.
8. If retry_count > 5 or repeated failures: action = FAILED.
9. Never invent actions outside the allowed list.
10. Never include explanatory text outside the JSON object."""

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Plans next action based on current observations."""

    def __init__(self, vertex_client: Optional[Any] = None) -> None:
        """
        Initialize planner.

        Args:
            vertex_client: Optional Vertex AI client for AI-powered planning.
                          If None, uses deterministic heuristic planning.
        """
        self._vertex_client = vertex_client
        self._config = get_config()

    async def plan(
        self,
        goal: str,
        screenshot_b64: Optional[str] = None,
        ocr_text: Optional[str] = None,
        current_app: Optional[str] = None,
        current_screen: Optional[str] = None,
        last_actions: Optional[List[str]] = None,
        retry_count: int = 0,
    ) -> PlannedAction:
        """Plan the next action.

        Args:
            goal: The testing goal to achieve.
            screenshot_b64: Base64 encoded screenshot if available.
            ocr_text: OCR text extracted from screenshot.
            current_app: Currently active app.
            current_screen: Current screen identifier.
            last_actions: Recent actions taken.
            retry_count: Number of retries so far.

        Returns:
            PlannedAction with action, confidence, reason, and optional params.
        """
        if retry_count >= self._config.max_steps_per_run:
            logger.warning("Max retries reached (%d), returning FAILED", retry_count)
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=1.0,
                reason=f"Exceeded maximum steps ({self._config.max_steps_per_run})",
            )

        context = self._build_context(
            goal=goal,
            has_screenshot=screenshot_b64 is not None,
            ocr_text=ocr_text,
            current_app=current_app,
            current_screen=current_screen,
            last_actions=last_actions or [],
            retry_count=retry_count,
        )

        if self._vertex_client is not None:
            return await self._plan_with_vertex(context, screenshot_b64)
        return self._plan_heuristic(goal, last_actions or [], retry_count)

    def _build_context(
        self,
        goal: str,
        has_screenshot: bool,
        ocr_text: Optional[str],
        current_app: Optional[str],
        current_screen: Optional[str],
        last_actions: List[str],
        retry_count: int,
    ) -> str:
        """Build the context string sent to the planner model."""
        parts = [
            f"Goal: {goal}",
            f"Current app: {current_app or 'unknown'}",
            f"Current screen: {current_screen or 'unknown'}",
            f"Has screenshot: {has_screenshot}",
            f"Retry count: {retry_count}",
            f"Supported actions: {', '.join(_SUPPORTED_ACTIONS)}",
        ]
        if ocr_text:
            parts.append(f"Screen text (OCR): {ocr_text[:500]}")
        if last_actions:
            parts.append(f"Last actions: {', '.join(str(a) for a in last_actions[-5:])}")
        return "\n".join(parts)

    async def _plan_with_vertex(
        self, context: str, screenshot_b64: Optional[str]
    ) -> PlannedAction:
        """Use Vertex AI to plan next action."""
        try:
            prompt = f"{PLANNER_SYSTEM_PROMPT}\n\nCurrent situation:\n{context}"
            # TODO: Add screenshot as multimodal input when vertex_client supports it
            response = await self._vertex_client.generate_content(prompt)
            return self._parse_action(response)
        except Exception as exc:
            logger.error("Vertex AI planning failed: %s, falling back to heuristic", exc)
            return self._plan_heuristic(context, [], 0)

    def _plan_heuristic(
        self, goal: str, last_actions: List[str], retry_count: int
    ) -> PlannedAction:
        """Deterministic heuristic planner for mock/testing mode.

        Decision priority:
        1. Too many retries → FAILED
        2. No prior context → CAPTURE_SCREENSHOT
        3. Last action was a screenshot → GET_STATE
        4. retry_count > 3 → FAILED (likely stuck)
        5. Default → NEED_BETTER_VIEW
        """
        if not last_actions:
            return PlannedAction(
                action=ActionType.CAPTURE_SCREENSHOT,
                confidence=0.9,
                reason="No prior context - capturing screenshot to observe current state",
            )
        # last_actions may contain either ActionType enum members or plain strings
        # (use_enum_values=True means PlannedAction stores strings; external callers
        # may pass enum members).  ActionType inherits from str so == works for both.
        if last_actions[-1] == ActionType.CAPTURE_SCREENSHOT.value:
            return PlannedAction(
                action=ActionType.GET_STATE,
                confidence=0.8,
                reason="Screenshot captured - getting app state",
            )
        if retry_count > 3:
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=0.9,
                reason="Multiple retries without progress in heuristic mode",
            )
        return PlannedAction(
            action=ActionType.NEED_BETTER_VIEW,
            confidence=0.7,
            reason="Heuristic planner cannot determine next action without Vertex AI",
        )

    def _validate_action(self, action: PlannedAction) -> PlannedAction:
        """Post-parse validation: re-validate via Pydantic to catch constraint violations.

        Returns the action unchanged if valid, or a FAILED action if the
        parsed result violates any schema constraint. This is a second safety
        net on top of the initial Pydantic parse so that partial JSON objects
        assembled in ``_parse_action`` are always fully validated.
        """
        try:
            # Re-validate by round-tripping through the model
            validated = PlannedAction.model_validate(action.model_dump())
            return validated
        except (ValidationError, Exception) as exc:
            logger.error("Planner action failed validation: %s", exc)
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=0.5,
                reason=f"Action failed validation: {exc}",
            )

    def _parse_action(self, response_text: str) -> PlannedAction:
        """Parse and validate action JSON from model response.

        Handles raw JSON as well as JSON wrapped in markdown code fences
        (e.g. ```json ... ```). After parsing, the result is passed through
        :meth:`_validate_action` to catch constraint violations.
        """
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # First line should be ``` or ```json -- skip it
                fence_label = lines[0].strip().lstrip("`").lower()
                if fence_label not in ("", "json"):
                    logger.warning("Unexpected opening fence: %s", lines[0])
                # Strip fences only when the closing fence is present
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])
            data = json.loads(text)
            action = PlannedAction(**data)
            return self._validate_action(action)
        except (ValidationError, Exception) as exc:
            logger.error("Failed to parse planner response: %s | response: %s", exc, response_text)
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=0.5,
                reason=f"Failed to parse planner response: {exc}",
            )
