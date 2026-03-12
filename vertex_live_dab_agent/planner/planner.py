"""Planning layer - takes observations and produces next action."""
import json
import logging
from typing import Any, Dict, List, Optional

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.planner.schemas import ActionType, PlannedAction

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """
You are an AI agent controlling an Android TV / Google TV device via DAB protocol.

Your job is to decide the next single action to take to achieve the given goal.

You must respond ONLY with a valid JSON object matching this exact schema:
{
  "action": "<ActionType>",
  "confidence": <float 0.0-1.0>,
  "reason": "<explanation>",
  "params": {<optional key-value params>}
}

Allowed action values:
PRESS_UP, PRESS_DOWN, PRESS_LEFT, PRESS_RIGHT, PRESS_OK,
PRESS_BACK, PRESS_HOME, LAUNCH_APP, GET_STATE,
CAPTURE_SCREENSHOT, WAIT, DONE, FAILED, NEED_BETTER_VIEW

Rules:
- If the goal is achieved, return DONE.
- If the screen is unclear or you need a better view, return NEED_BETTER_VIEW.
- If you have retried many times and failed, return FAILED.
- For LAUNCH_APP, include {"app_id": "<app_id>"} in params.
- For WAIT, include {"seconds": <int>} in params.
- Never invent random commands outside the allowed list.
- Be concise in your reason.
"""


class Planner:
    """Plans next action based on current observations."""

    def __init__(self, vertex_client: Optional[Any] = None) -> None:
        """
        Initialize planner.

        Args:
            vertex_client: Optional Vertex AI client for AI-powered planning.
                          If None, uses heuristic/mock planning.
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
        """
        Plan the next action.

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
            logger.warning("Max retries reached, returning FAILED")
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=1.0,
                reason=f"Exceeded maximum steps ({self._config.max_steps_per_run})",
            )

        context = self._build_context(
            goal=goal,
            ocr_text=ocr_text,
            current_app=current_app,
            current_screen=current_screen,
            last_actions=last_actions or [],
            retry_count=retry_count,
        )

        if self._vertex_client is not None:
            return await self._plan_with_vertex(context, screenshot_b64)
        else:
            return self._plan_heuristic(goal, last_actions or [], retry_count)

    def _build_context(
        self,
        goal: str,
        ocr_text: Optional[str],
        current_app: Optional[str],
        current_screen: Optional[str],
        last_actions: List[str],
        retry_count: int,
    ) -> str:
        """Build the context string for the planner."""
        parts = [
            f"Goal: {goal}",
            f"Current app: {current_app or 'unknown'}",
            f"Current screen: {current_screen or 'unknown'}",
            f"Retry count: {retry_count}",
        ]
        if ocr_text:
            parts.append(f"Screen text (OCR): {ocr_text[:500]}")
        if last_actions:
            parts.append(f"Last actions: {', '.join(last_actions[-5:])}")
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
        """Simple heuristic planner for mock/testing mode."""
        if not last_actions:
            return PlannedAction(
                action=ActionType.CAPTURE_SCREENSHOT,
                confidence=0.9,
                reason="No prior context - capturing screenshot to observe current state",
            )
        if last_actions and last_actions[-1] == ActionType.CAPTURE_SCREENSHOT.value:
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

    def _parse_action(self, response_text: str) -> PlannedAction:
        """Parse action JSON from model response.

        Handles raw JSON as well as JSON wrapped in markdown code fences
        (e.g. triple-backtick json ... triple-backtick).
        """
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # First line should be ``` or ```json – skip it
                if not lines[0].strip().lstrip("`").lower() in ("", "json"):
                    logger.warning("Unexpected opening fence: %s", lines[0])
                # Strip fences only when the closing fence is present
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])
            data = json.loads(text)
            return PlannedAction(**data)
        except Exception as exc:
            logger.error("Failed to parse planner response: %s | response: %s", exc, response_text)
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=0.5,
                reason=f"Failed to parse planner response: {exc}",
            )
