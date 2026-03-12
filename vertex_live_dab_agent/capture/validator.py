"""Validation layer - semantic and deterministic validation."""
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    SKIP = "SKIP"


class StepResult:
    """Result of a single step validation."""

    def __init__(self, result: ValidationResult, reason: str, data: Optional[Dict[str, Any]] = None):
        self.result = result
        self.reason = reason
        self.data = data or {}

    def __repr__(self) -> str:
        return f"StepResult(result={self.result}, reason={self.reason!r})"


class Validator:
    """Validates step outcomes using semantic and deterministic methods."""

    def __init__(self, vertex_client: Optional[Any] = None) -> None:
        self._vertex_client = vertex_client
        self._deterministic_hooks: List[Callable] = []

    def add_deterministic_hook(self, hook: Callable[[str, str], bool]) -> None:
        """Add a deterministic validation hook."""
        self._deterministic_hooks.append(hook)

    def validate_deterministic(self, goal: str, ocr_text: str) -> StepResult:
        """Run deterministic validation hooks."""
        for hook in self._deterministic_hooks:
            try:
                if hook(goal, ocr_text):
                    return StepResult(ValidationResult.PASS, "Deterministic validation passed")
            except Exception as exc:
                logger.warning("Deterministic hook failed: %s", exc)
        return StepResult(ValidationResult.FAIL, "No deterministic validation passed")

    async def validate_semantic(
        self, goal: str, screenshot_b64: Optional[str], ocr_text: Optional[str]
    ) -> StepResult:
        """Run semantic validation using Vertex AI."""
        if self._vertex_client is None:
            return StepResult(ValidationResult.SKIP, "No Vertex AI client - semantic validation skipped")
        try:
            prompt = (
                f"Goal: {goal}\n"
                f"Screen text: {ocr_text or 'N/A'}\n"
                "Has the goal been achieved? Answer PASS or FAIL with a brief reason."
            )
            response = await self._vertex_client.generate_content(prompt)
            text = response.strip().upper()
            result = ValidationResult.PASS if "PASS" in text else ValidationResult.FAIL
            return StepResult(result, response)
        except Exception as exc:
            logger.error("Semantic validation failed: %s", exc)
            return StepResult(ValidationResult.ERROR, str(exc))

    def map_action_outcome(self, action_success: bool, timed_out: bool) -> ValidationResult:
        """Map action outcome to validation result."""
        if timed_out:
            return ValidationResult.TIMEOUT
        if action_success:
            return ValidationResult.PASS
        return ValidationResult.FAIL
