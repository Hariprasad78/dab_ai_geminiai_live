"""Action schemas for the planner."""
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    PRESS_UP = "PRESS_UP"
    PRESS_DOWN = "PRESS_DOWN"
    PRESS_LEFT = "PRESS_LEFT"
    PRESS_RIGHT = "PRESS_RIGHT"
    PRESS_OK = "PRESS_OK"
    PRESS_BACK = "PRESS_BACK"
    PRESS_HOME = "PRESS_HOME"
    LAUNCH_APP = "LAUNCH_APP"
    GET_STATE = "GET_STATE"
    CAPTURE_SCREENSHOT = "CAPTURE_SCREENSHOT"
    WAIT = "WAIT"
    DONE = "DONE"
    FAILED = "FAILED"
    NEED_BETTER_VIEW = "NEED_BETTER_VIEW"


class PlannedAction(BaseModel):
    """Structured action output from the planner."""

    action: ActionType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    params: Optional[Dict[str, Any]] = None

    model_config = {"use_enum_values": True}
