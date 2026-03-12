"""Action schemas for the planner."""
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    """Structured action output from the planner.

    Validation rules
    ----------------
    * ``reason`` must be a non-empty string.
    * ``LAUNCH_APP`` requires ``params["app_id"]`` (non-empty string).
    * ``WAIT`` requires ``params["seconds"]`` (a number).
    """

    action: ActionType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    params: Optional[Dict[str, Any]] = None

    model_config = {"use_enum_values": True}

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reason must not be empty")
        return v

    @model_validator(mode="after")
    def check_required_params(self) -> "PlannedAction":
        """Enforce that action-specific required params are present.

        Because ``use_enum_values=True``, ``self.action`` is always a plain
        string after Pydantic validation, so we compare against the raw string
        values rather than the enum members.
        """
        action = self.action  # always a str due to use_enum_values=True
        params = self.params or {}
        if action == "LAUNCH_APP":
            app_id = params.get("app_id", "")
            if not app_id or not str(app_id).strip():
                raise ValueError(
                    "LAUNCH_APP requires params['app_id'] to be a non-empty string"
                )
        if action == "WAIT":
            if "seconds" not in params:
                raise ValueError("WAIT requires params['seconds']")
        return self
