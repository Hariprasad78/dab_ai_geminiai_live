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
    OPEN_CONTENT = "OPEN_CONTENT"
    GET_SETTING = "GET_SETTING"
    SET_SETTING = "SET_SETTING"
    GET_STATE = "GET_STATE"
    CAPTURE_SCREENSHOT = "CAPTURE_SCREENSHOT"
    WAIT = "WAIT"
    DONE = "DONE"
    FAILED = "FAILED"
    NEED_BETTER_VIEW = "NEED_BETTER_VIEW"
    NEED_PLAYER_CONTROLS_VISIBLE = "NEED_PLAYER_CONTROLS_VISIBLE"
    NEED_VIDEO_PLAYBACK_CONFIRMED = "NEED_VIDEO_PLAYBACK_CONFIRMED"
    NEED_SETTINGS_GEAR_LOCATION = "NEED_SETTINGS_GEAR_LOCATION"
    NEED_PLAYER_MENU_CONFIRMATION = "NEED_PLAYER_MENU_CONFIRMATION"
    NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION = "NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION"


class SubPlanAction(BaseModel):
    """One queued follow-up action to execute after the primary action."""

    action: ActionType
    params: Optional[Dict[str, Any]] = None

    model_config = {"use_enum_values": True}


class NavigationBatchAction(BaseModel):
    """One low-level action inside a navigation batch."""

    action: ActionType
    params: Optional[Dict[str, Any]] = None

    model_config = {"use_enum_values": True}


class ExecutionMode(str, Enum):
    # Direct DAB protocol operation (key validation, settings ops, etc.)
    DIRECT_DAB_OPERATION = "DIRECT_DAB_OPERATION"
    # Legacy alias kept for backward compatibility
    DIRECT_SETTING_OPERATION = "DIRECT_SETTING_OPERATION"
    DIRECT_APP_LAUNCH = "DIRECT_APP_LAUNCH"
    DIRECT_APP_LAUNCH_WITH_PARAMS = "DIRECT_APP_LAUNCH_WITH_PARAMS"
    DIRECT_CONTENT_OPEN = "DIRECT_CONTENT_OPEN"
    CONTINUE_IN_CURRENT_APP = "CONTINUE_IN_CURRENT_APP"
    # Navigate home first, then launch target app
    GO_HOME_AND_RECOVER = "GO_HOME_AND_RECOVER"
    # Legacy alias kept for backward compatibility
    GO_HOME_THEN_LAUNCH = "GO_HOME_THEN_LAUNCH"
    UI_NAVIGATION_ONLY = "UI_NAVIGATION_ONLY"
    # Re-launch target app after being displaced
    RELAUNCH_TARGET_APP = "RELAUNCH_TARGET_APP"
    # Legacy alias kept for backward compatibility
    RECOVERY_RELAUNCH = "RECOVERY_RELAUNCH"
    # Terminal failure with grounded explanation
    FAIL_WITH_GROUNDED_REASON = "FAIL_WITH_GROUNDED_REASON"


class StepType(str, Enum):
    DIRECT_KEY_VALIDATION = "DIRECT_KEY_VALIDATION"
    APP_LAUNCH = "APP_LAUNCH"
    CONTENT_OPEN = "CONTENT_OPEN"
    SETTING_CHANGE = "SETTING_CHANGE"
    SETTING_VERIFY = "SETTING_VERIFY"
    MENU_NAVIGATION = "MENU_NAVIGATION"
    STATE_RECOVERY = "STATE_RECOVERY"
    VISUAL_CONFIRMATION = "VISUAL_CONFIRMATION"


class TaskPrePlan(BaseModel):
    goal: str = ""
    target_app: Optional[str] = None
    target_ui_context: str = ""
    required_subgoals: list[str] = Field(default_factory=list)
    verification_condition: str = ""
    forbidden_detours: list[str] = Field(default_factory=list)
    starting_context: str = "UNKNOWN"
    required_action: str = ""
    target_domain: str = "GENERAL_UI"
    expected_outcome: str = ""
    verification_mode: str = "VISUAL"
    step_type: StepType = StepType.MENU_NAVIGATION
    needs_app_launch: bool = False
    needs_settings_navigation: bool = False
    needs_home_first: bool = False
    minimal_action_path: list[str] = Field(default_factory=list)
    selected_strategy: str = "UI_NAVIGATION_ONLY"
    reason: str = ""


class NavigationPlan(BaseModel):
    """Structured planner output for efficient multi-step TV navigation."""

    phase: str
    intent: str
    subgoal: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.UI_NAVIGATION_ONLY
    strategy: Optional[str] = None
    target_app_name: Optional[str] = None
    target_app_domain: Optional[str] = None
    target_app_hint: Optional[str] = None
    launch_parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    starting_assumption: str
    action_batch: list[NavigationBatchAction] = Field(default_factory=list)
    checkpoint_required: bool = False
    validate_before_commit: bool = False
    expected_result: str = ""
    fallback_if_failed: Optional[NavigationBatchAction] = None
    need_screenshot: bool = False
    done: bool = False
    evidence_used: list[str] = Field(default_factory=list)
    user_explanation: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_fallback_if_failed(cls, data: Any) -> Any:
        """Accept legacy fallback string and normalize to structured action."""
        if not isinstance(data, dict):
            return data
        fallback = data.get("fallback_if_failed")
        if fallback is None:
            return data
        if isinstance(fallback, str):
            action = fallback.strip()
            data["fallback_if_failed"] = {"action": action, "params": {}} if action else None
        elif isinstance(fallback, dict):
            action = str(fallback.get("action", "")).strip()
            params = fallback.get("params")
            data["fallback_if_failed"] = {
                "action": action,
                "params": params if isinstance(params, dict) else {},
            } if action else None
        else:
            data["fallback_if_failed"] = None
        return data


class PlannedAction(BaseModel):
    """Structured action output from the planner.

    Validation rules
    ----------------
    * ``reason`` must be a non-empty string.
    * ``LAUNCH_APP`` requires ``params["app_id"]`` (non-empty string).
    * ``WAIT`` requires ``params["seconds"]`` (a number).
    * ``subplan`` (if present) must not include terminal actions.
    """

    action: ActionType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    params: Optional[Dict[str, Any]] = None
    subplan: Optional[list[SubPlanAction]] = None

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
        if self.subplan:
            for idx, sub in enumerate(self.subplan):
                if sub.action in {"DONE", "FAILED"}:
                    raise ValueError(f"subplan[{idx}] cannot contain terminal action {sub.action}")
        return self
