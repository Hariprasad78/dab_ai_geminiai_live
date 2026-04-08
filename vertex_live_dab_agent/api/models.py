"""Pydantic request/response models for the API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    goal: str
    app_id: Optional[str] = None
    device_id: Optional[str] = None
    content: Optional[str] = None
    device_profile_id: Optional[str] = None
    policy_mode: Optional[str] = None
    ui_navigation_allowed: bool = False
    max_steps: Optional[int] = Field(default=None, gt=0, description="Override max AI planning requests for this run (capped at 50)")


class StartRunResponse(BaseModel):
    run_id: str
    status: str
    goal: str


class RunSummaryItem(BaseModel):
    run_id: str
    goal: str
    status: str
    step_count: int
    started_at: Optional[str] = None


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    goal: str
    step_count: int
    ai_request_count: int = 0
    ui_navigation_allowed: bool = False
    retries: int
    current_app: Optional[str] = None
    current_screen: Optional[str] = None
    last_actions: List[str] = []
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    has_screenshot: bool = False
    artifacts_dir: Optional[str] = None
    dab_log_count: int = 0
    dab_logs_tail: List[Dict[str, Any]] = []
    ai_log_count: int = 0
    ai_logs_tail: List[Dict[str, Any]] = []
    narration_count: int = 0
    narration_tail: List[Dict[str, Any]] = []
    device_profile_id: Optional[str] = None
    hybrid_policy_mode: Optional[str] = None
    hybrid_policy_rationale: Optional[str] = None
    retrieved_experiences: List[Dict[str, Any]] = []
    observation_features: Dict[str, Any] = {}
    local_action_suggestions: List[Dict[str, Any]] = []
    local_model_version: Optional[str] = None


class ActionRecordItem(BaseModel):
    step: int
    action: str
    params: Optional[Dict[str, Any]] = None
    confidence: float
    reason: str
    result: str
    timestamp: str


class ActionHistoryResponse(BaseModel):
    run_id: str
    goal: str
    action_count: int
    actions: List[ActionRecordItem]


class DABTranscriptResponse(BaseModel):
    run_id: str
    goal: str
    count: int
    events: List[Dict[str, Any]]


class AITranscriptResponse(BaseModel):
    run_id: str
    goal: str
    count: int
    events: List[Dict[str, Any]]


class FriendlyStepItem(BaseModel):
    step: int
    title: str
    simple_action: str
    what_happened: str
    what_screen_was_seen: str = ""
    why_this_step: str
    why_it_failed: Optional[str] = None
    what_recovery_was_tried: Optional[str] = None
    what_happens_next: Optional[str] = None
    result: str
    simple_status: str


class FinalDiagnosis(BaseModel):
    status: str
    final_summary: str
    root_cause: str
    user_friendly_reason: str
    technical_reason: str
    recovery_attempts: int
    what_failed_first: Optional[str] = None
    what_failed_last: Optional[str] = None
    failure_reason_short: Optional[str] = None
    failure_reason_detailed: Optional[str] = None
    failure_reason_user_friendly: Optional[str] = None
    screen_based_reason: Optional[str] = None
    goal_based_reason: Optional[str] = None
    recovery_summary: Optional[str] = None
    evidence_used: List[str] = []


class FriendlyRunExplanationResponse(BaseModel):
    run_id: str
    goal: str
    status: str
    timeline: List[FriendlyStepItem]
    diagnosis: FinalDiagnosis


class NarrationEventItem(BaseModel):
    idx: int
    step: int
    tts_text: str
    tts_priority: int
    tts_category: str
    tts_should_play: bool
    tts_interruptible: bool


class NarrationResponse(BaseModel):
    run_id: str
    goal: str
    count: int
    events: List[NarrationEventItem]


class TTSSpeakRequest(BaseModel):
    text: str
    use_ssml: Optional[bool] = None


class TTSSpeakResponse(BaseModel):
    success: bool
    audio_b64: Optional[str] = None
    voice_name: Optional[str] = None
    language_code: Optional[str] = None
    error: Optional[str] = None


class ManualActionRequest(BaseModel):
    action: str
    params: Optional[Dict[str, Any]] = None
    device_id: Optional[str] = None
    control_mode: Optional[str] = None
    ir_device_id: Optional[str] = None


class ManualActionResponse(BaseModel):
    success: bool
    action: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ManualActionBatchRequest(BaseModel):
    actions: List[ManualActionRequest]
    continue_on_error: bool = True


class ManualActionBatchResponse(BaseModel):
    success: bool
    total: int
    results: List[ManualActionResponse]


class TaskMacroRequest(BaseModel):
    instruction: str
    execute: bool = False
    continue_on_error: bool = True


class TaskMacroResponse(BaseModel):
    success: bool
    instruction: str
    planned_count: int
    planned_actions: List[ManualActionRequest]
    execution: Optional[ManualActionBatchResponse] = None


class PlannerDebugRequest(BaseModel):
    goal: str
    device_id: Optional[str] = None
    ocr_text: Optional[str] = None
    screenshot_b64: Optional[str] = None
    use_live_capture: bool = False
    current_app: Optional[str] = None
    current_screen: Optional[str] = None
    last_actions: Optional[List[str]] = None


class PlannerDebugResponse(BaseModel):
    action: str
    confidence: float
    reason: str
    params: Optional[Dict[str, Any]] = None
    used_live_capture: bool = False
    capture_source: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    mock_mode: bool
    version: str = "1.0.0"


class ConfigSummaryResponse(BaseModel):
    google_cloud_project: str
    google_cloud_location: str
    vertex_planner_model: str
    vertex_live_model: str
    enable_livekit_agent: bool
    dab_mock_mode: bool
    image_source: str
    youtube_app_id: str
    dab_device_id: str
    max_steps_per_run: int
    artifacts_base_dir: str
    log_level: str
    tts_enabled: bool
    tts_voice_provider: str
    tts_model: str
    tts_voice_name: str
    tts_language_code: str


class RuntimeModelUpdateRequest(BaseModel):
    model: str


class RuntimeModelResponse(BaseModel):
    success: bool
    active_vertex_planner_model: str
    configured_vertex_planner_model: str
    available_models: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class CaptureSourceResponse(BaseModel):
    configured_source: str
    hdmi_configured: bool
    hdmi_available: bool
    hdmi_device: Optional[str] = None
    hdmi_info: Dict[str, float] = Field(default_factory=dict)
    enable_hdmi_capture: bool = True
    enable_camera_capture: bool = True
    selected_video_device: Optional[str] = None
    rotation_degrees: int = 0
    preferred_video_kind: str = "auto"
    effective_preferred_kind: str = "auto"
    video_devices: List[str] = Field(default_factory=list)
    video_device_details: List[Dict[str, Any]] = Field(default_factory=list)
    device_readable: Dict[str, bool] = Field(default_factory=dict)
    user_in_video_group: Optional[bool] = None


class CaptureSelectRequest(BaseModel):
    source: Optional[str] = None
    device: Optional[str] = None
    preferred_kind: Optional[str] = None
    rotation_degrees: Optional[int] = None
    persist: bool = True
