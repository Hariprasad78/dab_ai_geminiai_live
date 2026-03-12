"""Pydantic request/response models for the API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StartRunRequest(BaseModel):
    goal: str
    app_id: Optional[str] = None
    max_steps: Optional[int] = None


class StartRunResponse(BaseModel):
    run_id: str
    status: str
    goal: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    goal: str
    step_count: int
    retries: int
    current_app: Optional[str] = None
    current_screen: Optional[str] = None
    last_actions: List[str] = []
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    has_screenshot: bool = False


class ManualActionRequest(BaseModel):
    action: str
    params: Optional[Dict[str, Any]] = None


class ManualActionResponse(BaseModel):
    success: bool
    action: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PlannerDebugRequest(BaseModel):
    goal: str
    ocr_text: Optional[str] = None
    current_app: Optional[str] = None
    current_screen: Optional[str] = None
    last_actions: Optional[List[str]] = None


class PlannerDebugResponse(BaseModel):
    action: str
    confidence: float
    reason: str
    params: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    mock_mode: bool
    version: str = "1.0.0"


class ConfigSummaryResponse(BaseModel):
    google_cloud_project: str
    google_cloud_location: str
    vertex_live_model: str
    dab_mock_mode: bool
    dab_device_id: str
    max_steps_per_run: int
    artifacts_base_dir: str
