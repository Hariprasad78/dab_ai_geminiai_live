"""RunState model for orchestration."""
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class ActionRecord(BaseModel):
    step: int
    action: str
    params: Optional[Dict[str, Any]] = None
    confidence: float
    reason: str
    result: str
    timestamp: str


class RunState(BaseModel):
    """Complete state for a single run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    status: RunStatus = RunStatus.PENDING
    current_app: Optional[str] = None
    launch_content: Optional[str] = None
    current_screen: Optional[str] = None
    current_screen_guess: Optional[str] = None
    highlighted_item_guess: Optional[str] = None
    focus_target_guess: Optional[str] = None
    nav_confidence: float = 0.0
    supported_operations: List[str] = Field(default_factory=list)
    app_catalog: List[Dict[str, Any]] = Field(default_factory=list)
    supported_keys: List[str] = Field(default_factory=list)
    supported_settings: List[Dict[str, Any]] = Field(default_factory=list)
    resolved_apps_cache: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    current_app_state: Optional[str] = None
    current_app_id: Optional[str] = None
    last_verified_foreground_app: Optional[str] = None
    strategy_selected: Optional[str] = None
    resolution_failures: int = 0
    task_preplan: Dict[str, Any] = Field(default_factory=dict)
    verification_mode: Optional[str] = None
    last_expected_screen: Optional[str] = None
    stuck_diagnosis_count: int = 0
    last_stuck_context: Dict[str, Any] = Field(default_factory=dict)
    failed_paths: List[str] = Field(default_factory=list)
    last_checkpoint: Optional[str] = None
    master_plan: List[str] = Field(default_factory=list)
    last_actions: List[str] = Field(default_factory=list)
    latest_screenshot_b64: Optional[str] = None
    latest_ocr_text: Optional[str] = None
    last_screen_fingerprint: Optional[str] = None
    is_video_playback_context: bool = False
    player_controls_visible: bool = False
    last_ok_effect: Optional[str] = None
    repeated_commit_count: int = 0
    no_progress_count: int = 0
    last_player_phase: Optional[str] = None
    steps_since_observe: int = 0
    pending_subplan: List[Dict[str, Any]] = Field(default_factory=list)
    action_history: List[ActionRecord] = Field(default_factory=list)
    dab_transcript: List[Dict[str, Any]] = Field(default_factory=list)
    ai_transcript: List[Dict[str, Any]] = Field(default_factory=list)
    narration_transcript: List[Dict[str, Any]] = Field(default_factory=list)
    retries: int = 0
    step_count: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    artifacts_dir: Optional[str] = None

    def start(self) -> None:
        self.status = RunStatus.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()

    def finish(self, status: RunStatus, error: Optional[str] = None) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if error:
            self.error = error

    def record_action(
        self,
        action: str,
        params: Optional[Dict[str, Any]],
        confidence: float,
        reason: str,
        result: str,
    ) -> None:
        record = ActionRecord(
            step=self.step_count,
            action=action,
            params=params,
            confidence=confidence,
            reason=reason,
            result=result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.action_history.append(record)
        self.last_actions.append(action)
        if len(self.last_actions) > 10:
            self.last_actions = self.last_actions[-10:]
        self.step_count += 1

    def record_dab_event(self, event: Dict[str, Any]) -> None:
        self.dab_transcript.append(event)
        if len(self.dab_transcript) > 500:
            self.dab_transcript = self.dab_transcript[-500:]

    def record_ai_event(self, event: Dict[str, Any]) -> None:
        self.ai_transcript.append(event)
        if len(self.ai_transcript) > 500:
            self.ai_transcript = self.ai_transcript[-500:]

    def record_narration_event(self, event: Dict[str, Any]) -> None:
        self.narration_transcript.append(event)
        if len(self.narration_transcript) > 300:
            self.narration_transcript = self.narration_transcript[-300:]
