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
    current_screen: Optional[str] = None
    last_actions: List[str] = Field(default_factory=list)
    latest_screenshot_b64: Optional[str] = None
    latest_ocr_text: Optional[str] = None
    action_history: List[ActionRecord] = Field(default_factory=list)
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
