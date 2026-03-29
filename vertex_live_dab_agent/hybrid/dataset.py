"""Dataset schema for local navigation training examples."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalTrainingExample(BaseModel):
    """One normalized training row for local ranking models."""

    recorded_at: str = Field(default_factory=_utc_now_iso)
    run_id: str
    step: int
    goal: str
    device_id: str
    device_profile_id: Optional[str] = None
    hybrid_policy_mode: Optional[str] = None
    current_app: str = ""
    current_screen: str = ""
    visual_summary_before: str = ""
    visual_summary_after: str = ""
    observation_features: Dict[str, Any] = Field(default_factory=dict)
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    result: str
    strategy_selected: Optional[str] = None
    retrieved_actions: List[str] = Field(default_factory=list)
    local_ranker_actions: List[str] = Field(default_factory=list)
    reason: str = ""
