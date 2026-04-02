"""Deterministic device-agent context and state machine models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    NORMALIZE_REQUEST = "NORMALIZE_REQUEST"
    DISCOVER_CAPABILITIES = "DISCOVER_CAPABILITIES"
    RESOLVE_REQUIREMENT = "RESOLVE_REQUIREMENT"
    SELECT_ROUTE = "SELECT_ROUTE"
    EXECUTE_DIRECT = "EXECUTE_DIRECT"
    EXECUTE_ADB = "EXECUTE_ADB"
    EXECUTE_UI = "EXECUTE_UI"
    VERIFY_RESULT = "VERIFY_RESULT"
    RECOVER = "RECOVER"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class DeviceAgentContext(BaseModel):
    raw_user_text: str
    corrected_user_text: str = ""
    parsed_intent: Dict[str, Any] = Field(default_factory=dict)
    canonical_target: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    capability_snapshot: Dict[str, Any] = Field(default_factory=dict)
    selected_route: Optional[str] = None
    rejected_routes: Dict[str, str] = Field(default_factory=dict)
    verification_status: str = "not-required"
    state: AgentState = AgentState.NORMALIZE_REQUEST
    evidence: List[str] = Field(default_factory=list)
