"""Shared execution context models for capability-aware planning and execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CapabilitySnapshotModel(BaseModel):
    """Normalized capability view consumed by planner and executor."""

    supported_operations: List[str] = Field(default_factory=list)
    supported_settings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    supported_keys: List[str] = Field(default_factory=list)
    supported_voices: List[str] = Field(default_factory=list)
    installed_applications: List[Dict[str, Any]] = Field(default_factory=list)
    platform_type: str = "unknown"
    is_android: bool = False
    can_use_adb: bool = False
    unsupported_or_missing_capabilities: List[str] = Field(default_factory=list)


class AgentExecutionContext(BaseModel):
    """Single state object used across planner and execution policy."""

    goal: str
    device_id: str = ""
    capability_snapshot: CapabilitySnapshotModel = Field(default_factory=CapabilitySnapshotModel)
    current_observation: Dict[str, Any] = Field(default_factory=dict)
    previous_actions: List[str] = Field(default_factory=list)
    previous_results: List[Dict[str, Any]] = Field(default_factory=list)
    active_plan: List[str] = Field(default_factory=list)
    active_subplan: List[Dict[str, Any]] = Field(default_factory=list)
    transport_availability: Dict[str, bool] = Field(default_factory=dict)
    retry_history: List[Dict[str, Any]] = Field(default_factory=list)
    recovery_state: Dict[str, Any] = Field(default_factory=dict)
    blocked_actions: List[str] = Field(default_factory=list)

    def compact(self) -> Dict[str, Any]:
        """Return compact JSON-serializable dict for prompt builders."""
        return {
            "goal": self.goal,
            "device_id": self.device_id,
            "capability_snapshot": self.capability_snapshot.model_dump(),
            "current_observation": self.current_observation,
            "previous_actions": self.previous_actions[-8:],
            "previous_results": self.previous_results[-8:],
            "active_plan": self.active_plan,
            "active_subplan": self.active_subplan,
            "transport_availability": self.transport_availability,
            "retry_history": self.retry_history[-8:],
            "recovery_state": self.recovery_state,
            "blocked_actions": self.blocked_actions[-12:],
        }
