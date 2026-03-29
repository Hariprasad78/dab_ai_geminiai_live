"""Lightweight hybrid policy recommendations for the orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PolicyRecommendation(BaseModel):
    mode: str
    rationale: str
    preferred_operations: List[str] = Field(default_factory=list)
    retrieved_examples_count: int = 0
    device_profile_id: str | None = None


class HybridPolicyEngine:
    """Derive a planning mode from device capability + local experience."""

    def recommend(
        self,
        *,
        goal: str,
        device_profile: Dict[str, Any] | None,
        similar_experiences: List[Dict[str, Any]],
    ) -> PolicyRecommendation:
        goal_l = str(goal or "").lower()
        features = dict((device_profile or {}).get("features") or {})

        if any(token in goal_l for token in ("timezone", "time zone", "language", "wifi", "network")):
            if features.get("direct_setting_set") or features.get("direct_setting_get"):
                return PolicyRecommendation(
                    mode="DIRECT_DAB_PREFERRED",
                    rationale="Device exposes direct system settings operations for this goal.",
                    preferred_operations=["system/settings/set", "system/settings/get"],
                    retrieved_examples_count=len(similar_experiences),
                    device_profile_id=(device_profile or {}).get("profile_id"),
                )

        if similar_experiences:
            return PolicyRecommendation(
                mode="LOCAL_MEMORY_ASSISTED",
                rationale="Use retrieved local trajectories to bias navigation before cloud planning.",
                preferred_operations=self._top_actions(similar_experiences),
                retrieved_examples_count=len(similar_experiences),
                device_profile_id=(device_profile or {}).get("profile_id"),
            )

        if features.get("direct_app_launch") or features.get("direct_content_open"):
            return PolicyRecommendation(
                mode="HYBRID_DIRECT_FIRST",
                rationale="Prefer direct app/content operations and fall back to UI navigation only when needed.",
                preferred_operations=["applications/launch", "content/open"],
                retrieved_examples_count=0,
                device_profile_id=(device_profile or {}).get("profile_id"),
            )

        return PolicyRecommendation(
            mode="UI_NAVIGATION_HEAVY",
            rationale="No strong direct-ops shortcut available; rely on observe-plan-act-verify with visual checkpoints.",
            preferred_operations=[],
            retrieved_examples_count=len(similar_experiences),
            device_profile_id=(device_profile or {}).get("profile_id"),
        )

    @staticmethod
    def _top_actions(similar_experiences: List[Dict[str, Any]]) -> List[str]:
        counts: Dict[str, int] = {}
        for item in similar_experiences:
            action = str(item.get("action", "")).strip().upper()
            if action:
                counts[action] = counts.get(action, 0) + 1
        ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        return [name for name, _ in ranked[:3]]
