"""Lightweight local next-action ranker backed by a distilled JSON model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


class RankedAction(BaseModel):
    action: str
    score: float
    reason: str


class LocalActionRanker:
    """Score a small set of likely next actions using a local distilled model."""

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._model = self._load_model()

    def _load_model(self) -> Dict[str, Any]:
        if not self._model_path.exists():
            return {"version": "untrained", "goal_action_counts": {}, "screen_label_action_counts": {}, "global_action_counts": {}}
        try:
            data = json.loads(self._model_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {"version": "invalid", "goal_action_counts": {}, "screen_label_action_counts": {}, "global_action_counts": {}}

    @property
    def version(self) -> str:
        return str(self._model.get("version") or "untrained")

    def rank(
        self,
        *,
        goal: str,
        current_app: str,
        observation_features: Dict[str, Any],
        retrieved_experiences: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[RankedAction]:
        scores: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {}
        goal_counts = dict(self._model.get("goal_action_counts") or {})
        label_counts = dict(self._model.get("screen_label_action_counts") or {})
        global_counts = dict(self._model.get("global_action_counts") or {})

        for token in _tokenize(goal):
            for action, count in dict(goal_counts.get(token) or {}).items():
                scores[action] = scores.get(action, 0.0) + float(count) * 1.5
                reasons.setdefault(action, []).append(f"goal-token:{token}")

        for label in list(observation_features.get("screen_labels") or []):
            for action, count in dict(label_counts.get(str(label)) or {}).items():
                scores[action] = scores.get(action, 0.0) + float(count) * 1.25
                reasons.setdefault(action, []).append(f"screen-label:{label}")

        for item in retrieved_experiences:
            action = str(item.get("action", "")).strip().upper()
            if not action:
                continue
            weight = 2.0 if str(item.get("result", "")).upper() == "PASS" else 0.5
            scores[action] = scores.get(action, 0.0) + weight
            reasons.setdefault(action, []).append("retrieved-trajectory")

        for action, count in global_counts.items():
            scores[action] = scores.get(action, 0.0) + float(count) * 0.1
            reasons.setdefault(action, []).append("global-prior")

        self._apply_heuristic_biases(scores, reasons, goal=goal, current_app=current_app, observation_features=observation_features)

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [
            RankedAction(action=action, score=round(score, 3), reason=", ".join(reasons.get(action, [])[:4]))
            for action, score in ranked[: max(1, int(top_k or 3))]
        ]

    @staticmethod
    def _apply_heuristic_biases(
        scores: Dict[str, float],
        reasons: Dict[str, List[str]],
        *,
        goal: str,
        current_app: str,
        observation_features: Dict[str, Any],
    ) -> None:
        goal_l = str(goal or "").lower()
        current_app_l = str(current_app or "").lower()
        screen_labels = set(str(item) for item in (observation_features.get("screen_labels") or []))

        def bump(action: str, amount: float, reason: str) -> None:
            scores[action] = scores.get(action, 0.0) + amount
            reasons.setdefault(action, []).append(reason)

        if "settings" in goal_l and current_app_l != "settings":
            bump("LAUNCH_APP", 3.0, "settings-goal")
        if {"timezone", "language", "accessibility", "network"} & screen_labels:
            bump("PRESS_OK", 1.2, "detail-screen-visible")
            bump("PRESS_DOWN", 0.9, "menu-navigation")
        if "youtube" in goal_l and current_app_l not in {"youtube", "com.google.android.youtube", "com.google.android.youtube.tv"}:
            bump("LAUNCH_APP", 2.5, "youtube-goal")
        if observation_features.get("player_like"):
            bump("PRESS_OK", 1.1, "player-surface")
            bump("PRESS_RIGHT", 0.8, "player-controls")
