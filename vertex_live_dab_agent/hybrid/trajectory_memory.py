"""Simple local trajectory memory for successful and failed runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(str(text or "").lower()) if token}


@dataclass
class ExperienceQuery:
    goal: str
    device_id: str
    current_app: str = ""
    limit: int = 5


class TrajectoryMemory:
    """Appends JSONL trajectory records and retrieves similar examples."""

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._file_path.exists():
            self._file_path.write_text("", encoding="utf-8")

    @property
    def file_path(self) -> Path:
        return self._file_path

    def append(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        with self._file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def find_similar(self, query: ExperienceQuery) -> List[Dict[str, Any]]:
        goal_tokens = _tokenize(query.goal)
        app_target = str(query.current_app or "").strip().lower()
        device_id = str(query.device_id or "").strip()
        scored: List[tuple[float, Dict[str, Any]]] = []
        for item in self._read_all():
            if device_id and str(item.get("device_id", "")).strip() != device_id:
                continue
            score = 0.0
            item_goal_tokens = _tokenize(str(item.get("goal", "")))
            overlap = len(goal_tokens & item_goal_tokens)
            if overlap:
                score += overlap * 3.0
            if app_target and str(item.get("current_app", "")).strip().lower() == app_target:
                score += 2.0
            if str(item.get("result", "")).upper() == "PASS":
                score += 0.75
            if item.get("strategy_selected"):
                score += 0.25
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: max(1, int(query.limit or 5))]]

    def _read_all(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line in self._file_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows
