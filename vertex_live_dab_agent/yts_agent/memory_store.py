"""Persistent JSON memory for YTS guided validation runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(fallback)
    return parsed if isinstance(parsed, dict) else dict(fallback)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


class YtsMemoryStore:
    """JSON-backed memory. Current live evidence always overrides this context."""

    def __init__(self, base_dir: str | Path):
        self.root = Path(base_dir).expanduser()
        self.tests_dir = self.root / "tests"
        self.runs_dir = self.root / "runs"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _test_path(self, test_key: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(test_key or "unknown"))[:120]
        return self.tests_dir / f"{safe or 'unknown'}.json"

    def _run_path(self, run_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(run_id or "unknown"))[:120]
        return self.runs_dir / safe / "memory.json"

    def load_test_memory(self, test_key: str) -> Dict[str, Any]:
        return _read_json(self._test_path(test_key), {})

    def load_run_memory(self, run_id: str) -> Dict[str, Any]:
        return _read_json(self._run_path(run_id), {})

    def start_run(self, run_id: str, *, test_name: str = "", test_version: str = "", command: str = "") -> Dict[str, Any]:
        payload = self.load_run_memory(run_id)
        if not payload:
            payload = {
                "run_id": run_id,
                "started_at": _now(),
                "test_version": test_version,
                "test_name": test_name,
                "command": command,
                "console_prompt_events": [],
                "visual_observations": [],
                "agent_decisions": [],
                "final_summary": {},
            }
        _write_json(self._run_path(run_id), payload)
        self._update_index(run_id=run_id, test_key="")
        return payload

    def record_prompt(
        self,
        *,
        run_id: str,
        test_key: str,
        test_title: str,
        prompt_hash: str,
        prompt_text: str,
        options: List[Dict[str, Any]],
        extracted_expectation: Dict[str, Any],
        test_version: str = "",
    ) -> Dict[str, Any]:
        now = _now()
        test_payload = self.load_test_memory(test_key)
        if not test_payload:
            test_payload = {
                "test_key": test_key,
                "test_title": test_title,
                "first_seen_at": now,
                "last_seen_at": now,
                "version_of_tests": test_version,
                "prompt_history": [],
                "successful_evidence_patterns": [],
                "known_failure_patterns": [],
                "previous_decisions": [],
            }
        test_payload["test_title"] = test_payload.get("test_title") or test_title
        test_payload["last_seen_at"] = now
        test_payload["version_of_tests"] = test_version or test_payload.get("version_of_tests") or ""
        prompt_history = list(test_payload.get("prompt_history") or [])
        existing = next((item for item in prompt_history if item.get("prompt_hash") == prompt_hash), None)
        prompt_record = {
            "prompt_hash": prompt_hash,
            "prompt_text": prompt_text,
            "options": options,
            "extracted_expectation": extracted_expectation,
            "last_seen_at": now,
        }
        if existing:
            existing.update(prompt_record)
        else:
            prompt_record["first_seen_at"] = now
            prompt_history.append(prompt_record)
        test_payload["prompt_history"] = prompt_history
        _write_json(self._test_path(test_key), test_payload)

        run_payload = self.load_run_memory(run_id) or self.start_run(run_id)
        run_payload.setdefault("console_prompt_events", []).append(
            {
                "timestamp": now,
                "test_key": test_key,
                "prompt_hash": prompt_hash,
                "prompt_text": prompt_text,
                "options": options,
                "extracted_expectation": extracted_expectation,
            }
        )
        _write_json(self._run_path(run_id), run_payload)
        self._update_index(run_id=run_id, test_key=test_key)
        return test_payload

    def record_visual_observations(self, run_id: str, observations: List[Dict[str, Any]]) -> None:
        run_payload = self.load_run_memory(run_id) or self.start_run(run_id)
        run_payload["visual_observations"] = list(observations or [])[-500:]
        _write_json(self._run_path(run_id), run_payload)

    def record_decision(self, *, run_id: str, test_key: str, decision: Dict[str, Any], evidence_summary: str = "") -> None:
        now = _now()
        decision_record = dict(decision)
        decision_record["timestamp"] = now
        decision_record["evidence_summary"] = decision_record.get("evidence_summary") or evidence_summary
        run_payload = self.load_run_memory(run_id) or self.start_run(run_id)
        run_payload.setdefault("agent_decisions", []).append(decision_record)
        _write_json(self._run_path(run_id), run_payload)

        test_payload = self.load_test_memory(test_key)
        if test_payload:
            test_payload.setdefault("previous_decisions", []).append(
                {
                    "run_id": run_id,
                    "selected_option": decision.get("selected_option"),
                    "selected_label": decision.get("selected_label"),
                    "confidence": decision.get("confidence"),
                    "reason": decision.get("reason"),
                    "evidence_summary": decision_record.get("evidence_summary"),
                    "final_outcome": str(decision.get("selected_label") or "unknown").lower(),
                    "timestamp": now,
                }
            )
            test_payload["previous_decisions"] = test_payload["previous_decisions"][-100:]
            _write_json(self._test_path(test_key), test_payload)

    def finalize_run(self, run_id: str, summary: Dict[str, Any]) -> None:
        run_payload = self.load_run_memory(run_id) or self.start_run(run_id)
        run_payload["final_summary"] = dict(summary or {})
        run_payload["finished_at"] = _now()
        _write_json(self._run_path(run_id), run_payload)

    def _update_index(self, *, run_id: str, test_key: str) -> None:
        index = _read_json(self.index_path, {"runs": [], "tests": []})
        if run_id:
            runs = list(index.get("runs") or [])
            if run_id not in runs:
                runs.append(run_id)
            index["runs"] = runs[-500:]
        if test_key:
            tests = list(index.get("tests") or [])
            if test_key not in tests:
                tests.append(test_key)
            index["tests"] = tests[-500:]
        index["updated_at"] = _now()
        _write_json(self.index_path, index)
