#!/usr/bin/env python3
"""Offline trainer for the lightweight local action ranker."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def train_model(input_path: Path, output_path: Path) -> Dict[str, Any]:
    goal_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    screen_label_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    global_action_counts: Dict[str, int] = defaultdict(int)
    samples = 0

    for line in input_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        if str(item.get("result", "")).upper() != "PASS":
            continue
        action = str(item.get("action", "")).strip().upper()
        if not action:
            continue
        samples += 1
        global_action_counts[action] += 1
        for token in tokenize(str(item.get("goal", ""))):
            goal_action_counts[token][action] += 1
        observation_features = dict(item.get("observation_features") or {})
        for label in observation_features.get("screen_labels") or []:
            screen_label_action_counts[str(label)][action] += 1

    model = {
        "version": f"local-ranker-{utc_now_iso()}",
        "generated_at": utc_now_iso(),
        "samples": samples,
        "goal_action_counts": {k: dict(v) for k, v in goal_action_counts.items()},
        "screen_label_action_counts": {k: dict(v) for k, v in screen_label_action_counts.items()},
        "global_action_counts": dict(global_action_counts),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=True, indent=2), encoding="utf-8")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight local action ranker from trajectory JSONL.")
    parser.add_argument("input", help="Path to trajectories.jsonl")
    parser.add_argument("output", help="Path to output model JSON")
    args = parser.parse_args()

    model = train_model(Path(args.input), Path(args.output))
    print(json.dumps({"ok": True, "samples": model["samples"], "output": args.output}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
