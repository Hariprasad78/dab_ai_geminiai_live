"""Deterministic recovery guards for no-progress loops."""

from __future__ import annotations

from typing import Iterable


def detect_repeated_loop(actions: Iterable[str], window: int = 8) -> bool:
    seq = [str(a).upper() for a in list(actions or [])[-window:]]
    if len(seq) < 4:
        return False
    captures = seq.count("CAPTURE_SCREENSHOT")
    gets = seq.count("GET_STATE")
    backs = seq.count("PRESS_BACK")
    if captures >= 3 and gets >= 2:
        return True
    if backs >= 3 and captures >= 2:
        return True
    return False
