"""Execution safety guards shared by orchestrator and YTS flows."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


_TRUSTED_KEY_FALLBACK = {
    "PRESS_UP",
    "PRESS_DOWN",
    "PRESS_LEFT",
    "PRESS_RIGHT",
    "PRESS_OK",
    "PRESS_BACK",
    "PRESS_HOME",
}


def enforce_action_guards(*, action: str, ui_navigation_allowed: bool, chosen_route: str, supported_keys: Iterable[str], snapshot: Dict[str, object], is_android: bool, requested_route: str) -> Tuple[bool, str]:
    action_u = str(action or "").upper()
    route = str(requested_route or "").strip().upper()

    if not route:
        return True, "ok"

    keys = {str(k).strip().upper() for k in (supported_keys or []) if str(k).strip()}
    ops = {str(o).strip().lower() for o in (snapshot.get("supported_operations") or [])}

    if route == "ADB_FALLBACK" and not bool(is_android):
        return False, "ADB route forbidden on non-Android platform"

    if route == "UI_NAVIGATION_FALLBACK" and not bool(ui_navigation_allowed):
        return False, "UI fallback forbidden because checkbox is disabled"

    if action_u.startswith("PRESS_"):
        if keys:
            return True, "ok"
        if "input/key-press" in ops and action_u in _TRUSTED_KEY_FALLBACK:
            return True, "ok (trusted key fallback)"
        return False, "keypress forbidden: supported key list empty and no trusted fallback"

    return True, "ok"
