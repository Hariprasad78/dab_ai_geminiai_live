from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class IrDeviceRecord:
    device_id: str
    brand: str
    model: str
    sender_channel: str
    key_count: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalized_key_token(value: str) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


_KEY_ALIASES: Dict[str, List[str]] = {
    "power": ["power"],
    "home": ["home"],
    "back": ["back", "return"],
    "menu": ["menu", "settings"],
    "input": ["input", "source"],
    "up": ["up", "dpadup"],
    "down": ["down", "dpaddown"],
    "left": ["left", "dpadleft"],
    "right": ["right", "dpadright"],
    "enter": ["enter", "ok", "select", "center", "dpadcenter"],
    "ok": ["ok", "enter", "select", "center", "dpadcenter"],
    "center": ["center", "ok", "enter", "select", "dpadcenter"],
    "mute": ["mute"],
    "volup": ["volup", "volumeup", "volplus"],
    "voldown": ["voldown", "volumedown", "volminus"],
    "chup": ["chup", "channelup"],
    "chdown": ["chdown", "channeldown"],
    "play": ["play"],
    "pause": ["pause"],
    "stop": ["stop"],
    "rewind": ["rewind", "rev"],
    "fastforward": ["fastforward", "ff"],
    "playpause": ["playpause"],
}


def _build_alias_index(aliases: Dict[str, List[str]]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for canonical, terms in aliases.items():
        normalized_terms = [str(canonical)] + [str(item) for item in terms]
        unique_terms = [item for item in dict.fromkeys(normalized_terms) if item]
        for term in unique_terms:
            token = _normalized_key_token(term)
            if not token:
                continue
            existing = index.get(token, [])
            index[token] = [item for item in dict.fromkeys(existing + unique_terms) if item]
    return index


_ALIAS_INDEX: Dict[str, List[str]] = _build_alias_index(_KEY_ALIASES)


def _lookup_candidates(key_name: str) -> List[str]:
    raw = str(key_name or "").strip()
    if not raw:
        return []
    candidates: List[str] = [raw, raw.lower(), raw.upper()]
    token = _normalized_key_token(raw)
    if token:
        candidates.append(token)
        alias_terms = _ALIAS_INDEX.get(token)
        if alias_terms:
            for term in alias_terms:
                candidates.extend([term, term.lower(), term.upper()])
        if token.startswith("num") and len(token) == 4 and token[3].isdigit():
            candidates.extend([f"NUM_{token[3]}", token[3]])
    unique: List[str] = []
    seen = set()
    for item in candidates:
        marker = str(item or "").strip()
        if not marker:
            continue
        key = marker.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(marker)
    return unique


class IrDatasetStore:
    """JSON-backed IR dataset store (Raspberry Pi is source of truth)."""

    def __init__(self, dataset_path: Path) -> None:
        self._path = Path(dataset_path)
        self._lock = threading.Lock()

    def _default_payload(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "devices": {
                "samsung_tv_default": {
                    "brand": "Samsung",
                    "model": "Samsung TV (Default)",
                    "sender_channel": "D2",
                    "keys": {},
                }
            },
        }

    def _load_unlocked(self) -> Dict[str, Any]:
        if not self._path.exists():
            payload = self._default_payload()
            self._save_unlocked(payload)
            return payload
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("devices", {})
                if "samsung_tv_default" not in payload["devices"]:
                    payload["devices"]["samsung_tv_default"] = {
                        "brand": "Samsung",
                        "model": "Samsung TV (Default)",
                        "sender_channel": "D2",
                        "keys": {},
                    }
                return payload
        except Exception:
            pass
        payload = self._default_payload()
        self._save_unlocked(payload)
        return payload

    def _save_unlocked(self, payload: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = _utc_now_iso()
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    @staticmethod
    def _extract_key_map(record: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        keys = record.get("keys") if isinstance(record.get("keys"), dict) else {}
        codes = record.get("codes") if isinstance(record.get("codes"), dict) else {}
        for key, value in {**codes, **keys}.items():
            if isinstance(value, dict):
                merged[str(key)] = value
        return merged

    @staticmethod
    def _match_key_payload(key_map: Dict[str, Any], key_name: str) -> Optional[Dict[str, Any]]:
        if not key_map:
            return None
        for candidate in _lookup_candidates(key_name):
            value = key_map.get(candidate)
            if isinstance(value, dict):
                return value
            lowered_match = next(
                (
                    value2
                    for key2, value2 in key_map.items()
                    if str(key2).strip().lower() == str(candidate).strip().lower() and isinstance(value2, dict)
                ),
                None,
            )
            if isinstance(lowered_match, dict):
                return lowered_match
        return None

    @staticmethod
    def _is_samsung_record(device_id: str, record: Dict[str, Any]) -> bool:
        brand = str(record.get("brand") or "").strip().lower()
        did = str(device_id or "").strip().lower()
        return brand == "samsung" or "samsung" in did

    def list_devices(self) -> List[IrDeviceRecord]:
        with self._lock:
            payload = self._load_unlocked()
            rows: List[IrDeviceRecord] = []
            for device_id, raw in (payload.get("devices") or {}).items():
                if not isinstance(raw, dict):
                    continue
                key_map = self._extract_key_map(raw)
                rows.append(
                    IrDeviceRecord(
                        device_id=str(device_id),
                        brand=str(raw.get("brand") or "UNKNOWN"),
                        model=str(raw.get("model") or device_id),
                        sender_channel=str(raw.get("sender_channel") or "D2"),
                        key_count=len(key_map),
                    )
                )
            rows.sort(key=lambda item: item.device_id)
            return rows

    def list_keys(self, device_id: str) -> List[str]:
        with self._lock:
            payload = self._load_unlocked()
            devices = payload.get("devices") or {}
            record = devices.get(device_id)
            if not isinstance(record, dict):
                return []
            key_map = self._extract_key_map(record)
            if key_map:
                return sorted(str(k) for k in key_map.keys())
            # Backward-compatibility: many captures are stored under `samsung`
            # while UI defaults to `samsung_tv_default`.
            if str(device_id).strip().lower() == "samsung_tv_default":
                for candidate_id, candidate_record in devices.items():
                    if str(candidate_id) == str(device_id) or not isinstance(candidate_record, dict):
                        continue
                    if not self._is_samsung_record(str(candidate_id), candidate_record):
                        continue
                    candidate_map = self._extract_key_map(candidate_record)
                    if candidate_map:
                        return sorted(str(k) for k in candidate_map.keys())
            return []

    def get_key_payload(self, device_id: str, key_name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            payload = self._load_unlocked()
            devices = payload.get("devices") or {}
            record = devices.get(device_id)
            if not isinstance(record, dict):
                return None
            key_map = self._extract_key_map(record)
            direct_match = self._match_key_payload(key_map, key_name)
            if isinstance(direct_match, dict):
                return direct_match
            if str(device_id).strip().lower() == "samsung_tv_default":
                for candidate_id, candidate_record in devices.items():
                    if str(candidate_id) == str(device_id) or not isinstance(candidate_record, dict):
                        continue
                    if not self._is_samsung_record(str(candidate_id), candidate_record):
                        continue
                    fallback_map = self._extract_key_map(candidate_record)
                    fallback_match = self._match_key_payload(fallback_map, key_name)
                    if isinstance(fallback_match, dict):
                        return fallback_match
            return None

    def upsert_key_payload(self, device_id: str, key_name: str, key_payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            payload = self._load_unlocked()
            devices = payload.setdefault("devices", {})
            if not isinstance(devices, dict):
                payload["devices"] = {}
                devices = payload["devices"]
            record = devices.get(device_id)
            if not isinstance(record, dict):
                record = {
                    "brand": "UNKNOWN",
                    "model": "Generic TV",
                    "sender_channel": "D2",
                    "codes": {},
                }
                devices[device_id] = record
            # Preserve existing schema style: prefer `codes` if dataset uses it,
            # otherwise keep compatibility with legacy `keys`.
            if isinstance(record.get("codes"), dict) or not isinstance(record.get("keys"), dict):
                keys = record.get("codes") if isinstance(record.get("codes"), dict) else {}
                record["codes"] = keys
            else:
                keys = record.get("keys") if isinstance(record.get("keys"), dict) else {}
                record["keys"] = keys
            normalized_payload = dict(key_payload or {})
            normalized_payload.setdefault("captured_at", _utc_now_iso())
            keys[key_name] = normalized_payload
            self._save_unlocked(payload)
            return normalized_payload
