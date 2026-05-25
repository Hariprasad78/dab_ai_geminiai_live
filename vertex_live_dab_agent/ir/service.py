from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from vertex_live_dab_agent.ir.dataset import IrDatasetStore
from vertex_live_dab_agent.ir.transport import SerialIrTransport


_DAB_TO_SAMSUNG_KEY_MAP: Dict[str, str] = {
    "PRESS_UP": "UP",
    "PRESS_DOWN": "DOWN",
    "PRESS_LEFT": "LEFT",
    "PRESS_RIGHT": "RIGHT",
    "PRESS_OK": "ENTER",
    "PRESS_BACK": "RETURN",
    "PRESS_HOME": "HOME",
    "PRESS_MENU": "MENU",
    "PRESS_INPUT": "SOURCE",
    "PRESS_EXIT": "EXIT",
    "PRESS_INFO": "INFO",
    "PRESS_GUIDE": "GUIDE",
    "PRESS_POWER": "POWER",
    "PRESS_MUTE": "MUTE",
    "PRESS_VOLUME_UP": "VOL_UP",
    "PRESS_VOLUME_DOWN": "VOL_DOWN",
    "PRESS_CHANNEL_UP": "CH_UP",
    "PRESS_CHANNEL_DOWN": "CH_DOWN",
    "PRESS_PLAY_PAUSE": "PLAY_PAUSE",
    "PRESS_PLAY": "PLAY",
    "PRESS_PAUSE": "PAUSE",
    "PRESS_STOP": "STOP",
    "PRESS_REWIND": "REWIND",
    "PRESS_FAST_FORWARD": "FAST_FORWARD",
    "PRESS_0": "NUM_0",
    "PRESS_1": "NUM_1",
    "PRESS_2": "NUM_2",
    "PRESS_3": "NUM_3",
    "PRESS_4": "NUM_4",
    "PRESS_5": "NUM_5",
    "PRESS_6": "NUM_6",
    "PRESS_7": "NUM_7",
    "PRESS_8": "NUM_8",
    "PRESS_9": "NUM_9",
    "PRESS_RED": "RED",
    "PRESS_GREEN": "GREEN",
    "PRESS_YELLOW": "YELLOW",
    "PRESS_BLUE": "BLUE",
}

_SAMSUNG_NODEMCU_KEY_ALIASES: Dict[str, List[str]] = {
    "POWER": ["POWER", "KEY_POWER", "POWER_TOGGLE"],
    "MUTE": ["MUTE", "KEY_MUTE"],
    "VOL_UP": ["VOL_UP", "VOLUP", "VOLUME_UP", "VOLUMEUP", "KEY_VOLUP", "KEY_VOLUMEUP"],
    "VOL_DOWN": ["VOL_DOWN", "VOLDOWN", "VOLUME_DOWN", "VOLUMEDOWN", "KEY_VOLDOWN", "KEY_VOLUMEDOWN"],
    "CH_UP": ["CH_UP", "CHUP", "CHANNEL_UP", "CHANNELUP", "KEY_CHUP", "KEY_CHANNELUP"],
    "CH_DOWN": ["CH_DOWN", "CHDOWN", "CHANNEL_DOWN", "CHANNELDOWN", "KEY_CHDOWN", "KEY_CHANNELDOWN"],
    "ENTER": ["ENTER", "OK", "SELECT", "KEY_ENTER", "KEY_OK"],
    "RETURN": ["RETURN", "BACK", "KEY_RETURN", "KEY_BACK"],
    "SOURCE": ["SOURCE", "INPUT", "KEY_SOURCE", "KEY_INPUT"],
    "HOME": ["HOME", "SMART", "KEY_HOME"],
    "MENU": ["MENU", "KEY_MENU"],
    "UP": ["UP", "KEY_UP"],
    "DOWN": ["DOWN", "KEY_DOWN"],
    "LEFT": ["LEFT", "KEY_LEFT"],
    "RIGHT": ["RIGHT", "KEY_RIGHT"],
    "EXIT": ["EXIT", "KEY_EXIT"],
    "INFO": ["INFO", "KEY_INFO"],
    "GUIDE": ["GUIDE", "KEY_GUIDE"],
    "PLAY_PAUSE": ["PLAY_PAUSE", "PLAYPAUSE", "KEY_PLAYPAUSE"],
    "PLAY": ["PLAY", "KEY_PLAY"],
    "PAUSE": ["PAUSE", "KEY_PAUSE"],
    "STOP": ["STOP", "KEY_STOP"],
    "REWIND": ["REWIND", "REW", "KEY_REWIND"],
    "FAST_FORWARD": ["FAST_FORWARD", "FF", "FFWD", "KEY_FASTFORWARD"],
}


class SamsungIrService:
    """Samsung-only IR service adapter (dataset + serial endpoint)."""

    def __init__(
        self,
        dataset_path: Path,
        serial_port: str,
        baudrate: int = 115200,
        timeout_seconds: float = 3.0,
        sender_channel: str = "D2",
    ) -> None:
        self._dataset = IrDatasetStore(dataset_path)
        self._transport = SerialIrTransport(serial_port, baudrate=baudrate, timeout_seconds=timeout_seconds)
        self._sender_channel = str(sender_channel or "D2").strip() or "D2"
        self._strategy_lock = threading.Lock()
        self._preferred_send_strategy = "auto"
        self._preferred_train_strategy = "auto"

    @property
    def dataset_path(self) -> Path:
        return Path(self._dataset._path)

    @property
    def serial_port(self) -> str:
        return self._transport.port

    def normalize_key_name(self, key_name: str) -> str:
        raw = str(key_name or "").strip().upper()
        if not raw:
            return ""
        if raw in _DAB_TO_SAMSUNG_KEY_MAP:
            return _DAB_TO_SAMSUNG_KEY_MAP[raw]
        simplified = raw.replace(" ", "_").replace("-", "_")
        if simplified.startswith("SAMSUNG_"):
            simplified = simplified[len("SAMSUNG_"):]
        return simplified

    def list_devices(self) -> List[Dict[str, Any]]:
        return [
            {
                "device_id": row.device_id,
                "brand": row.brand,
                "model": row.model,
                "sender_channel": row.sender_channel,
                "key_count": row.key_count,
            }
            for row in self._dataset.list_devices()
        ]

    def list_keys(self, device_id: str) -> List[str]:
        return self._dataset.list_keys(device_id)

    def status(self) -> Dict[str, Any]:
        return {
            "available": self._transport.configured,
            "serial_port": self._transport.port,
            "dataset_path": str(self.dataset_path),
            "sender_channel": self._sender_channel,
            "brand": "Samsung",
            "devices": self.list_devices(),
        }

    def train_key(self, device_id: str, key_name: str, timeout_ms: int = 8000) -> Dict[str, Any]:
        normalized_key = self.normalize_key_name(key_name)
        if not normalized_key:
            return {"success": False, "error": "key_name is required"}

        # Prefer legacy firmware flow first: CAPTURE? + TRAIN:<device>,<key>
        # This matches the NodeMCU CLI protocol used in production.
        preferred_train = self._preferred_train_strategy

        if preferred_train == "legacy":
            quick_legacy = self._try_legacy_train(device_id, normalized_key)
            if bool(quick_legacy.get("success")):
                return quick_legacy

        legacy_capture = self._transport.request(
            {
                "legacy_cmd": "CAPTURE?",
                "expect_prefixes": ["IRCAPTURE:"],
                "timeout_seconds": 1.0,
            }
        )
        if bool(legacy_capture.get("success")):
            train_device = "samsung"
            legacy_train = self._transport.request(
                {
                    "legacy_cmd": f"TRAIN:{train_device},{normalized_key}",
                    "expect_prefixes": ["IRTRAIN:"],
                    "timeout_seconds": 1.0,
                }
            )
            if bool(legacy_train.get("success")):
                payload = self._extract_legacy_capture_payload(legacy_capture)
                if payload:
                    saved = self._dataset.upsert_key_payload(device_id, normalized_key, payload)
                    self._set_preferred_train_strategy("legacy")
                    return {
                        "success": True,
                        "device_id": device_id,
                        "key_name": normalized_key,
                        "trained": True,
                        "stored_payload": saved,
                        "raw": {
                            "capture": legacy_capture,
                            "train": legacy_train,
                            "protocol": "legacy",
                        },
                    }

        read_payload = {
            "cmd": "ir_read",
            "brand": "samsung",
            "sender": self._sender_channel,
            "key": normalized_key,
            "timeout_ms": max(1000, int(timeout_ms)),
            "timeout_seconds": 0.8,
        }
        response = self._request_with_unknown_cmd_fallback(
            read_payload,
            cmd_aliases=["read_ir", "irrecv", "learn_ir", "learn", "read"],
            extra_attempts=[
                {
                    "command": "read",
                    "brand": "samsung",
                    "sender": self._sender_channel,
                    "key": normalized_key,
                    "timeout_ms": max(1000, int(timeout_ms)),
                }
            ],
        )
        if not bool(response.get("success")):
            return {
                "success": False,
                "device_id": device_id,
                "key_name": normalized_key,
                "error": str(response.get("error") or "IR read failed"),
                "raw": response,
            }

        payload = self._extract_code_payload(response)
        if not payload:
            return {
                "success": False,
                "device_id": device_id,
                "key_name": normalized_key,
                "error": "NodeMCU response did not include an IR payload",
                "raw": response,
            }

        saved = self._dataset.upsert_key_payload(device_id, normalized_key, payload)
        self._set_preferred_train_strategy("json")
        return {
            "success": True,
            "device_id": device_id,
            "key_name": normalized_key,
            "trained": True,
            "stored_payload": saved,
            "raw": response,
        }

    def send_key(self, device_id: str, key_name: str) -> Dict[str, Any]:
        normalized_key = self.normalize_key_name(key_name)
        if not normalized_key:
            return {"success": False, "error": "key_name is required"}

        payload = self._dataset.get_key_payload(device_id, normalized_key)
        has_payload = isinstance(payload, dict)
        if not has_payload:
            payload = {}

        # When the key has not been trained, still attempt key-name sends first.
        # Many NodeMCU firmwares support SENDK for built-in Samsung key names.
        if not has_payload:
            sendk_result = self._try_legacy_sendk(device_id, normalized_key)
            if bool(sendk_result.get("success")):
                self._set_preferred_send_strategy("legacy_sendk")
                return sendk_result
            json_key_result = self._try_json_send_key_only(device_id, normalized_key)
            if bool(json_key_result.get("success")):
                self._set_preferred_send_strategy("json")
                return json_key_result
            return {
                "success": False,
                "device_id": device_id,
                "key_name": normalized_key,
                "error": "Key is not trained and direct key send failed",
                "raw": {
                    "legacy_sendk": sendk_result,
                    "json_key_only": json_key_result,
                },
            }

        # Prefer legacy SENDP/SENDK first: this is the native protocol for
        # the NodeMCU firmware used by the serial host CLI.
        preferred_send = self._preferred_send_strategy

        if preferred_send == "legacy_sendp":
            fast = self._try_legacy_sendp(device_id, normalized_key, payload)
            if bool(fast.get("success")):
                return fast
        elif preferred_send == "legacy_sendk":
            fast = self._try_legacy_sendk(device_id, normalized_key)
            if bool(fast.get("success")):
                return fast
        elif preferred_send == "json":
            fast = self._try_json_send(device_id, normalized_key, payload)
            if bool(fast.get("success")):
                return fast

        sendp_result = self._try_legacy_sendp(device_id, normalized_key, payload)
        if bool(sendp_result.get("success")):
            self._set_preferred_send_strategy("legacy_sendp")
            return sendp_result

        sendk_result = self._try_legacy_sendk(device_id, normalized_key)
        if bool(sendk_result.get("success")):
            self._set_preferred_send_strategy("legacy_sendk")
            return sendk_result

        json_result = self._try_json_send(device_id, normalized_key, payload)
        if bool(json_result.get("success")):
            self._set_preferred_send_strategy("json")
            return json_result

        return {
            "success": False,
            "device_id": device_id,
            "key_name": normalized_key,
            "error": str(json_result.get("error") or "IR send failed"),
            "raw": json_result,
        }

    def _try_json_send_key_only(self, device_id: str, normalized_key: str) -> Dict[str, Any]:
        response = self._request_with_unknown_cmd_fallback(
            {
                "cmd": "ir_send",
                "brand": "samsung",
                "sender": self._sender_channel,
                "key": normalized_key,
                "timeout_seconds": 0.2,
            },
            cmd_aliases=["send_ir", "irsend", "send"],
            extra_attempts=[
                {
                    "cmd": "send",
                    "brand": "samsung",
                    "sender": self._sender_channel,
                    "key": normalized_key,
                    "timeout_seconds": 0.2,
                },
                {
                    "command": "send",
                    "brand": "samsung",
                    "sender": self._sender_channel,
                    "key": normalized_key,
                    "timeout_seconds": 0.2,
                },
            ],
            deadline_seconds=0.45,
        )
        if not bool(response.get("success")):
            return response
        return {
            "success": True,
            "device_id": device_id,
            "key_name": normalized_key,
            "sender_channel": self._sender_channel,
            "raw": response,
        }

    def _try_legacy_train(self, device_id: str, normalized_key: str) -> Dict[str, Any]:
        capture = self._transport.request(
            {
                "legacy_cmd": "CAPTURE?",
                "expect_prefixes": ["IRCAPTURE:"],
                "timeout_seconds": 1.0,
            }
        )
        if not bool(capture.get("success")):
            return capture
        train = self._transport.request(
            {
                "legacy_cmd": f"TRAIN:samsung,{normalized_key}",
                "expect_prefixes": ["IRTRAIN:"],
                "timeout_seconds": 1.0,
            }
        )
        if not bool(train.get("success")):
            return train
        payload = self._extract_legacy_capture_payload(capture)
        if not payload:
            return {"success": False, "error": "IRCAPTURE payload missing from NodeMCU"}
        saved = self._dataset.upsert_key_payload(device_id, normalized_key, payload)
        return {
            "success": True,
            "device_id": device_id,
            "key_name": normalized_key,
            "trained": True,
            "stored_payload": saved,
            "raw": {"capture": capture, "train": train, "protocol": "legacy"},
        }

    def _try_legacy_sendp(self, device_id: str, normalized_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        sendp_cmd = self._build_sendp_legacy_cmd(payload)
        if not sendp_cmd:
            return {"success": False, "error": "Missing protocol/code for SENDP"}
        legacy_sendp = self._transport.request(
            {
                "legacy_cmd": sendp_cmd,
                "expect_prefixes": ["IRSENT:"],
                "timeout_seconds": 0.15,
            }
        )
        if not bool(legacy_sendp.get("success")):
            return legacy_sendp
        return {
            "success": True,
            "device_id": device_id,
            "key_name": normalized_key,
            "sender_channel": self._sender_channel,
            "raw": legacy_sendp,
        }

    def _try_legacy_sendk(self, device_id: str, normalized_key: str) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for alias in self._samsung_key_aliases(normalized_key):
            for command in (
                f"SENDK:samsung,{alias}",
                f"SENDK:SAMSUNG,{alias}",
                f"SENDK:samsung:{alias}",
                f"SENDK:SAMSUNG:{alias}",
                f"SENDK:{alias}",
            ):
                if command in seen:
                    continue
                seen.add(command)
                attempts.append(
                    {
                        "legacy_cmd": command,
                        "expect_prefixes": ["IRSENT:", "IRSEND:", "SENT:"],
                        "timeout_seconds": 0.2,
                    }
                )
                if len(attempts) >= 2:
                    break
            if len(attempts) >= 2:
                break

        legacy_sendk: Dict[str, Any] = {"success": False, "error": "No SENDK attempts were made"}
        for attempt in attempts:
            legacy_sendk = self._transport.request(attempt)
            if bool(legacy_sendk.get("success")):
                break
            if not self._is_unknown_command_error(legacy_sendk):
                break
        if not bool(legacy_sendk.get("success")):
            legacy_sendk = dict(legacy_sendk)
            legacy_sendk["attempted_commands"] = [str(item.get("legacy_cmd") or "") for item in attempts]
            return legacy_sendk
        return {
            "success": True,
            "device_id": device_id,
            "key_name": normalized_key,
            "sender_channel": self._sender_channel,
            "raw": legacy_sendk,
        }

    def _try_json_send(self, device_id: str, normalized_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        send_payload = {
            "cmd": "ir_send",
            "brand": "samsung",
            "sender": self._sender_channel,
            "key": normalized_key,
            "payload": payload,
            "timeout_seconds": 0.2,
        }
        flat_payload = {
            "cmd": "send",
            "brand": "samsung",
            "sender": self._sender_channel,
            "key": normalized_key,
            "protocol": payload.get("protocol"),
            "code": payload.get("code"),
            "bits": payload.get("bits"),
            "raw": payload.get("raw"),
            "value": payload.get("value"),
            "timeout_seconds": 0.2,
        }
        response = self._request_with_unknown_cmd_fallback(
            send_payload,
            cmd_aliases=["send_ir", "irsend", "send"],
            extra_attempts=[
                flat_payload,
                {
                    "command": "send",
                    "brand": "samsung",
                    "sender": self._sender_channel,
                    "key": normalized_key,
                    "protocol": payload.get("protocol"),
                    "code": payload.get("code"),
                    "bits": payload.get("bits"),
                    "raw": payload.get("raw"),
                    "value": payload.get("value"),
                    "timeout_seconds": 0.2,
                },
            ],
            deadline_seconds=0.45,
        )
        if not bool(response.get("success")):
            return response
        return {
            "success": True,
            "device_id": device_id,
            "key_name": normalized_key,
            "sender_channel": self._sender_channel,
            "raw": response,
        }

    def _set_preferred_send_strategy(self, strategy: str) -> None:
        with self._strategy_lock:
            self._preferred_send_strategy = strategy

    def _set_preferred_train_strategy(self, strategy: str) -> None:
        with self._strategy_lock:
            self._preferred_train_strategy = strategy

    def send_dab_style_action(self, device_id: str, action: str) -> Dict[str, Any]:
        return self.send_key(device_id=device_id, key_name=self.normalize_key_name(action))

    @staticmethod
    def _samsung_key_aliases(normalized_key: str) -> List[str]:
        key = str(normalized_key or "").strip().upper()
        aliases = list(_SAMSUNG_NODEMCU_KEY_ALIASES.get(key) or [key])
        if key.startswith("NUM_"):
            digit = key.removeprefix("NUM_")
            aliases.extend([digit, f"KEY_{digit}", f"NUM{digit}"])
        if key and key not in aliases:
            aliases.insert(0, key)
        deduped: List[str] = []
        seen: set[str] = set()
        for alias in aliases:
            clean = str(alias or "").strip().upper()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped

    def _request_with_unknown_cmd_fallback(
        self,
        payload: Dict[str, Any],
        cmd_aliases: List[str],
        extra_attempts: Optional[List[Dict[str, Any]]] = None,
        deadline_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = [dict(payload)]
        base_cmd = str(payload.get("cmd") or "").strip().lower()
        for alias in cmd_aliases:
            candidate = str(alias or "").strip()
            if not candidate:
                continue
            if candidate.lower() == base_cmd:
                continue
            patched = dict(payload)
            patched["cmd"] = candidate
            attempts.append(patched)
        for extra in extra_attempts or []:
            if isinstance(extra, dict):
                attempts.append(dict(extra))

        last_response: Dict[str, Any] = {"success": False, "error": "IR request failed"}
        started = time.monotonic()
        for idx, candidate_payload in enumerate(attempts):
            if deadline_seconds is not None and idx > 0 and (time.monotonic() - started) >= float(deadline_seconds):
                break
            response = self._transport.request(candidate_payload)
            if bool(response.get("success")):
                return response
            last_response = response
            if idx >= len(attempts) - 1:
                break
            if not self._is_unknown_command_error(response):
                break
        return last_response

    @staticmethod
    def _is_unknown_command_error(response: Dict[str, Any]) -> bool:
        fragments = [
            str(response.get("error") or ""),
            str(response.get("raw_line") or ""),
        ]
        text = " ".join(fragments).strip().lower()
        if not text:
            return False
        return any(
            token in text
            for token in (
                "uknw_cmd",
                "unknown_cmd",
                "unknown command",
                "unknown cmd",
                "last line: wn",
                "no json response from nodemcu",
                "no response for legacy command",
                "no response from nodemcu over serial",
                "returned no data",
                "multiple access on port",
                "last line: irrecv:",
            )
        ) or text == "wn"

    @staticmethod
    def _build_sendp_legacy_cmd(payload: Dict[str, Any]) -> Optional[str]:
        protocol = str(payload.get("protocol") or "").strip().upper()
        code = str(payload.get("code") or "").strip()
        bits = payload.get("bits")
        if not protocol or not code:
            return None
        if bits is None or str(bits).strip() == "":
            return f"SENDP:{protocol},{code}"
        try:
            bits_int = int(bits)
            if bits_int > 0:
                return f"SENDP:{protocol},{code},{bits_int}"
        except Exception:
            pass
        return f"SENDP:{protocol},{code}"

    @staticmethod
    def _extract_legacy_capture_payload(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        line = str(response.get("raw_line") or "").strip()
        if not line.upper().startswith("IRCAPTURE:"):
            return None
        body = line.split(":", 1)[1].strip()
        parts = [part.strip() for part in body.split(",") if str(part).strip()]
        if len(parts) < 2:
            return None
        payload: Dict[str, Any] = {
            "protocol": parts[0].upper(),
            "code": parts[1],
            "source": "nodemcu",
        }
        if len(parts) >= 3:
            try:
                payload["bits"] = int(parts[2])
            except Exception:
                payload["bits"] = parts[2]
        return payload

    @staticmethod
    def _extract_code_payload(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        candidate_keys = ("payload", "code", "raw", "data", "value")
        for key in candidate_keys:
            value = response.get(key)
            if isinstance(value, dict):
                cloned = dict(value)
                cloned.setdefault("source", "nodemcu")
                return cloned
            if isinstance(value, (str, int, float)) and str(value).strip():
                return {"code": str(value).strip(), "source": "nodemcu"}
        return None
