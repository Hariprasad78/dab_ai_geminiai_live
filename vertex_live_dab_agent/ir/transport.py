from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict


class SerialIrTransport:
    """NodeMCU serial transport for IR read/send commands."""

    _PORT_LOCKS: Dict[str, threading.Lock] = {}
    _PORT_LOCKS_GUARD = threading.Lock()

    def __init__(self, port: str, baudrate: int = 115200, timeout_seconds: float = 3.0) -> None:
        self._port = str(port or "").strip()
        self._baudrate = int(baudrate)
        self._timeout_seconds = float(timeout_seconds)
        self._serial_conn: Any = None
        self._last_used_monotonic: float = 0.0
        self._idle_close_seconds: float = 30.0

    @property
    def configured(self) -> bool:
        return bool(self._port)

    @property
    def port(self) -> str:
        return self._port

    def _port_lock(self) -> threading.Lock:
        with self._PORT_LOCKS_GUARD:
            lock = self._PORT_LOCKS.get(self._port)
            if lock is None:
                lock = threading.Lock()
                self._PORT_LOCKS[self._port] = lock
            return lock

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._port:
            return {"success": False, "error": "IR serial port is not configured"}

        try:
            import serial  # type: ignore
        except Exception:
            return {"success": False, "error": "pyserial is not installed"}

        serial_exception_type = getattr(serial, "SerialException", Exception)
        with self._port_lock():
            self._close_if_idle_unlocked()
            for attempt in range(2):
                timeout_seconds = self._resolve_timeout_seconds(payload)
                try:
                    if attempt > 0:
                        self._close_serial_unlocked()
                    ser = self._ensure_serial_unlocked(serial)
                    start = time.monotonic()
                    cmd_name = str(payload.get("cmd") or payload.get("command") or "").strip().lower()
                    read_like = any(token in cmd_name for token in ("read", "learn", "recv", "capture"))
                    legacy_cmd = str(payload.get("legacy_cmd") or "").strip()
                    if legacy_cmd:
                        expect_raw = payload.get("expect_prefixes")
                        expect_prefixes = [str(expect_raw)] if isinstance(expect_raw, str) else []
                        if isinstance(expect_raw, (list, tuple)):
                            expect_prefixes = [str(item) for item in expect_raw if str(item or "").strip()]
                        result = self._request_legacy_line(ser, legacy_cmd, expect_prefixes, start, timeout_seconds)
                        self._last_used_monotonic = time.monotonic()
                        return result
                    if hasattr(ser, "reset_input_buffer"):
                        try:
                            ser.reset_input_buffer()
                        except Exception:
                            pass
                    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                    ser.write(raw)
                    ser.flush()

                    deadline = start + max(0.2, timeout_seconds)
                    last_text = ""
                    while time.monotonic() < deadline:
                        line = ser.readline()
                        if not line:
                            continue
                        text = line.decode(errors="replace").strip()
                        if not text:
                            continue
                        last_text = text
                        if read_like and text.upper().startswith("IRRECV:"):
                            parsed = self._parse_irrecv_payload(
                                text,
                                allow_unknown=bool(payload.get("accept_unknown_irrecv", False)),
                            )
                            if parsed:
                                self._last_used_monotonic = time.monotonic()
                                return {
                                    "success": True,
                                    "payload": parsed,
                                    "raw_line": text,
                                    "transport": "serial",
                                }
                        if text.startswith("{") and text.endswith("}"):
                            try:
                                parsed = json.loads(text)
                                if isinstance(parsed, dict):
                                    parsed.setdefault("success", bool(parsed.get("ok", True)))
                                    parsed.setdefault("raw_line", text)
                                    parsed.setdefault("transport", "serial")
                                    self._last_used_monotonic = time.monotonic()
                                    return parsed
                            except Exception:
                                pass
                        if text.upper().startswith("OK"):
                            self._last_used_monotonic = time.monotonic()
                            return {"success": True, "raw_line": text, "transport": "serial"}
                        if text.upper().startswith("ERR"):
                            self._last_used_monotonic = time.monotonic()
                            return {"success": False, "error": text, "raw_line": text, "transport": "serial"}
                    self._last_used_monotonic = time.monotonic()
                    if last_text:
                        return {
                            "success": False,
                            "error": f"No JSON response from NodeMCU (last line: {last_text})",
                            "transport": "serial",
                        }
                    return {"success": False, "error": "No response from NodeMCU over serial", "transport": "serial"}
                except serial_exception_type as exc:
                    self._close_serial_unlocked()
                    message = str(exc)
                    lowered = message.lower()
                    transient = "returned no data" in lowered or "multiple access on port" in lowered
                    if transient and attempt == 0:
                        time.sleep(0.2)
                        continue
                    if transient:
                        message = (
                            f"{message}. Another process may be using {self._port} "
                            f"(close `pio device monitor` or any serial terminal and retry)."
                        )
                    return {"success": False, "error": f"Serial request failed: {message}"}
                except Exception as exc:
                    self._close_serial_unlocked()
                    return {"success": False, "error": f"Serial request failed: {exc}"}
        return {"success": False, "error": "Serial request failed: unknown error"}

    def _ensure_serial_unlocked(self, serial_module: Any) -> Any:
        conn = self._serial_conn
        if conn is not None:
            is_open = bool(getattr(conn, "is_open", True))
            if is_open:
                return conn
            self._serial_conn = None
        conn = serial_module.Serial(self._port, self._baudrate, timeout=0.05, write_timeout=0.2)
        # Best effort to avoid auto-reset side effects on some USB-serial boards.
        for attr, value in (("dtr", False), ("rts", False)):
            try:
                setattr(conn, attr, value)
            except Exception:
                pass
        self._serial_conn = conn
        self._last_used_monotonic = time.monotonic()
        return conn

    def _close_serial_unlocked(self) -> None:
        conn = self._serial_conn
        self._serial_conn = None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass

    def _close_if_idle_unlocked(self) -> None:
        if self._serial_conn is None:
            return
        if self._last_used_monotonic <= 0:
            return
        if (time.monotonic() - self._last_used_monotonic) >= self._idle_close_seconds:
            self._close_serial_unlocked()

    def _request_legacy_line(
        self,
        ser: Any,
        command: str,
        expect_prefixes: list[str],
        start: float,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        if hasattr(ser, "reset_input_buffer"):
            try:
                ser.reset_input_buffer()
            except Exception:
                pass
        raw = (command.rstrip("\n") + "\n").encode("utf-8")
        ser.write(raw)
        ser.flush()
        deadline = start + max(0.2, timeout_seconds)
        last_text = ""
        prefixes = [item for item in expect_prefixes if item]
        while time.monotonic() < deadline:
            line = ser.readline()
            if not line:
                continue
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            last_text = text
            upper = text.upper()
            if upper.startswith("OK"):
                return {"success": True, "raw_line": text, "transport": "serial"}
            if upper.startswith("IRSEND"):
                return {"success": True, "raw_line": text, "transport": "serial"}
            if command.upper().startswith("CAPTURE") and upper.startswith("IRRECV:"):
                parsed = self._parse_irrecv_payload(text, allow_unknown=False)
                if parsed:
                    return {
                        "success": True,
                        "payload": parsed,
                        "raw_line": text,
                        "transport": "serial",
                    }
            if any(text.startswith(prefix) for prefix in prefixes):
                return {"success": True, "raw_line": text, "transport": "serial"}
            if upper.startswith("ERR"):
                return {"success": False, "error": text, "raw_line": text, "transport": "serial"}
        if last_text:
            return {"success": False, "error": f"No response for legacy command (last line: {last_text})", "transport": "serial"}
        return {"success": False, "error": "No response from NodeMCU over serial", "transport": "serial"}

    def _resolve_timeout_seconds(self, payload: Dict[str, Any]) -> float:
        raw = payload.get("timeout_seconds")
        if raw is None:
            raw = payload.get("timeout")
        if raw is None:
            return float(self._timeout_seconds)
        try:
            value = float(raw)
            return max(0.2, value)
        except Exception:
            return float(self._timeout_seconds)

    @staticmethod
    def _parse_irrecv_payload(line: str, allow_unknown: bool = False) -> Dict[str, Any] | None:
        text = str(line or "").strip()
        if not text.upper().startswith("IRRECV:"):
            return None
        body = text.split(":", 1)[1].strip()
        if not body:
            return None
        parts = [part.strip() for part in body.split(",")]
        if len(parts) < 2:
            return None
        protocol = parts[0].upper()
        if protocol == "UNKNOWN" and not allow_unknown:
            return None
        payload: Dict[str, Any] = {
            "protocol": protocol,
            "code": parts[1],
            "source": "nodemcu",
        }
        if len(parts) >= 3:
            try:
                payload["bits"] = int(parts[2])
            except Exception:
                payload["bits"] = parts[2]
        bits_value = payload.get("bits")
        if isinstance(bits_value, int) and bits_value > 0 and bits_value < 8 and not allow_unknown:
            return None
        return payload
