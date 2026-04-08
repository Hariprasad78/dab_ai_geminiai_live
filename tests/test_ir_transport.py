from __future__ import annotations

import types

from vertex_live_dab_agent.ir.transport import SerialIrTransport


class _FakeConnection:
    def __init__(self, lines):
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def reset_input_buffer(self):
        return None

    def write(self, data):
        return len(data)

    def flush(self):
        return None

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return b""


def test_request_retries_after_transient_serial_no_data(monkeypatch):
    attempts = {"count": 0}

    class FakeSerialException(Exception):
        pass

    def fake_serial_ctor(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FakeSerialException(
                "device reports readiness to read but returned no data (device disconnected or multiple access on port?)"
            )
        return _FakeConnection([b'{"ok": true}\n'])

    fake_serial_module = types.SimpleNamespace(
        Serial=fake_serial_ctor,
        SerialException=FakeSerialException,
    )
    monkeypatch.setitem(__import__("sys").modules, "serial", fake_serial_module)

    transport = SerialIrTransport("/dev/ttyUSB1", timeout_seconds=0.5)
    result = transport.request({"cmd": "ping"})

    assert attempts["count"] == 2
    assert result["success"] is True


def test_request_surfaces_port_contention_hint(monkeypatch):
    class FakeSerialException(Exception):
        pass

    def fake_serial_ctor(*args, **kwargs):
        raise FakeSerialException(
            "device reports readiness to read but returned no data (device disconnected or multiple access on port?)"
        )

    fake_serial_module = types.SimpleNamespace(
        Serial=fake_serial_ctor,
        SerialException=FakeSerialException,
    )
    monkeypatch.setitem(__import__("sys").modules, "serial", fake_serial_module)

    transport = SerialIrTransport("/dev/ttyUSB1", timeout_seconds=0.5)
    result = transport.request({"cmd": "ping"})

    assert result["success"] is False
    assert "close `pio device monitor`" in str(result.get("error", "")).lower()


def test_request_parses_irrecv_as_success_for_read_commands(monkeypatch):
    class FakeSerialException(Exception):
        pass

    def fake_serial_ctor(*args, **kwargs):
        return _FakeConnection([b"IRRECV:SAMSUNG,0xE0E01AE5,32\n"])

    fake_serial_module = types.SimpleNamespace(
        Serial=fake_serial_ctor,
        SerialException=FakeSerialException,
    )
    monkeypatch.setitem(__import__("sys").modules, "serial", fake_serial_module)

    transport = SerialIrTransport("/dev/ttyUSB1", timeout_seconds=0.5)
    result = transport.request({"cmd": "ir_read"})

    assert result["success"] is True
    payload = result.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("protocol") == "SAMSUNG"
    assert payload.get("code") == "0xE0E01AE5"
    assert payload.get("bits") == 32


def test_request_ignores_unknown_irrecv_noise_for_read_commands(monkeypatch):
    class FakeSerialException(Exception):
        pass

    def fake_serial_ctor(*args, **kwargs):
        return _FakeConnection(
            [
                b"IRRECV:UNKNOWN,0x22AE7A29,3\n",
                b"IRRECV:SAMSUNG,0xE0E01AE5,32\n",
            ]
        )

    fake_serial_module = types.SimpleNamespace(
        Serial=fake_serial_ctor,
        SerialException=FakeSerialException,
    )
    monkeypatch.setitem(__import__("sys").modules, "serial", fake_serial_module)

    transport = SerialIrTransport("/dev/ttyUSB1", timeout_seconds=0.5)
    result = transport.request({"cmd": "ir_read"})

    assert result["success"] is True
    payload = result.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("protocol") == "SAMSUNG"


def test_legacy_capture_accepts_valid_irrecv_line(monkeypatch):
    class FakeSerialException(Exception):
        pass

    def fake_serial_ctor(*args, **kwargs):
        return _FakeConnection(
            [
                b"IRRECV:UNKNOWN,0x22AE7A29,3\n",
                b"IRRECV:SAMSUNG,0xE0E01AE5,32\n",
            ]
        )

    fake_serial_module = types.SimpleNamespace(
        Serial=fake_serial_ctor,
        SerialException=FakeSerialException,
    )
    monkeypatch.setitem(__import__("sys").modules, "serial", fake_serial_module)

    transport = SerialIrTransport("/dev/ttyUSB1", timeout_seconds=0.5)
    result = transport.request({"legacy_cmd": "CAPTURE?", "expect_prefixes": ["IRCAPTURE:"]})

    assert result["success"] is True
    payload = result.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("protocol") == "SAMSUNG"
