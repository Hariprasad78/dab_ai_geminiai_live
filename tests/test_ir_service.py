from __future__ import annotations

import json

from vertex_live_dab_agent.ir.service import SamsungIrService


class _FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, payload):
        self.calls.append(dict(payload))
        if self._responses:
            return dict(self._responses.pop(0))
        return {"success": False, "error": "exhausted fake responses"}


def _write_dataset(path):
    path.write_text(
        json.dumps(
            {
                "devices": {
                    "samsung_tv_default": {
                        "brand": "Samsung",
                        "model": "Samsung TV",
                        "sender_channel": "D2",
                        "codes": {
                            "HOME": {"protocol": "SAMSUNG", "code": "0xE0E09E61"},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_send_key_retries_on_unknown_command(tmp_path):
    dataset_path = tmp_path / "ir.json"
    _write_dataset(dataset_path)

    service = SamsungIrService(dataset_path=dataset_path, serial_port="/dev/null")
    fake = _FakeTransport(
        [
            {"success": False, "error": "No response for legacy command (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No response for legacy command (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No JSON response from NodeMCU (last line: UKNW_CMD)", "transport": "serial"},
            {"success": True, "ok": True, "raw_line": "OK"},
        ]
    )
    service._transport = fake

    result = service.send_key("samsung_tv_default", "HOME")

    assert result["success"] is True
    assert len(fake.calls) >= 4
    assert str(fake.calls[0].get("legacy_cmd", "")).startswith("SENDP:")


def test_train_key_retries_on_unknown_command(tmp_path):
    dataset_path = tmp_path / "ir.json"
    _write_dataset(dataset_path)

    service = SamsungIrService(dataset_path=dataset_path, serial_port="/dev/null")
    fake = _FakeTransport(
        [
            {"success": False, "error": "No response for legacy command (last line: WN)", "transport": "serial"},
            {"success": False, "error": "UKNW_CMD", "transport": "serial"},
            {"success": True, "payload": {"protocol": "SAMSUNG", "code": "0xABCDEF"}},
        ]
    )
    service._transport = fake

    result = service.train_key("samsung_tv_default", "HOME")

    assert result["success"] is True
    assert len(fake.calls) >= 3
    assert fake.calls[0].get("legacy_cmd") == "CAPTURE?"
    assert result["stored_payload"]["code"] == "0xABCDEF"


def test_train_key_uses_legacy_capture_and_train_when_available(tmp_path):
    dataset_path = tmp_path / "ir.json"
    _write_dataset(dataset_path)

    service = SamsungIrService(dataset_path=dataset_path, serial_port="/dev/null")
    fake = _FakeTransport(
        [
            {"success": True, "raw_line": "IRCAPTURE:SAMSUNG,0xE0E09E61,32"},
            {"success": True, "raw_line": "IRTRAIN:samsung,HOME,OK"},
        ]
    )
    service._transport = fake

    result = service.train_key("samsung_tv_default", "HOME")

    assert result["success"] is True
    assert result["stored_payload"]["protocol"] == "SAMSUNG"
    assert result["stored_payload"]["code"] == "0xE0E09E61"
    assert result["stored_payload"]["bits"] == 32


def test_send_key_falls_back_to_legacy_sendp_on_wn_response(tmp_path):
    dataset_path = tmp_path / "ir.json"
    _write_dataset(dataset_path)

    service = SamsungIrService(dataset_path=dataset_path, serial_port="/dev/null")
    # Exhaust JSON-style attempts with WN-like unknown responses, then succeed on legacy attempt.
    fake = _FakeTransport(
        [
            {"success": False, "error": "No JSON response from NodeMCU (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No JSON response from NodeMCU (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No JSON response from NodeMCU (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No JSON response from NodeMCU (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No JSON response from NodeMCU (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No JSON response from NodeMCU (last line: WN)", "transport": "serial"},
            {"success": False, "error": "No JSON response from NodeMCU (last line: WN)", "transport": "serial"},
            {"success": True, "raw_line": "IRSENT:SAMSUNG,0xE0E09E61,32"},
        ]
    )
    service._transport = fake

    result = service.send_key("samsung_tv_default", "HOME")

    assert result["success"] is True
    assert any(str(call.get("legacy_cmd", "")).startswith("SENDP:") for call in fake.calls)


def test_send_key_falls_back_to_legacy_sendp_on_irrecv_noise(tmp_path):
    dataset_path = tmp_path / "ir.json"
    _write_dataset(dataset_path)

    service = SamsungIrService(dataset_path=dataset_path, serial_port="/dev/null")
    fake = _FakeTransport(
        [
            {
                "success": False,
                "error": "No JSON response from NodeMCU (last line: IRRECV:UNKNOWN,0x24AE7D4F,3)",
                "transport": "serial",
            },
            {"success": True, "raw_line": "IRSENT:SAMSUNG,0xE0E09E61,32"},
        ]
    )
    service._transport = fake

    result = service.send_key("samsung_tv_default", "HOME")

    assert result["success"] is True
    assert any(str(call.get("legacy_cmd", "")).startswith("SENDP:") for call in fake.calls)


def test_send_key_legacy_ok_ack_is_success(tmp_path):
    dataset_path = tmp_path / "ir.json"
    _write_dataset(dataset_path)

    service = SamsungIrService(dataset_path=dataset_path, serial_port="/dev/null")
    fake = _FakeTransport(
        [
            {"success": True, "raw_line": "OK"},
        ]
    )
    service._transport = fake

    result = service.send_key("samsung_tv_default", "HOME")

    assert result["success"] is True
    assert str(fake.calls[0].get("legacy_cmd", "")).startswith("SENDP:")
