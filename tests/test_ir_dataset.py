from __future__ import annotations

import json

from vertex_live_dab_agent.ir.dataset import IrDatasetStore


def test_get_key_payload_falls_back_from_default_samsung_device(tmp_path):
    dataset_path = tmp_path / "ir_samsung_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "devices": {
                    "samsung_tv_default": {
                        "brand": "Samsung",
                        "model": "Samsung TV (Default)",
                        "sender_channel": "D2",
                        "keys": {},
                    },
                    "samsung": {
                        "brand": "Samsung",
                        "model": "OLED",
                        "codes": {
                            "dpaddown": {"protocol": "SAMSUNG", "code": "0xE0E08679"},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    store = IrDatasetStore(dataset_path)
    payload = store.get_key_payload("samsung_tv_default", "DOWN")

    assert isinstance(payload, dict)
    assert payload["code"] == "0xE0E08679"


def test_list_keys_falls_back_from_default_samsung_device(tmp_path):
    dataset_path = tmp_path / "ir_samsung_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "devices": {
                    "samsung_tv_default": {
                        "brand": "Samsung",
                        "model": "Samsung TV (Default)",
                        "sender_channel": "D2",
                        "keys": {},
                    },
                    "samsung": {
                        "brand": "Samsung",
                        "model": "OLED",
                        "codes": {
                            "home": {"protocol": "SAMSUNG", "code": "0xE0E09E61"},
                            "dpaddown": {"protocol": "SAMSUNG", "code": "0xE0E08679"},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    store = IrDatasetStore(dataset_path)
    keys = store.list_keys("samsung_tv_default")

    assert "dpaddown" in keys
    assert "home" in keys


def test_get_key_payload_matches_return_with_back_alias(tmp_path):
    dataset_path = tmp_path / "ir_samsung_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "devices": {
                    "samsung_tv_default": {
                        "brand": "Samsung",
                        "model": "Samsung TV (Default)",
                        "sender_channel": "D2",
                        "keys": {},
                    },
                    "samsung": {
                        "brand": "Samsung",
                        "model": "OLED",
                        "codes": {
                            "back": {"protocol": "SAMSUNG", "code": "0xE0E01AE5"},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    store = IrDatasetStore(dataset_path)
    payload = store.get_key_payload("samsung_tv_default", "RETURN")

    assert isinstance(payload, dict)
    assert payload["code"] == "0xE0E01AE5"
