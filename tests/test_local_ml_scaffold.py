"""Tests for local visual features and local ranker scaffold."""

import base64
from pathlib import Path

from scripts.train_local_ranker import train_model
from vertex_live_dab_agent.hybrid import LocalActionRanker, extract_local_visual_features


def _tiny_png_b64() -> str:
    return base64.b64encode(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
    ).decode("ascii")


def test_extract_local_visual_features_uses_ocr_and_png_metadata() -> None:
    features = extract_local_visual_features(
        image_b64=_tiny_png_b64(),
        ocr_text="Settings > Language and Time zone",
    )
    assert features["width"] == 1
    assert features["height"] == 1
    assert "settings" in features["screen_labels"]
    assert features["language_visible"] is True
    assert features["timezone_visible"] is True


def test_train_local_ranker_and_rank_actions(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectories.jsonl"
    trajectories.write_text(
        "\n".join(
            [
                '{"goal":"change timezone","result":"PASS","action":"SET_SETTING","observation_features":{"screen_labels":["timezone"]}}',
                '{"goal":"change language","result":"PASS","action":"SET_SETTING","observation_features":{"screen_labels":["language"]}}',
                '{"goal":"open youtube","result":"PASS","action":"LAUNCH_APP","observation_features":{"screen_labels":["unknown"]}}',
            ]
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.json"
    model = train_model(trajectories, model_path)
    assert model["samples"] == 3

    ranker = LocalActionRanker(model_path)
    ranked = ranker.rank(
        goal="change timezone to Colombo",
        current_app="settings",
        observation_features={"screen_labels": ["timezone"]},
        retrieved_experiences=[],
        top_k=2,
    )
    assert ranked
    assert ranked[0].action == "SET_SETTING"
