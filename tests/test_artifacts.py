"""Tests for ArtifactStore."""
import json

import pytest

import vertex_live_dab_agent.config as cfg_mod
from vertex_live_dab_agent.artifacts.logger import ArtifactStore, setup_logging


@pytest.fixture(autouse=True)
def tmp_artifacts_dir(tmp_path, monkeypatch):
    """Override ARTIFACTS_BASE_DIR to a temp directory for every test."""
    monkeypatch.setenv("ARTIFACTS_BASE_DIR", str(tmp_path))
    cfg_mod.reset_config()
    yield tmp_path
    cfg_mod.reset_config()


def test_artifact_store_creates_directories(tmp_artifacts_dir):
    store = ArtifactStore("run-abc")
    assert store.run_dir.exists()
    assert (store.run_dir / "screenshots").exists()
    assert (store.run_dir / "planner_traces").exists()


def test_save_metadata(tmp_artifacts_dir):
    store = ArtifactStore("run-meta")
    store.save_metadata({"run_id": "run-meta", "goal": "test goal"})
    path = store.run_dir / "metadata.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["run_id"] == "run-meta"
    assert data["goal"] == "test goal"


def test_save_action(tmp_artifacts_dir):
    store = ArtifactStore("run-action")
    store.save_action({"step": 0, "action": "PRESS_OK", "result": "PASS"})
    store.save_action({"step": 1, "action": "PRESS_DOWN", "result": "PASS"})
    path = store.run_dir / "actions.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["action"] == "PRESS_OK"


def test_save_screenshot(tmp_artifacts_dir):
    # 1×1 white PNG (valid base64)
    white_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
    )
    store = ArtifactStore("run-ss")
    path = store.save_screenshot(white_png_b64, step=3)
    assert path is not None
    assert (store.run_dir / "screenshots" / "step_0003.png").exists()


def test_save_screenshot_invalid_b64(tmp_artifacts_dir):
    store = ArtifactStore("run-ss-bad")
    result = store.save_screenshot("not-valid-base64!!!", step=0)
    assert result is None  # Should return None, not raise


def test_save_planner_trace(tmp_artifacts_dir):
    store = ArtifactStore("run-trace")
    store.save_planner_trace({"step": 2, "planned_action": "PRESS_UP"}, step=2)
    path = store.run_dir / "planner_traces" / "step_0002.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["planned_action"] == "PRESS_UP"


def test_save_final_summary(tmp_artifacts_dir):
    store = ArtifactStore("run-final")
    store.save_final_summary({"status": "DONE", "step_count": 5})
    path = store.run_dir / "final_summary.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["status"] == "DONE"


def test_setup_logging_does_not_raise():
    """setup_logging should configure logging without raising."""
    setup_logging("DEBUG")
    setup_logging("INFO")
    setup_logging("WARNING")
    setup_logging("invalid-level")  # Should not raise, falls back to INFO
