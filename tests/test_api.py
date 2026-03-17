"""Tests for the FastAPI backend."""
import pytest
from httpx import ASGITransport, AsyncClient

import vertex_live_dab_agent.config as cfg_mod
import vertex_live_dab_agent.api.api as api_mod
from vertex_live_dab_agent.api.api import app, _runs, _run_tasks, _dab_client, _planner, _screen_capture
from vertex_live_dab_agent.orchestrator.run_state import RunState, RunStatus


@pytest.fixture(autouse=True)
def reset_api_state(tmp_path, monkeypatch):
    """Clear shared API state and point artifacts to a temp dir before each test."""
    _runs.clear()
    _run_tasks.clear()
    api_mod._dab_client = None
    api_mod._planner = None
    api_mod._screen_capture = None
    api_mod._tts_service = None
    monkeypatch.setenv("ARTIFACTS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("DAB_MOCK_MODE", "true")
    monkeypatch.setenv("IMAGE_SOURCE", "auto")
    monkeypatch.setenv("HDMI_AUDIO_ENABLED", "false")
    monkeypatch.setenv("ENABLE_VERTEX_PLANNER", "false")
    cfg_mod.reset_config()
    yield
    api_mod._dab_client = None
    api_mod._planner = None
    api_mod._screen_capture = None
    api_mod._tts_service = None
    cfg_mod.reset_config()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "mock_mode" in data
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_config_summary(client):
    resp = await client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "dab_mock_mode" in data
    assert "image_source" in data
    assert "youtube_app_id" in data
    assert "vertex_planner_model" in data
    assert "vertex_live_model" in data
    assert "enable_livekit_agent" in data
    assert "max_steps_per_run" in data
    assert "log_level" in data
    assert "tts_enabled" in data
    assert "tts_voice_provider" in data
    assert "tts_model" in data
    assert "tts_voice_name" in data
    assert "tts_language_code" in data


@pytest.mark.asyncio
async def test_capture_source_status(client):
    resp = await client.get("/capture/source")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured_source" in data
    assert "hdmi_available" in data
    assert "hdmi_configured" in data


@pytest.mark.asyncio
async def test_audio_source_status(client):
    resp = await client.get("/audio/source")
    assert resp.status_code == 200
    data = resp.json()
    assert "enabled" in data
    assert "ffmpeg_available" in data
    assert "devices" in data


@pytest.mark.asyncio
async def test_stream_audio_disabled_by_default(client):
    resp = await client.get("/stream/audio")
    assert resp.status_code == 400


def test_get_planner_auto_uses_vertex_when_project_available(monkeypatch):
    class FakeVertexPlannerClient:
        def __init__(self, *, project, location, model):
            self.project = project
            self.location = location
            self.model = model

    monkeypatch.delenv("ENABLE_VERTEX_PLANNER", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
    monkeypatch.setenv("VERTEX_PLANNER_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("VERTEX_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    cfg_mod.reset_config()
    api_mod._planner = None
    monkeypatch.setattr(api_mod, "VertexPlannerClient", FakeVertexPlannerClient)

    planner = api_mod.get_planner()
    assert planner._vertex_client is not None
    assert planner._vertex_client.project == "demo-project"
    assert planner._vertex_client.model == "gemini-2.5-flash"


def test_get_planner_respects_explicit_disable(monkeypatch):
    monkeypatch.setenv("ENABLE_VERTEX_PLANNER", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
    monkeypatch.setenv("VERTEX_PLANNER_MODEL", "gemini-2.5-flash")
    cfg_mod.reset_config()
    api_mod._planner = None

    planner = api_mod.get_planner()
    assert planner._vertex_client is None


def test_get_planner_defaults_to_non_live_model(monkeypatch):
    class FakeVertexPlannerClient:
        def __init__(self, *, project, location, model):
            self.project = project
            self.location = location
            self.model = model

    monkeypatch.delenv("ENABLE_VERTEX_PLANNER", raising=False)
    monkeypatch.delenv("VERTEX_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("VERTEX_MODEL", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
    monkeypatch.setenv("VERTEX_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    cfg_mod.reset_config()
    api_mod._planner = None
    monkeypatch.setattr(api_mod, "VertexPlannerClient", FakeVertexPlannerClient)

    planner = api_mod.get_planner()
    assert planner._vertex_client is not None
    assert planner._vertex_client.model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_list_runs_empty(client):
    resp = await client.get("/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_run_not_found(client):
    resp = await client.get("/run/nonexistent-id/status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_screenshot_endpoint(client):
    resp = await client.post("/screenshot")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "image_b64" in data


@pytest.mark.asyncio
async def test_manual_action_key_press(client):
    resp = await client.post("/action", json={"action": "PRESS_OK"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["action"] == "PRESS_OK"


@pytest.mark.asyncio
async def test_manual_action_launch_app(client):
    resp = await client.post(
        "/action",
        json={"action": "LAUNCH_APP", "params": {"app_id": "youtube"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_manual_action_launch_app_with_content(client):
    resp = await client.post(
        "/action",
        json={"action": "LAUNCH_APP", "params": {"app_id": "youtube", "content": "lofi"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["result"].get("content") == "lofi"


@pytest.mark.asyncio
async def test_manual_action_launch_app_allows_non_youtube(client):
    resp = await client.post(
        "/action",
        json={"action": "LAUNCH_APP", "params": {"app_id": "com.netflix.ninja"}},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_manual_action_launch_app_missing_app_id(client):
    resp = await client.post("/action", json={"action": "LAUNCH_APP"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_manual_action_unknown(client):
    resp = await client.post("/action", json={"action": "UNKNOWN_THING"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_manual_action_full_operations(client):
    for action in ("OPERATIONS_LIST", "APPLICATIONS_LIST", "KEY_LIST"):
        resp = await client.post("/action", json={"action": action})
        assert resp.status_code == 200
        assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_manual_action_long_key_press(client):
    resp = await client.post(
        "/action",
        json={"action": "LONG_KEY_PRESS", "params": {"key_action": "PRESS_OK", "duration_ms": 1200}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["result"]["durationMs"] == 1200


@pytest.mark.asyncio
async def test_manual_action_wait(client):
    resp = await client.post(
        "/action",
        json={"action": "WAIT", "params": {"seconds": 0.01}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["result"]["seconds"] >= 0


@pytest.mark.asyncio
async def test_manual_action_exit_app(client):
    resp = await client.post(
        "/action",
        json={"action": "EXIT_APP", "params": {"app_id": "youtube"}},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_dab_discovery_endpoints(client):
    for path in ("/dab/operations", "/dab/keys", "/dab/apps"):
        resp = await client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "result" in data


@pytest.mark.asyncio
async def test_task_macro_plan(client):
    resp = await client.post(
        "/task/macro",
        json={"instruction": "open netflix, down 2, ok"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["planned_count"] >= 3
    assert data["planned_actions"][0]["action"] == "LAUNCH_APP"


@pytest.mark.asyncio
async def test_task_macro_execute(client):
    resp = await client.post(
        "/task/macro",
        json={"instruction": "home, wait 0.01, ok", "execute": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution"] is not None
    assert data["execution"]["total"] >= 2


@pytest.mark.asyncio
async def test_manual_actions_batch(client):
    resp = await client.post(
        "/actions/batch",
        json={
            "actions": [
                {"action": "PRESS_HOME"},
                {"action": "LAUNCH_APP", "params": {"app_id": "com.netflix.ninja"}},
            ],
            "continue_on_error": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2


@pytest.mark.asyncio
async def test_planner_debug(client):
    resp = await client.post(
        "/planner/debug",
        json={"goal": "Launch Netflix"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "action" in data
    assert "confidence" in data
    assert "reason" in data


@pytest.mark.asyncio
async def test_start_run(client):
    resp = await client.post("/run/start", json={"goal": "Open settings"})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["goal"] == "Open settings"
    assert data["status"] in ("PENDING", "RUNNING")


@pytest.mark.asyncio
async def test_start_run_with_max_steps(client):
    resp = await client.post("/run/start", json={"goal": "Navigate menu", "max_steps": 5})
    assert resp.status_code == 200
    assert "run_id" in resp.json()


@pytest.mark.asyncio
async def test_start_run_allows_non_youtube_app_id(client):
    resp = await client.post(
        "/run/start",
        json={"goal": "Open app", "app_id": "com.netflix.ninja"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_start_run_infers_launch_content_from_goal(client):
    resp = await client.post("/run/start", json={"goal": "open lofi songs"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    state = api_mod._runs[run_id]
    assert state.launch_content == "lofi songs"


@pytest.mark.asyncio
async def test_start_run_prefers_explicit_content_over_inferred(client):
    resp = await client.post(
        "/run/start",
        json={"goal": "open netflix", "content": "my explicit query"},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    state = api_mod._runs[run_id]
    assert state.launch_content == "my explicit query"


@pytest.mark.asyncio
async def test_get_run_status_after_start(client):
    start_resp = await client.post("/run/start", json={"goal": "Test goal"})
    run_id = start_resp.json()["run_id"]

    status_resp = await client.get(f"/run/{run_id}/status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["run_id"] == run_id
    assert data["goal"] == "Test goal"
    assert "has_screenshot" in data
    assert "artifacts_dir" in data
    assert "dab_log_count" in data
    assert "dab_logs_tail" in data
    assert "ai_log_count" in data
    assert "ai_logs_tail" in data


@pytest.mark.asyncio
async def test_get_run_history(client):
    start_resp = await client.post("/run/start", json={"goal": "History test"})
    run_id = start_resp.json()["run_id"]
    hist_resp = await client.get(f"/run/{run_id}/history")
    assert hist_resp.status_code == 200
    data = hist_resp.json()
    assert data["run_id"] == run_id
    assert "action_count" in data
    assert "actions" in data
    assert isinstance(data["actions"], list)


@pytest.mark.asyncio
async def test_get_run_dab_transcript(client):
    start_resp = await client.post("/run/start", json={"goal": "Open settings"})
    run_id = start_resp.json()["run_id"]
    resp = await client.get(f"/run/{run_id}/dab-transcript")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert "count" in data
    assert "events" in data
    assert isinstance(data["events"], list)


@pytest.mark.asyncio
async def test_get_run_ai_transcript(client):
    start_resp = await client.post("/run/start", json={"goal": "Open settings"})
    run_id = start_resp.json()["run_id"]
    resp = await client.get(f"/run/{run_id}/ai-transcript")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert "count" in data
    assert "events" in data
    assert isinstance(data["events"], list)


@pytest.mark.asyncio
async def test_get_run_narration(client):
    start_resp = await client.post("/run/start", json={"goal": "Open settings"})
    run_id = start_resp.json()["run_id"]
    resp = await client.get(f"/run/{run_id}/narration")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert "events" in data


@pytest.mark.asyncio
async def test_tts_speak_endpoint(client):
    resp = await client.post("/tts/speak", json={"text": "Hello from test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "success" in data


@pytest.mark.asyncio
async def test_get_run_explain_returns_friendly_timeline_and_diagnosis(client):
    run_id = "r-friendly"
    state = RunState(run_id=run_id, goal="Open YouTube and verify")
    state.start()
    state.record_action("LAUNCH_APP", {"app_id": "youtube"}, 0.95, "open app", "PASS")
    state.record_action("GET_STATE", {"app_id": "youtube"}, 0.8, "verify", "FAIL")
    state.current_app = "youtube"
    state.current_screen = "BACKGROUND"
    state.current_app_state = "BACKGROUND"
    state.record_ai_event({"type": "planner-decision", "intent": "parse_failure", "reason": "parse_failure"})
    state.record_ai_event({"type": "planner-decision", "intent": "parse_failure_limit_reached", "reason": "parse_failure_limit_reached"})
    state.finish(RunStatus.FAILED, "parse_failure_limit_reached")
    _runs[run_id] = state

    resp = await client.get(f"/run/{run_id}/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert "timeline" in data and len(data["timeline"]) >= 2
    assert "what_screen_was_seen" in data["timeline"][0]
    assert data["diagnosis"]["root_cause"] == "Launch verification + planner parse failure loop"
    assert "could not confirm" in data["diagnosis"]["user_friendly_reason"].lower()
    assert "screen_based_reason" in data["diagnosis"]
    assert "goal_based_reason" in data["diagnosis"]
    assert "recovery_summary" in data["diagnosis"]


@pytest.mark.asyncio
async def test_get_run_explain_reports_repeated_ok_timeout_on_youtube(client):
    run_id = "r-youtube-ok-timeout"
    state = RunState(
        run_id=run_id,
        goal="Play any YouTube video and enable Stats for Nerds using the settings gear icon",
    )
    state.start()
    for _ in range(4):
        state.record_action("PRESS_OK", {"ok_intent": "SELECT_FOCUSED_CONTROL"}, 0.9, "loop", "PASS")
    state.record_ai_event({"type": "ok-effect-analysis", "step": 3, "effect": "NO_VISIBLE_CHANGE"})
    state.record_ai_event({"type": "commit-guard-blocked", "step": 4, "reason": "Repeated commit without progress"})
    state.latest_ocr_text = "Pause Up next"
    state.finish(RunStatus.TIMEOUT, "Max steps exceeded")
    _runs[run_id] = state

    resp = await client.get(f"/run/{run_id}/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["diagnosis"]["root_cause"] == "Repeated commit action in playback context"
    assert "kept pressing ok" in data["diagnosis"]["user_friendly_reason"].lower()


@pytest.mark.asyncio
async def test_get_run_history_not_found(client):
    resp = await client.get("/run/no-such-run/history")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_screenshot_not_found_for_run(client):
    start_resp = await client.post("/run/start", json={"goal": "Test screenshot"})
    run_id = start_resp.json()["run_id"]
    # The screenshot may or may not be present depending on async timing
    resp = await client.get(f"/run/{run_id}/screenshot")
    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_list_runs_returns_summary_items(client):
    await client.post("/run/start", json={"goal": "Run A"})
    await client.post("/run/start", json={"goal": "Run B"})
    resp = await client.get("/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert isinstance(runs, list)
    assert len(runs) >= 2
    # Each item should have the summary fields
    for r in runs:
        assert "run_id" in r
        assert "goal" in r
        assert "status" in r
        assert "step_count" in r
