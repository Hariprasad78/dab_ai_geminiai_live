"""Tests for the FastAPI backend."""
import asyncio
import json
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
    api_mod._yts_live_commands.clear()
    api_mod._yts_live_tasks.clear()
    api_mod._yts_live_visual_tasks.clear()
    api_mod._yts_live_processes.clear()
    api_mod._yts_live_visual_cache.clear()
    api_mod._dab_client = None
    api_mod._planner = None
    api_mod._screen_capture = None
    api_mod._tts_service = None
    api_mod._vertex_live_visual_client = None
    monkeypatch.setenv("ARTIFACTS_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("DAB_MOCK_MODE", "true")
    monkeypatch.setenv("IMAGE_SOURCE", "auto")
    monkeypatch.setenv("HDMI_AUDIO_ENABLED", "false")
    monkeypatch.setenv("ENABLE_VERTEX_PLANNER", "false")
    api_mod._close_yts_live_db()
    cfg_mod.reset_config()
    yield
    api_mod._yts_live_commands.clear()
    api_mod._yts_live_tasks.clear()
    api_mod._yts_live_visual_tasks.clear()
    api_mod._yts_live_processes.clear()
    api_mod._yts_live_visual_cache.clear()
    api_mod._dab_client = None
    api_mod._planner = None
    api_mod._screen_capture = None
    api_mod._tts_service = None
    api_mod._vertex_live_visual_client = None
    api_mod._close_yts_live_db()
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
    assert "video_devices" in data
    assert "device_readable" in data
    assert "user_in_video_group" in data
    assert "enable_hdmi_capture" in data
    assert "enable_camera_capture" in data


@pytest.mark.asyncio
async def test_capture_devices_status(client):
    resp = await client.get("/capture/devices")
    assert resp.status_code == 200
    data = resp.json()
    assert "devices" in data
    assert "configured_source" in data


@pytest.mark.asyncio
async def test_capture_select_updates_source(client):
    resp = await client.post(
        "/capture/select",
        json={"source": "dab", "preferred_kind": "camera", "persist": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["configured_source"] == "dab"
    assert data["preferred_video_kind"] == "camera"


@pytest.mark.asyncio
async def test_capture_select_rejects_invalid_source(client):
    resp = await client.post(
        "/capture/select",
        json={"source": "invalid-source", "persist": False},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_capture_select_rejects_hdmi_when_disabled(client, monkeypatch):
    monkeypatch.setenv("ENABLE_HDMI_CAPTURE", "false")
    cfg_mod.reset_config()
    api_mod._screen_capture = None

    resp = await client.post(
        "/capture/select",
        json={"source": "hdmi-capture", "persist": False},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_capture_select_rejects_camera_when_disabled(client, monkeypatch):
    monkeypatch.setenv("ENABLE_CAMERA_CAPTURE", "false")
    cfg_mod.reset_config()
    api_mod._screen_capture = None

    resp = await client.post(
        "/capture/select",
        json={"source": "camera-capture", "persist": False},
    )
    assert resp.status_code == 400


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


def test_get_vertex_live_visual_client_uses_non_live_model(monkeypatch):
    captured = {}

    class FakeVertexPlannerClient:
        def __init__(self, *, project, location, model):
            captured["project"] = project
            captured["location"] = location
            captured["model"] = model

    monkeypatch.setenv("ENABLE_VERTEX_PLANNER", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
    monkeypatch.setenv("VERTEX_PLANNER_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("VERTEX_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
    cfg_mod.reset_config()
    api_mod._vertex_live_visual_client = None
    monkeypatch.setattr(api_mod, "VertexPlannerClient", FakeVertexPlannerClient)

    client = api_mod.get_vertex_live_visual_client()

    assert client is not None
    assert captured["project"] == "demo-project"
    assert captured["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_list_runs_empty(client):
    resp = await client.get("/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_yts_job_endpoints(client):
    # create job via yts route
    start_resp = await client.post("/yts/job", json={"goal": "Verify YouTube launch"})
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert "run_id" in start_data
    run_id = start_data["run_id"]

    list_resp = await client.get("/yts/jobs")
    assert list_resp.status_code == 200
    assert any(r["run_id"] == run_id for r in list_resp.json())

    status_resp = await client.get(f"/yts/job/{run_id}")
    assert status_resp.status_code == 200
    st = status_resp.json()
    assert st["run_id"] == run_id

    stop_resp = await client.post(f"/yts/job/{run_id}/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["run_id"] == run_id


@pytest.mark.asyncio
async def test_yts_tests_catalog(client):
    resp = await client.get("/yts/tests")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data
    assert any(item.get("test_id") for item in data)
    assert any(item.get("test_title") for item in data)


@pytest.mark.asyncio
async def test_yts_tests_catalog_guided(client, monkeypatch):
    monkeypatch.setattr(api_mod, "_read_yts_test_catalog", lambda path: [])

    def fake_refresh(path, guided=False, raise_on_error=False):
        assert guided is True
        return [{"test_id": "GUIDED-001", "test_title": "Guided sample test"}]

    monkeypatch.setattr(api_mod, "_refresh_yts_test_catalog", fake_refresh)

    resp = await client.get("/yts/tests?guided=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data == [{"test_id": "GUIDED-001", "test_title": "Guided sample test"}]


@pytest.mark.asyncio
async def test_yts_command_supports_multiple_test_ids(client, monkeypatch, tmp_path):
    output_file = tmp_path / "yts-result.json"

    class FakeCompletedProcess:
        def __init__(self, stdout: str = "ok"):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args_list, capture_output, text, cwd, check):
        if args_list[:3] == ["yts", "discover", "--list"]:
            return FakeCompletedProcess(stdout="")
        assert args_list[:3] == ["yts", "test", "adb:device-01"]
        assert "ID-001" in args_list
        assert "ID-002" in args_list
        assert "--json-output" in args_list
        output_file.write_text(json.dumps({"tests": ["ID-001", "ID-002"]}), encoding="utf-8")
        return FakeCompletedProcess()

    monkeypatch.setattr(api_mod.subprocess, "run", fake_run)

    resp = await client.post(
        "/yts/command",
        json={
            "command": "test",
            "params": ["adb:device-01", "ID-001", "ID-002", "--json-output", str(output_file)],
            "output_file": str(output_file),
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["returncode"] == 0
    assert "ID-001" in data["command"]
    assert "ID-002" in data["command"]
    assert data["result_file_name"] == output_file.name
    assert json.loads(data["result_file_content"]) == {"tests": ["ID-001", "ID-002"]}


@pytest.mark.asyncio
async def test_yts_live_command_logs(client, monkeypatch):
    async def fake_run(command_id, request):
        state = api_mod._yts_live_commands[command_id]
        state["command"] = "yts list --guided"
        state["logs"].append({"stream": "stdout", "message": "loading"})
        state["stdout"] = "loading\n"
        state["returncode"] = 0
        state["status"] = "completed"

    monkeypatch.setattr(api_mod, "_run_yts_command_live", fake_run)

    start_resp = await client.post(
        "/yts/command/live",
        json={"command": "list", "params": ["--guided"]},
    )
    assert start_resp.status_code == 200
    command_id = start_resp.json()["command_id"]

    await asyncio.sleep(0)

    status_resp = await client.get(f"/yts/command/live/{command_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["status"] == "completed"
    assert data["command"] == "yts list --guided"
    assert data["logs"][0]["message"] == "loading"


@pytest.mark.asyncio
async def test_yts_live_command_record_video_flow(monkeypatch):
    command_id = "cmd-record-video"
    state = api_mod._new_yts_live_state(command_id)
    state["record_video"] = True
    api_mod._yts_live_commands[command_id] = state

    class FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        async def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return b""

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStream([b"loading\n"])
            self.stderr = FakeStream([])
            self.stdin = None
            self.returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_create_subprocess_exec(*args, **kwargs):
        assert args[:2] == ("yts", "list")
        return FakeProcess()

    async def fake_start_recording(command_id_arg):
        assert command_id_arg == command_id
        recording_path = api_mod._get_yts_live_artifacts_dir(command_id_arg) / "session.mp4"
        recording_path.write_bytes(b"video-data")
        live_state = api_mod._yts_live_commands[command_id_arg]
        live_state["video_recording_status"] = "recording"
        live_state["video_file_path"] = str(recording_path)
        live_state["video_file_name"] = recording_path.name

    async def fake_stop_recording(command_id_arg):
        assert command_id_arg == command_id
        api_mod._yts_live_commands[command_id_arg]["video_recording_status"] = "completed"

    monkeypatch.setattr(api_mod.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(api_mod, "_start_yts_video_recording", fake_start_recording)
    monkeypatch.setattr(api_mod, "_stop_yts_video_recording", fake_stop_recording)

    request = api_mod.YtsCommandRequest(command="list", params=["--guided"], record_video=True)
    await api_mod._run_yts_command_live(command_id, request)

    final_state = api_mod._yts_live_commands[command_id]
    assert final_state["status"] == "completed"
    assert final_state["video_recording_status"] == "completed"
    assert final_state["video_file_name"] == "session.mp4"
    assert "loading" in final_state["stdout"]


@pytest.mark.asyncio
async def test_yts_terminal_log_download(client):
    command_id = "cmd-terminal-log"
    state = api_mod._new_yts_live_state(command_id)
    state["command"] = "yts list"
    state["logs"] = [{"stream": "stdout", "message": "loading"}]
    state["stdout"] = "loading\n"
    api_mod._yts_live_commands[command_id] = state

    resp = await client.get(f"/yts/command/live/{command_id}/terminal-log")
    assert resp.status_code == 200
    assert "loading" in resp.text
    assert resp.headers["content-type"].startswith("text/plain")


@pytest.mark.asyncio
async def test_yts_video_download_returns_recorded_artifact(client):
    command_id = "cmd-video-log"
    state = api_mod._new_yts_live_state(command_id)
    video_path = api_mod._get_yts_live_artifacts_dir(command_id) / "session.mp4"
    video_path.write_bytes(b"fake-video")
    state["record_video"] = True
    state["video_file_name"] = video_path.name
    state["video_file_path"] = str(video_path)
    state["video_recording_status"] = "completed"
    api_mod._yts_live_commands[command_id] = state

    resp = await client.get(f"/yts/command/live/{command_id}/video")
    assert resp.status_code == 200
    assert resp.content == b"fake-video"
    assert resp.headers["content-type"].startswith("video/mp4")


@pytest.mark.asyncio
async def test_yts_live_command_manual_response(client, monkeypatch):
    command_id = "cmd-manual"
    api_mod._yts_live_commands[command_id] = {
        "command_id": command_id,
        "status": "running",
        "pending_prompt": {"id": 1, "text": "Continue? yes/no", "options": ["yes", "no"], "answered": False},
        "prompts": [{"id": 1, "text": "Continue? yes/no", "options": ["yes", "no"], "answered": False}],
    }

    async def fake_send(command_id_arg, response_text, source="manual"):
        assert command_id_arg == command_id
        assert response_text == "yes"
        assert source == "manual"
        return {"command_id": command_id_arg, "response": response_text, "source": source}

    monkeypatch.setattr(api_mod, "_send_yts_command_input", fake_send)

    resp = await client.post(f"/yts/command/live/{command_id}/respond", json={"response": "yes"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "yes"


@pytest.mark.asyncio
async def test_yts_live_command_suggest_and_send(client, monkeypatch):
    command_id = "cmd-suggest"
    api_mod._yts_live_commands[command_id] = {
        "command_id": command_id,
        "status": "running",
        "pending_prompt": {"id": 1, "text": "Select option 1/2/3/4", "options": ["1", "2", "3", "4"], "answered": False},
        "prompts": [{"id": 1, "text": "Select option 1/2/3/4", "options": ["1", "2", "3", "4"], "answered": False}],
    }

    async def fake_suggest(command_id_arg, prompt_text, options=None):
        assert command_id_arg == command_id
        assert "Select option" in prompt_text
        return {"response": "1", "source": "gemini"}

    async def fake_send(command_id_arg, response_text, source="manual"):
        assert command_id_arg == command_id
        assert response_text == "1"
        assert source == "gemini"
        return {"command_id": command_id_arg, "response": response_text, "source": source}

    monkeypatch.setattr(api_mod, "_suggest_yts_prompt_response", fake_suggest)
    monkeypatch.setattr(api_mod, "_send_yts_command_input", fake_send)

    resp = await client.post(f"/yts/command/live/{command_id}/suggest", json={"send_response": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["suggestion"] == "1"
    assert data["source"] == "gemini"
    assert data["sent"] is True


@pytest.mark.asyncio
async def test_yts_live_command_suggest_rejects_incomplete_numeric_prompt(client, monkeypatch):
    command_id = "cmd-suggest-incomplete"
    api_mod._yts_live_commands[command_id] = {
        "command_id": command_id,
        "status": "running",
        "awaiting_input": True,
        "pending_prompt": {
            "id": 1,
            "text": "Does the image on screen render correctly?\nPlease select from the following options:",
            "options": [],
            "answered": False,
        },
        "prompts": [
            {
                "id": 1,
                "text": "Does the image on screen render correctly?\nPlease select from the following options:",
                "options": [],
                "answered": False,
            }
        ],
    }

    async def fail_suggest(*_args, **_kwargs):
        raise AssertionError("Gemini should not be called for an incomplete numeric prompt")

    monkeypatch.setattr(api_mod, "_suggest_yts_prompt_response", fail_suggest)

    resp = await client.post(f"/yts/command/live/{command_id}/suggest", json={"send_response": True})
    assert resp.status_code == 409
    assert "collecting the full question/options" in resp.text


@pytest.mark.asyncio
async def test_yts_live_command_auto_response_persists_multiline_prompt(monkeypatch):
    command_id = "cmd-auto-prompt"
    state = api_mod._new_yts_live_state(command_id, interactive_ai=True)
    api_mod._yts_live_commands[command_id] = state

    class FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        async def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return b""

    class FakeStdin:
        def __init__(self):
            self.writes = []

        def write(self, payload):
            self.writes.append(payload.decode())

        async def drain(self):
            return None

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.returncode = None

    async def fake_suggest(command_id_arg, prompt_text, options=None):
        assert command_id_arg == command_id
        assert "Does the image on screen render correctly?" in prompt_text
        assert "1: Yes" in prompt_text
        assert options == ["1", "2"]
        return {
            "response": "1",
            "source": "gemini",
            "visual_summary": "TV shows correct render",
            "visual_source": "hdmi-capture",
        }

    api_mod._yts_live_processes[command_id] = FakeProcess()
    monkeypatch.setattr(api_mod, "_suggest_yts_prompt_response", fake_suggest)

    await api_mod._append_yts_stream_output(
        command_id,
        "stdout",
        FakeStream(
            [
                b"\x1b[2m19:07:04.926\x1b[22m     Does the image on screen render correctly? (Marbles Lossless WebP)\n",
                b"\x1b[2m19:07:04.926\x1b[22m     Please select from the following options:\n",
                b"\x1b[2m19:07:04.926\x1b[22m     1: Yes\n",
                b"\x1b[2m19:07:04.926\x1b[22m     2: No\n",
            ]
        ),
    )

    final_state = api_mod._yts_live_commands[command_id]
    assert len(final_state["prompts"]) == 1
    prompt = final_state["prompts"][0]
    assert "Does the image on screen render correctly?" in prompt["text"]
    assert "1: Yes" in prompt["text"]
    assert prompt["options"] == ["1", "2"]
    assert prompt["answered"] is True
    assert prompt["response"] == "1"
    assert prompt["response_source"] == "gemini"
    assert prompt["ai_suggestion"] == "1"
    assert final_state["responses"] == [{"source": "gemini", "message": "1"}]
    assert api_mod._yts_live_processes[command_id].stdin.writes == ["1\n"]


@pytest.mark.asyncio
async def test_yts_visual_context_prefers_live_stream_when_hdmi_available(monkeypatch):
    command_id = "cmd-hdmi-only"
    api_mod._yts_live_commands[command_id] = api_mod._new_yts_live_state(command_id, interactive_ai=True)

    class FakeLiveResult:
        image_b64 = "live-image"
        source = "hdmi-capture"

    class FakeCaptureService:
        def __init__(self):
            self.live_calls = 0
            self.capture_calls = 0

        async def capture_live_stream_frame(self):
            self.live_calls += 1
            return FakeLiveResult()

        async def capture(self):
            self.capture_calls += 1
            raise AssertionError("DAB/general capture should not run when HDMI is healthy")

        def capture_source_status(self):
            return {
                "configured_source": "auto",
                "selected_video_device": "/dev/video0",
                "hdmi_available": True,
            }

    capture = FakeCaptureService()
    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: capture)
    visual_context = await api_mod._capture_yts_visual_context(command_id)

    assert capture.live_calls == api_mod._YTS_INTERACTIVE_CAPTURE_ATTEMPTS
    assert capture.capture_calls == 0
    assert visual_context["source"] == "hdmi-capture"
    assert visual_context["capture_status"]["live_stream_only"] is True


def test_prompt_ready_for_ai_response_requires_real_question_context():
    prompt_entry = {
        "text": "Please select from the following options:\n1: Yes\n2: No",
        "options": ["1", "2"],
    }

    assert api_mod._prompt_ready_for_ai_response(prompt_entry) is False


@pytest.mark.asyncio
async def test_yts_stream_output_does_not_answer_scaffolding_only_prompt(monkeypatch):
    command_id = "cmd-scaffold-only"
    api_mod._yts_live_commands[command_id] = api_mod._new_yts_live_state(command_id, interactive_ai=True)

    class FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        async def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return b""

    async def fail_suggest(*_args, **_kwargs):
        raise AssertionError("Gemini should not run for scaffolding-only prompt text")

    monkeypatch.setattr(api_mod, "_suggest_yts_prompt_response", fail_suggest)

    await api_mod._append_yts_stream_output(
        command_id,
        "stdout",
        FakeStream(
            [
                b"Please select from the following options:\n",
                b"1: Yes\n",
                b"2: No\n",
            ]
        ),
    )

    state = api_mod._yts_live_commands[command_id]
    assert len(state["prompts"]) == 1
    assert state["prompts"][0]["answered"] is False
    assert state["responses"] == []


@pytest.mark.asyncio
async def test_yts_stream_output_records_closed_prompt_without_crashing(monkeypatch):
    command_id = "cmd-closed-prompt"
    state = api_mod._new_yts_live_state(command_id, interactive_ai=True)
    api_mod._yts_live_commands[command_id] = state

    class FakeStream:
        def __init__(self, lines):
            self._lines = list(lines)

        async def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return b""

    async def fake_suggest(_command_id, _prompt_text, _options=None):
        api_mod._yts_live_commands[command_id]["awaiting_input"] = False
        api_mod._yts_live_commands[command_id]["pending_prompt"] = None
        return {
            "response": "yes",
            "source": "gemini",
            "visual_summary": "summary",
            "visual_source": "hdmi-capture",
        }

    monkeypatch.setattr(api_mod, "_suggest_yts_prompt_response", fake_suggest)

    await api_mod._append_yts_stream_output(
        command_id,
        "stdout",
        FakeStream([b"Continue? yes/no\n"]),
    )

    prompt = api_mod._yts_live_commands[command_id]["prompts"][0]
    assert prompt["answered"] is False
    assert prompt["ai_error"] == "Prompt closed before AI response could be sent"


def test_merge_yts_prompt_entry_clears_stale_ai_suggestion_on_prompt_expansion():
    prompt_entry = {
        "id": 1,
        "text": "Does the image on screen render correctly?\nPlease select from the following options:",
        "options": [],
        "stream": "stdout",
        "answered": False,
        "ai_suggestion": "no",
        "ai_source": "gemini",
    }

    api_mod._merge_yts_prompt_entry(prompt_entry, "1: Yes", "stdout")

    assert prompt_entry["options"] == ["1"]
    assert "ai_suggestion" not in prompt_entry
    assert "ai_source" not in prompt_entry


def test_prompt_ready_for_ai_response_waits_for_numeric_options():
    prompt_entry = {
        "text": "Does the image on screen render correctly?",
        "options": [],
    }

    assert api_mod._prompt_ready_for_ai_response(prompt_entry) is False

    prompt_entry = {
        "text": "Does the image on screen render correctly?\nPlease select from the following options:\n1: Yes",
        "options": ["1"],
    }

    assert api_mod._prompt_ready_for_ai_response(prompt_entry) is False

    prompt_entry = {
        "text": "Does the image on screen render correctly?\nPlease select from the following options:\n1: Yes\n2: No",
        "options": ["1", "2"],
    }

    assert api_mod._prompt_ready_for_ai_response(prompt_entry) is True


def test_yts_prompt_requires_numeric_response_only_after_numeric_options_visible():
    assert api_mod._yts_prompt_requires_numeric_response(
        "Does the image on screen render correctly?",
        [],
    ) is False

    assert api_mod._yts_prompt_requires_numeric_response(
        "Does the image on screen render correctly?\nPlease select from the following options:\n1: Yes",
        ["1"],
    ) is False

    assert api_mod._yts_prompt_requires_numeric_response(
        "Does the image on screen render correctly?\nPlease select from the following options:\n1: Yes\n2: No",
        ["1", "2"],
    ) is True


@pytest.mark.asyncio
async def test_yts_prompt_suggestion_maps_yes_no_to_numeric_option(monkeypatch):
    command_id = "cmd-numeric-mapping"
    state = api_mod._new_yts_live_state(command_id, interactive_ai=True)
    state["logs"] = [{"stream": "stdout", "message": "Choose the matching numeric option."}]
    api_mod._yts_live_commands[command_id] = state

    class FakeCaptureResult:
        image_b64 = "img"
        source = "hdmi-capture"

    class FakeCaptureService:
        async def capture(self):
            return FakeCaptureResult()

        def capture_source_status(self):
            return {
                "configured_source": "hdmi-capture",
                "selected_video_device": "/dev/video0",
                "hdmi_available": True,
            }

    class FakeVertexClient:
        async def generate_content(self, prompt, screenshot_b64=None, session_id=None):
            assert "Return only one numeric option token" in prompt
            assert "Do not return yes or no" in prompt
            return "yes"

    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: FakeCaptureService())
    monkeypatch.setattr(api_mod, "get_vertex_text_client", lambda: FakeVertexClient())

    suggestion = await api_mod._suggest_yts_prompt_response(
        command_id,
        "Does the image on screen render correctly?\nPlease select from the following options:\n1: Yes\n2: No",
        ["1", "2"],
    )

    assert suggestion["response"] == "1"


def test_heuristic_yts_prompt_response_prefers_numeric_option_for_numbered_prompt():
    response = api_mod._heuristic_yts_prompt_response(
        "Does the image on screen render correctly?\nPlease select from the following options:\n1: Yes\n2: No",
        ["1", "2"],
    )

    assert response == "1"


@pytest.mark.asyncio
async def test_refresh_yts_live_visual_monitor_records_gemini_analysis(monkeypatch):
    command_id = "cmd-live-monitor"
    state = api_mod._new_yts_live_state(command_id, interactive_ai=True)
    state["command"] = "yts guided test"
    state["logs"] = [{"stream": "stdout", "message": "Video validation in progress"}]
    api_mod._yts_live_commands[command_id] = state

    class FakeCaptureResult:
        image_b64 = "live-image"
        source = "hdmi-capture"

    class FakeCaptureService:
        async def capture_live_stream_frame(self):
            return FakeCaptureResult()

        async def capture(self):
            raise AssertionError("monitor should prefer live-stream-only capture when HDMI is healthy")

        def capture_source_status(self):
            return {
                "configured_source": "auto",
                "selected_video_device": "/dev/video0",
                "hdmi_available": True,
            }

    class FakeVertexClient:
        async def generate_content(self, prompt, screenshot_b64=None, session_id=None):
            assert "Do not rely on OCR or local text extraction" in prompt
            assert screenshot_b64 == "live-image"
            assert session_id == f"yts-live-visual-{command_id}"
            return json.dumps(
                {
                    "summary": "Playback is visible and player controls are open.",
                    "playback_visible": True,
                    "player_controls_visible": True,
                    "settings_gear_visible": True,
                    "stats_for_nerds_visible": False,
                    "focus_target": "settings gear",
                    "confidence": 0.92,
                }
            )

    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: FakeCaptureService())
    monkeypatch.setattr(api_mod, "get_vertex_live_visual_client", lambda: FakeVertexClient())
    monkeypatch.setattr(api_mod, "get_vertex_text_client", lambda: None)

    result = await api_mod._refresh_yts_live_visual_monitor(command_id)

    assert result is not None
    assert result["analysis"]["playback_visible"] is True
    assert api_mod._yts_live_commands[command_id]["visual_monitor_active"] is True
    assert api_mod._yts_live_commands[command_id]["latest_visual_analysis"]["analysis"]["settings_gear_visible"] is True
    assert api_mod._yts_live_visual_cache[command_id]["screenshot_b64"] == "live-image"


def test_parse_yts_live_visual_analysis_accepts_text_confidence():
    parsed = api_mod._parse_yts_live_visual_analysis(
        json.dumps(
            {
                "summary": "Playback visible.",
                "playback_visible": True,
                "player_controls_visible": True,
                "settings_gear_visible": False,
                "stats_for_nerds_visible": False,
                "focus_target": "player",
                "confidence": "High",
            }
        )
    )

    assert parsed["playback_visible"] is True
    assert parsed["confidence"] == 0.85


@pytest.mark.asyncio
async def test_capture_yts_visual_context_prefers_fresh_live_monitor_cache(monkeypatch):
    command_id = "cmd-live-cache"
    api_mod._yts_live_commands[command_id] = api_mod._new_yts_live_state(command_id, interactive_ai=True)
    api_mod._yts_live_visual_cache[command_id] = {
        "summary": "Cached live Gemini analysis",
        "source": "hdmi-capture",
        "screenshot_b64": "cached-image",
        "observations": [{"attempt": 1, "source": "hdmi-capture", "has_screenshot": True}],
        "capture_status": {"configured_source": "auto", "hdmi_available": True, "live_stream_only": True},
        "analysis": {"playback_visible": True, "summary": "Playback visible"},
        "captured_at": api_mod._utc_now_iso(),
    }

    class FailingCaptureService:
        def capture_source_status(self):
            raise AssertionError("fresh visual cache should avoid immediate recapture")

    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: FailingCaptureService())

    visual_context = await api_mod._capture_yts_visual_context(command_id)

    assert visual_context["screenshot_b64"] == "cached-image"
    assert visual_context["summary"] == "Cached live Gemini analysis"
    assert visual_context["analysis"]["playback_visible"] is True


@pytest.mark.asyncio
async def test_yts_setup_actions_execute_for_timezone_guidance(monkeypatch):
    command_id = "cmd-setup-timezone"
    api_mod._yts_live_commands[command_id] = api_mod._new_yts_live_state(command_id, interactive_ai=True)

    executed = []

    async def fake_manual_action(request):
        executed.append((request.action, request.params))
        return api_mod.ManualActionResponse(success=True, action=request.action, result=request.params)

    monkeypatch.setattr(api_mod, "manual_action", fake_manual_action)
    result = await api_mod._maybe_execute_yts_setup_actions(
        command_id,
        "Please change time zone to UTC before continuing.",
        "[stdout] Operator setup: change time zone to UTC",
    )

    assert executed == [("SET_SETTING", {"key": "timezone", "value": "UTC"})]
    assert result[0]["success"] is True
    assert api_mod._yts_live_commands[command_id]["setup_actions"]


@pytest.mark.asyncio
async def test_yts_setup_actions_execute_for_language_guidance(monkeypatch):
    command_id = "cmd-setup-language"
    api_mod._yts_live_commands[command_id] = api_mod._new_yts_live_state(command_id, interactive_ai=True)

    executed = []

    async def fake_manual_action(request):
        executed.append((request.action, request.params))
        return api_mod.ManualActionResponse(success=True, action=request.action, result=request.params)

    monkeypatch.setattr(api_mod, "manual_action", fake_manual_action)
    result = await api_mod._maybe_execute_yts_setup_actions(
        command_id,
        (
            "This test monitors changes to your device's navigator.language setting.\n"
            "To proceed, go to the device settings, change the value of the navigator.language setting, and then press OK.\n"
            "Initial value of navigator.language: en-GB.\n"
            "Please select from the following options:\n1: OK\n2: Not supported\n3: Fail\n4: Skip"
        ),
        "[stdout] initial_value: en-GB",
    )

    assert executed == [("SET_SETTING", {"key": "language", "value": "en-US"})]
    assert result[0]["success"] is True
    assert api_mod._yts_live_commands[command_id]["setup_actions"]


@pytest.mark.asyncio
async def test_yts_prompt_suggestion_uses_active_test_context_for_options_only_prompt(monkeypatch):
    command_id = "cmd-options-only-context"
    state = api_mod._new_yts_live_state(command_id, interactive_ai=True)
    state["command"] = "yts verify settings-language"
    state["logs"] = [
        {"stream": "stdout", "message": "-----------------"},
        {"stream": "stdout", "message": "Settings Language"},
        {"stream": "stdout", "message": "-----------------"},
        {"stream": "stdout", "message": "This test monitors changes to your device's navigator.language setting."},
        {"stream": "stdout", "message": "To proceed, go to the device settings, change the value of the navigator.language setting, and then press OK."},
        {"stream": "stdout", "message": "Initial value of navigator.language: en-GB."},
        {"stream": "stdout", "message": "Please select from the following options:"},
        {"stream": "stdout", "message": "1: OK"},
        {"stream": "stdout", "message": "2: Not supported"},
        {"stream": "stdout", "message": "3: Fail"},
        {"stream": "stdout", "message": "4: Skip"},
    ]
    api_mod._yts_live_commands[command_id] = state

    executed = []

    async def fake_manual_action(request):
        executed.append((request.action, request.params))
        return api_mod.ManualActionResponse(success=True, action=request.action, result=request.params)

    class FakeCaptureResult:
        image_b64 = "img"
        source = "hdmi-capture"

    class FakeCaptureService:
        async def capture(self):
            return FakeCaptureResult()

        def capture_source_status(self):
            return {
                "configured_source": "hdmi-capture",
                "selected_video_device": "/dev/video0",
                "hdmi_available": True,
            }

    class FakeVertexClient:
        def __init__(self):
            self.calls = []

        async def generate_content(self, prompt, screenshot_b64=None, session_id=None):
            self.calls.append(prompt)
            return "1"

    fake_client = FakeVertexClient()
    monkeypatch.setattr(api_mod, "manual_action", fake_manual_action)
    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: FakeCaptureService())
    monkeypatch.setattr(api_mod, "get_vertex_text_client", lambda: fake_client)

    suggestion = await api_mod._suggest_yts_prompt_response(
        command_id,
        "Please select from the following options:\n1: OK\n2: Not supported\n3: Fail\n4: Skip",
        ["1", "2", "3", "4"],
    )

    assert suggestion["response"] == "1"
    assert executed == [("SET_SETTING", {"key": "language", "value": "en-US"})]
    assert fake_client.calls
    assert "Settings Language" in fake_client.calls[0]
    assert "navigator.language setting" in fake_client.calls[0]
    assert "Initial value of navigator.language: en-GB." in fake_client.calls[0]
    assert "Active test terminal context:" in fake_client.calls[0]


def test_plan_task_macro_actions_supports_explicit_language_setting():
    actions = api_mod._plan_task_macro_actions("set navigator.language to en-US then press ok")

    assert actions[0].action == "SET_SETTING"
    assert actions[0].params == {"key": "language", "value": "en-US"}
    assert actions[1].action == "PRESS_OK"


@pytest.mark.asyncio
async def test_manual_action_set_setting_normalizes_language_alias(client, monkeypatch):
    class FakeResponse:
        def __init__(self, success, status, data):
            self.success = success
            self.status = status
            self.data = data

    class FakeDAB:
        async def list_operations(self):
            return FakeResponse(True, 200, {"operations": ["system/settings/set"]})

        async def get_device_info(self):
            return FakeResponse(True, 200, {"platform": "android-tv", "adbDeviceId": "emulator-5554"})

        async def set_setting(self, key, value):
            return FakeResponse(True, 200, {"key": key, "value": value, "updated": True})

    monkeypatch.setattr(api_mod, "get_dab_client", lambda: FakeDAB())
    async def _platform_info(_device_id):
        return {
            "reachable": True,
            "is_android": True,
            "is_android_tv": True,
            "connection_type": "usb",
            "error": None,
        }
    monkeypatch.setattr(api_mod, "get_device_platform_info", _platform_info)

    resp = await client.post(
        "/action",
        json={"action": "SET_SETTING", "params": {"key": "navigator.language", "value": "en-US"}},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["key"] == "language"
    assert data["result"]["value"] == "en-US"


@pytest.mark.asyncio
async def test_manual_action_timezone_uses_android_adb_fallback_when_dab_unavailable(client, monkeypatch):
    class FakeResponse:
        def __init__(self, success, status, data):
            self.success = success
            self.status = status
            self.data = data

    class FakeDAB:
        async def list_operations(self):
            return FakeResponse(True, 200, {"operations": []})

        async def set_setting(self, key, value):
            return FakeResponse(False, 501, {"error": "system/settings/set not supported", "key": key, "value": value})

        async def get_device_info(self):
            return FakeResponse(True, 200, {"platform": "android-tv", "adbDeviceId": "emulator-5554"})

    async def _online(_device_id):
        return True, "device"

    async def _list_tz(_device_id):
        return {"success": False, "timezones": [], "error": "tzdata not found"}

    async def _set_tz(_device_id, value):
        return {
            "success": True,
            "requested_timezone": value,
            "observed_timezone": value,
            "verified": True,
        }

    monkeypatch.setattr(api_mod, "get_dab_client", lambda: FakeDAB())
    async def _platform_info(_device_id):
        return {
            "reachable": True,
            "is_android": True,
            "is_android_tv": True,
            "connection_type": "usb",
            "error": None,
        }
    monkeypatch.setattr(api_mod, "get_device_platform_info", _platform_info)
    monkeypatch.setattr(api_mod, "is_adb_device_online", _online)
    monkeypatch.setattr(api_mod, "list_timezones_via_adb", _list_tz)
    monkeypatch.setattr(api_mod, "set_timezone_via_adb", _set_tz)

    resp = await client.post(
        "/action",
        json={"action": "SET_SETTING", "params": {"key": "timezone", "value": "America/Los_Angeles"}},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["result"]["path"] == "ADB_FALLBACK"
    assert data["result"]["verification"]["matched"] is True


@pytest.mark.asyncio
async def test_manual_action_timezone_does_not_fallback_when_adb_offline(client, monkeypatch):
    class FakeResponse:
        def __init__(self, success, status, data):
            self.success = success
            self.status = status
            self.data = data

    class FakeDAB:
        async def list_operations(self):
            return FakeResponse(True, 200, {"operations": []})

        async def set_setting(self, key, value):
            return FakeResponse(False, 501, {"error": "system/settings/set not supported", "key": key, "value": value})

        async def get_device_info(self):
            return FakeResponse(True, 200, {"platform": "android-tv", "adbDeviceId": "emulator-5554"})

    async def _online(_device_id):
        return False, "offline"

    monkeypatch.setattr(api_mod, "get_dab_client", lambda: FakeDAB())
    async def _platform_info(_device_id):
        return {
            "reachable": True,
            "is_android": True,
            "is_android_tv": True,
            "connection_type": "usb",
            "error": None,
        }
    monkeypatch.setattr(api_mod, "get_device_platform_info", _platform_info)
    monkeypatch.setattr(api_mod, "is_adb_device_online", _online)

    resp = await client.post(
        "/action",
        json={"action": "SET_SETTING", "params": {"key": "timezone", "value": "UTC"}},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "unavailable" in str(data.get("error", "")).lower() or "offline" in str(data.get("error", "")).lower()


@pytest.mark.asyncio
async def test_yts_prompt_suggestion_uses_tv_visual_context(monkeypatch):
    command_id = "cmd-visual"
    state = api_mod._new_yts_live_state(command_id, interactive_ai=True)
    state["logs"] = [
        {"stream": "stdout", "message": "Guide: if the TV shows the pairing screen, answer yes."},
        {"stream": "stdout", "message": "Waiting for confirmation"},
    ]
    api_mod._yts_live_commands[command_id] = state

    class FakeCaptureResult:
        image_b64 = "ZmFrZS1pbWFnZQ=="
        ocr_text = "TV screen shows pairing confirmation dialog"
        source = "hdmi-capture"

    class FakeCaptureService:
        async def capture(self):
            return FakeCaptureResult()

        def capture_source_status(self):
            return {
                "configured_source": "hdmi-capture",
                "selected_video_device": "/dev/video0",
                "hdmi_available": True,
            }

    class FakeVertexClient:
        def __init__(self):
            self.calls = []

        async def generate_content(self, prompt, screenshot_b64=None, session_id=None):
            self.calls.append(
                {
                    "prompt": prompt,
                    "screenshot_b64": screenshot_b64,
                    "session_id": session_id,
                }
            )
            return "yes"

    fake_client = FakeVertexClient()
    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: FakeCaptureService())
    monkeypatch.setattr(api_mod, "get_vertex_text_client", lambda: fake_client)

    suggestion = await api_mod._suggest_yts_prompt_response(command_id, "Continue? yes/no", ["yes", "no"])

    assert suggestion["response"] == "yes"
    assert suggestion["source"] == "gemini"
    assert suggestion["visual_source"] == "hdmi-capture"
    assert "Use the attached screenshot directly" in suggestion["visual_summary"]
    assert fake_client.calls
    assert fake_client.calls[0]["screenshot_b64"] == "ZmFrZS1pbWFnZQ=="
    assert "Guide: if the TV shows the pairing screen" in fake_client.calls[0]["prompt"]
    assert "TV screen shows pairing confirmation dialog" not in fake_client.calls[0]["prompt"]
    assert "attached screenshot is the primary visual context" in fake_client.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_yts_visual_context_collects_multiple_observations(monkeypatch):
    command_id = "cmd-operator"
    api_mod._yts_live_commands[command_id] = api_mod._new_yts_live_state(command_id, interactive_ai=True)

    class FakeCaptureResult:
        def __init__(self, image_b64, ocr_text, source="hdmi-capture"):
            self.image_b64 = image_b64
            self.ocr_text = ocr_text
            self.source = source

    class FakeCaptureService:
        def __init__(self):
            self.results = [
                FakeCaptureResult("img-1", "Loading spinner"),
                FakeCaptureResult("img-2", "Select the correct account"),
                FakeCaptureResult("img-3", "Select the correct account"),
            ]

        async def capture(self):
            return self.results.pop(0)

        def capture_source_status(self):
            return {
                "configured_source": "hdmi-capture",
                "selected_video_device": "/dev/video0",
                "hdmi_available": True,
            }

    async def fast_sleep(_seconds):
        return None

    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: FakeCaptureService())
    monkeypatch.setattr(api_mod.asyncio, "sleep", fast_sleep)

    visual_context = await api_mod._capture_yts_visual_context(command_id)

    assert visual_context["source"] == "hdmi-capture"
    assert visual_context["screenshot_b64"] == "img-3"
    assert len(visual_context["observations"]) == api_mod._YTS_INTERACTIVE_CAPTURE_ATTEMPTS
    assert visual_context["observations"][0]["has_screenshot"] is True
    assert visual_context["observations"][2]["has_screenshot"] is True
    assert "Latest stable OCR" not in visual_context["summary"]
    assert "Use the attached screenshot directly" in visual_context["summary"]


@pytest.mark.asyncio
async def test_yts_prompt_suggestion_toggles_ai_observing_state(monkeypatch):
    command_id = "cmd-observing"
    api_mod._yts_live_commands[command_id] = api_mod._new_yts_live_state(command_id, interactive_ai=True)

    class FakeCaptureResult:
        image_b64 = "img-final"
        ocr_text = "Confirmation screen visible"
        source = "hdmi-capture"

    class FakeCaptureService:
        async def capture(self):
            return FakeCaptureResult()

        def capture_source_status(self):
            return {
                "configured_source": "hdmi-capture",
                "selected_video_device": "/dev/video0",
                "hdmi_available": True,
            }

    class FakeVertexClient:
        async def generate_content(self, prompt, screenshot_b64=None, session_id=None):
            state_during_call = api_mod._yts_live_commands[command_id]
            assert state_during_call["ai_observing_tv"] is True
            assert "watching the TV stream" in (state_during_call["ai_status_message"] or "")
            return "yes"

    monkeypatch.setattr(api_mod, "get_screen_capture", lambda: FakeCaptureService())
    monkeypatch.setattr(api_mod, "get_vertex_text_client", lambda: FakeVertexClient())

    suggestion = await api_mod._suggest_yts_prompt_response(command_id, "Continue? yes/no", ["yes", "no"])

    assert suggestion["response"] == "yes"
    final_state = api_mod._yts_live_commands[command_id]
    assert final_state["ai_observing_tv"] is False
    assert final_state["ai_status_message"] is None


@pytest.mark.asyncio
async def test_yts_live_command_lookup_reads_persisted_state(client):
    state = api_mod._new_yts_live_state("cmd-persisted", interactive_ai=True)
    state["command"] = "yts list --guided"
    state["status"] = "completed"
    state["stdout"] = "done\n"
    state["logs"].append({"stream": "stdout", "message": "done"})
    api_mod._persist_yts_live_state(state)
    api_mod._yts_live_commands.clear()

    resp = await client.get("/yts/command/live/cmd-persisted")
    assert resp.status_code == 200
    data = resp.json()
    assert data["command_id"] == "cmd-persisted"
    assert data["interactive_ai"] is True
    assert data["status"] == "completed"
    assert data["logs"][0]["message"] == "done"


@pytest.mark.asyncio
async def test_yts_live_command_list_returns_active_persisted_commands(client):
    active = api_mod._new_yts_live_state("cmd-active")
    active["command"] = "yts test adb:device-01"
    active["status"] = "running"
    api_mod._persist_yts_live_state(active)

    completed = api_mod._new_yts_live_state("cmd-completed")
    completed["status"] = "completed"
    completed["command"] = "yts list"
    api_mod._persist_yts_live_state(completed)

    api_mod._yts_live_commands.clear()

    resp = await client.get("/yts/command/live?active_only=true&limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["command_id"] == "cmd-active"
    assert data[0]["status"] == "running"


def test_mark_stale_yts_live_commands_normalizes_legacy_persisted_state():
    conn = api_mod._ensure_yts_live_db()
    command_id = "cmd-legacy-stale"
    created_at = api_mod._utc_now_iso()
    legacy_state = {
        "command_id": command_id,
        "command": "yts test adb:device-01",
        "status": "running",
        "stdout": "",
        "stderr": "",
        "interactive_ai": False,
        "created_at": created_at,
        "updated_at": created_at,
    }

    with api_mod._yts_live_db_lock:
        conn.execute(
            """
            INSERT INTO yts_live_commands (
                command_id, status, command_text, interactive_ai, created_at, updated_at, state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command_id,
                "running",
                legacy_state["command"],
                0,
                created_at,
                created_at,
                json.dumps(legacy_state),
            ),
        )
        conn.commit()

    api_mod._mark_stale_yts_live_commands()

    restored = api_mod._load_yts_live_state(command_id)
    assert restored is not None
    assert restored["status"] == "failed"
    assert restored["awaiting_input"] is False
    assert isinstance(restored["logs"], list)
    assert any("Server restarted while this YTS command was running" in entry["message"] for entry in restored["logs"])


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
async def test_manual_action_set_setting(client):
    resp = await client.post(
        "/action",
        json={"action": "SET_SETTING", "params": {"key": "timezone", "value": "UTC"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["result"]["key"] == "timezone"
    assert data["result"]["value"] == "UTC"


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
async def test_task_macro_plan_time_zone_setting(client):
    resp = await client.post(
        "/task/macro",
        json={"instruction": "change time zone to UTC"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["planned_count"] == 1
    assert data["planned_actions"][0]["action"] == "SET_SETTING"
    assert data["planned_actions"][0]["params"]["key"] == "timezone"
    assert data["planned_actions"][0]["params"]["value"] == "UTC"


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
    state.latest_visual_summary = "Pause Up next"
    state.finish(RunStatus.TIMEOUT, "Max steps exceeded")
    _runs[run_id] = state

    resp = await client.get(f"/run/{run_id}/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["diagnosis"]["root_cause"] == "Repeated commit action in playback context"
    assert "kept pressing ok" in data["diagnosis"]["user_friendly_reason"].lower()


@pytest.mark.asyncio
async def test_get_run_explain_does_not_crash_when_diagnosis_builder_fails(client, monkeypatch):
    run_id = "r-explain-fallback"
    state = RunState(run_id=run_id, goal="Open settings")
    state.start()
    state.finish(RunStatus.FAILED, "forced")
    _runs[run_id] = state

    def _boom(_state):
        raise AttributeError("legacy field missing")

    monkeypatch.setattr(api_mod, "_build_final_diagnosis", _boom)

    resp = await client.get(f"/run/{run_id}/explain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert data["diagnosis"]["root_cause"] == "Insufficient diagnosis data"
    assert "fallback" in data["diagnosis"]["final_summary"].lower()


def test_build_final_diagnosis_accepts_missing_legacy_ocr_field():
    class LegacyState:
        def __init__(self):
            self.status = RunStatus.FAILED
            self.goal = "Open YouTube"
            self.action_history = []
            self.ai_transcript = []
            self.dab_transcript = []
            self.current_screen = None
            self.current_app_state = None
            self.current_app = None
            self.current_app_id = None
            self.latest_visual_summary = None

    diagnosis = api_mod._build_final_diagnosis(LegacyState())
    assert diagnosis is not None
    assert diagnosis.status == RunStatus.FAILED.value


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
    first = await client.post("/run/start", json={"goal": "Run A"})
    first_run_id = first.json()["run_id"]
    api_mod._runs[first_run_id].finish(RunStatus.DONE)

    second = await client.post("/run/start", json={"goal": "Run B"})
    assert second.status_code == 200

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


@pytest.mark.asyncio
async def test_start_run_rejects_when_another_run_is_active(client):
    active = RunState(run_id="active-run", goal="Already running")
    active.start()
    api_mod._runs[active.run_id] = active

    resp = await client.post("/run/start", json={"goal": "New run"})
    assert resp.status_code == 409
    assert "active-run" in str(resp.json().get("detail", ""))
