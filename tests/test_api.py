"""Tests for the FastAPI backend."""
import pytest
from httpx import ASGITransport, AsyncClient

from vertex_live_dab_agent.api.api import app


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
    assert "vertex_live_model" in data
    assert "max_steps_per_run" in data


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
        json={"action": "LAUNCH_APP", "params": {"app_id": "com.netflix.ninja"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_manual_action_launch_app_missing_app_id(client):
    resp = await client.post("/action", json={"action": "LAUNCH_APP"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_manual_action_unknown(client):
    resp = await client.post("/action", json={"action": "UNKNOWN_THING"})
    assert resp.status_code == 400


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
async def test_get_run_status_after_start(client):
    start_resp = await client.post("/run/start", json={"goal": "Test goal"})
    run_id = start_resp.json()["run_id"]

    status_resp = await client.get(f"/run/{run_id}/status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["run_id"] == run_id
    assert data["goal"] == "Test goal"


@pytest.mark.asyncio
async def test_screenshot_not_found_for_run(client):
    start_resp = await client.post("/run/start", json={"goal": "Test screenshot"})
    run_id = start_resp.json()["run_id"]
    # The run starts fresh with no screenshot before it runs
    # The screenshot may or may not be present depending on async timing
    # Just check it returns either 200 or 404
    resp = await client.get(f"/run/{run_id}/screenshot")
    assert resp.status_code in (200, 404)
