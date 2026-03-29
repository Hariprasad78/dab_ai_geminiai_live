"""Tests for device profiles, trajectory memory, and hybrid policy wiring."""

from pathlib import Path

from vertex_live_dab_agent.hybrid import DeviceProfileRegistry, ExperienceQuery, HybridPolicyEngine, TrajectoryMemory
from vertex_live_dab_agent.orchestrator.orchestrator import Orchestrator
from vertex_live_dab_agent.orchestrator.run_state import RunState


def test_device_profile_registry_derives_features(tmp_path: Path) -> None:
    registry = DeviceProfileRegistry(tmp_path / "profiles")
    profile = registry.upsert_from_capabilities(
        device_id="adt-4",
        supported_operations=[
            "applications/launch",
            "system/settings/get",
            "system/settings/set",
            "output/image",
        ],
        supported_keys=["KEY_UP", "KEY_DOWN", "KEY_LEFT", "KEY_RIGHT", "KEY_ENTER"],
        supported_settings=[{"key": "timezone"}, {"key": "language"}],
        app_catalog=[{"appId": "youtube", "friendlyName": "YouTube"}],
    )
    assert profile.features["direct_app_launch"] is True
    assert profile.features["direct_setting_set"] is True
    assert profile.features["timezone_setting_known"] is True
    assert registry.load("adt-4") is not None


def test_trajectory_memory_retrieves_similar_entries(tmp_path: Path) -> None:
    memory = TrajectoryMemory(tmp_path / "experience" / "trajectories.jsonl")
    memory.append(
        {
            "goal": "change time zone to Colombo",
            "device_id": "adt-4",
            "current_app": "settings",
            "action": "SET_SETTING",
            "result": "PASS",
            "strategy_selected": "DIRECT_SETTING_OPERATION",
        }
    )
    memory.append(
        {
            "goal": "open YouTube",
            "device_id": "adt-4",
            "current_app": "launcher",
            "action": "LAUNCH_APP",
            "result": "PASS",
        }
    )
    results = memory.find_similar(
        ExperienceQuery(goal="set timezone", device_id="adt-4", current_app="settings", limit=3)
    )
    assert results
    assert results[0]["action"] == "SET_SETTING"


def test_hybrid_policy_prefers_direct_settings() -> None:
    engine = HybridPolicyEngine()
    recommendation = engine.recommend(
        goal="change language to English",
        device_profile={
            "profile_id": "profile:adt-4",
            "features": {"direct_setting_set": True},
        },
        similar_experiences=[],
    )
    assert recommendation.mode == "DIRECT_DAB_PREFERRED"


def test_orchestrator_refreshes_hybrid_context(tmp_path: Path) -> None:
    orch = object.__new__(Orchestrator)
    orch._config = type(
        "_Cfg",
        (),
        {
            "dab_device_id": "adt-4",
            "hybrid_policy_mode": "auto",
        },
    )()
    orch._device_profiles = DeviceProfileRegistry(tmp_path / "profiles")
    orch._trajectory_memory = TrajectoryMemory(tmp_path / "experience" / "trajectories.jsonl")
    orch._hybrid_policy = HybridPolicyEngine()

    orch._device_profiles.upsert_from_capabilities(
        device_id="adt-4",
        supported_operations=["applications/launch"],
        supported_keys=[],
        supported_settings=[],
        app_catalog=[],
    )
    orch._trajectory_memory.append(
        {
            "goal": "open netflix",
            "device_id": "adt-4",
            "current_app": "launcher",
            "action": "LAUNCH_APP",
            "result": "PASS",
        }
    )
    state = RunState(goal="open netflix")
    orch._refresh_hybrid_context(state)
    assert state.device_profile_id == "profile:adt-4"
    assert state.hybrid_policy_mode in {"LOCAL_MEMORY_ASSISTED", "HYBRID_DIRECT_FIRST"}
    assert state.retrieved_experiences
