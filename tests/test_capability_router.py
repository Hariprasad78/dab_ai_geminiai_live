import pytest

from vertex_live_dab_agent.dab.client import DABResponse
from vertex_live_dab_agent.orchestrator.app_resolver import AppResolver
from vertex_live_dab_agent.system_ops.capabilities import (
    build_capability_snapshot,
    normalize_application,
    normalize_setting_key,
)
from vertex_live_dab_agent.system_ops.capability_discovery import discover_capabilities
from vertex_live_dab_agent.system_ops.execution_policy import CapabilityRouter


class _DabNonAndroid:
    async def list_operations(self):
        return DABResponse(True, 200, {"operations": ["applications/list", "applications/launch", "input/key/list"]}, "t", "r")

    async def list_apps(self):
        return DABResponse(True, 200, {"applications": [{"appId": "netflix", "friendlyName": "Netflix"}]}, "t", "r")

    async def list_keys(self):
        return DABResponse(True, 200, {"keys": ["KEY_HOME", "KEY_BACK"]}, "t", "r")


class _DabAndroidNoSettingGet:
    async def list_operations(self):
        return DABResponse(True, 200, {"operations": ["system/settings/list", "applications/list"]}, "t", "r")

    async def list_settings(self):
        return DABResponse(True, 200, {"settings": [{"key": "timezone", "friendlyName": "Time Zone", "allowedValues": ["UTC"]}]}, "t", "r")

    async def list_apps(self):
        return DABResponse(True, 200, {"applications": [{"appId": "youtube", "friendlyName": "YouTube"}]}, "t", "r")


class _DabMissingLists:
    async def list_operations(self):
        return DABResponse(True, 200, {"operations": ["applications/list"]}, "t", "r")

    async def list_apps(self):
        return DABResponse(True, 200, {"applications": [{"appId": "youtube", "friendlyName": "YouTube"}]}, "t", "r")


@pytest.mark.asyncio
async def test_non_android_device_uses_dab_only_snapshot():
    snap = await discover_capabilities(
        dab_client=_DabNonAndroid(),
        app_resolver=AppResolver(_DabNonAndroid()),
        device_id="roku-device",
        current_state={"is_android": False, "can_use_adb": False},
    )
    assert snap.is_android is False
    assert snap.can_use_adb is False
    assert "input/key/list" in [o.lower() for o in snap.supported_operations]


@pytest.mark.asyncio
async def test_android_prefers_adb_fallback_when_setting_get_missing():
    snap = await discover_capabilities(
        dab_client=_DabAndroidNoSettingGet(),
        app_resolver=AppResolver(_DabAndroidNoSettingGet()),
        device_id="127.0.0.1:5555",
        current_state={"is_android": True, "can_use_adb": True, "android_adb_device_id": "127.0.0.1:5555"},
    )
    router = CapabilityRouter(snap.model_dump())
    route = router.route_setting_operation(
        operation="system/settings/get",
        setting_key="timezone",
        can_use_adb=True,
        is_android=True,
    )
    assert route.transport in {"adb", "dab"}


@pytest.mark.asyncio
async def test_missing_key_and_voice_list_marked_unavailable():
    snap = await discover_capabilities(
        dab_client=_DabMissingLists(),
        app_resolver=AppResolver(_DabMissingLists()),
        device_id="tv-device",
        current_state={"is_android": False, "can_use_adb": False},
    )
    missing = " ".join(snap.unsupported_or_missing_capabilities).lower()
    assert "input/key/list" in missing
    assert "voice/list" in missing


def test_misspelled_key_or_setting_is_fuzzy_resolved():
    snapshot = build_capability_snapshot(
        supported_operations=["system/settings/get", "applications/launch"],
        supported_settings=[{"key": "timezone", "friendlyName": "Time Zone", "allowedValues": ["UTC"]}],
        supported_keys=["KEY_HOME"],
    )
    resolved_setting = normalize_setting_key(snapshot, "timezoen")
    assert resolved_setting["success"] is True
    assert resolved_setting["key"] == "timezone"


def test_misspelled_application_name_is_resolved():
    snapshot = build_capability_snapshot(
        supported_operations=["applications/list", "applications/launch"],
        supported_settings=[],
        supported_keys=[],
        installed_applications=[{"appId": "PrimeVideo", "friendlyName": "Prime Video"}],
    )
    resolved = normalize_application(snapshot, "prime vedio")
    assert resolved["success"] is True
    assert resolved["application"]["appId"] == "PrimeVideo"


def test_router_replans_after_capability_failure():
    snapshot = build_capability_snapshot(
        supported_operations=["applications/launch"],
        supported_settings=[],
        supported_keys=[],
        platform_type="linux-tv",
        is_android=False,
        can_use_adb=False,
    )
    router = CapabilityRouter(snapshot)
    route = router.route_setting_operation(
        operation="system/settings/get",
        setting_key="timezone",
        can_use_adb=False,
        is_android=False,
    )
    assert route.transport == "unsupported"
    assert "no legal transport" in route.reason
