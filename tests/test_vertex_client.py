"""Focused tests for the Gemini Live planner client."""

from __future__ import annotations

import sys
import types

import pytest

from vertex_live_dab_agent.planner.vertex_client import VertexPlannerClient


class _Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_api_key_live_client_reuses_retained_session(monkeypatch):
    counters = {"connects": 0, "exits": 0}

    class FakeSession:
        async def send_realtime_input(self, **_kwargs):
            return None

        async def receive(self):
            yield types.SimpleNamespace(text='{"summary":"watching"}', server_content=None)

    class FakeContext:
        async def __aenter__(self):
            counters["connects"] += 1
            return FakeSession()

        async def __aexit__(self, *_args):
            counters["exits"] += 1

    class FakeLive:
        def connect(self, **_kwargs):
            return FakeContext()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.aio = types.SimpleNamespace(live=FakeLive())

    fake_types = types.SimpleNamespace(
        LiveConnectConfig=_Config,
        AudioTranscriptionConfig=_Config,
        SpeechConfig=_Config,
        VoiceConfig=_Config,
        PrebuiltVoiceConfig=_Config,
        GenerationConfig=_Config,
        Blob=_Config,
    )
    fake_genai = types.SimpleNamespace(Client=FakeClient, types=fake_types)
    import google

    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    client = VertexPlannerClient(project="", location="", model="gemini-live-test", api_key="test-key")
    first = await client.generate_content("frame one", session_id="monitor-1", keep_live_session=True)
    second = await client.generate_content("frame two", session_id="monitor-1", keep_live_session=True)

    assert first == '{"summary":"watching"}'
    assert second == '{"summary":"watching"}'
    assert counters["connects"] == 1
    assert counters["exits"] == 0

    await client.close_live_session("monitor-1")

    assert counters["exits"] == 1
