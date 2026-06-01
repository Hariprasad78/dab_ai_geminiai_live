"""Tests for Google operator session authentication."""
from dataclasses import replace
import time

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vertex_live_dab_agent.api.auth import (
    GoogleAuthSettings,
    GoogleSessionMiddleware,
    create_session_cookie,
    decode_session_cookie,
    user_is_allowed,
)


def _settings(**changes) -> GoogleAuthSettings:
    defaults = GoogleAuthSettings(
        enabled=True,
        client_id="test.apps.googleusercontent.com",
        session_secret="a" * 48,
        allowed_domains=set(),
        allowed_emails=set(),
        session_cookie="dab_operator_session",
        session_ttl_seconds=3600,
        secure_cookie=False,
        cookie_samesite="lax",
    )
    return replace(defaults, **changes)


def test_signed_session_cookie_round_trip_and_tamper_rejection():
    settings = _settings()
    cookie = create_session_cookie({"sub": "123", "email": "user@example.com", "name": "User"}, settings)
    assert decode_session_cookie(cookie, settings)["email"] == "user@example.com"
    assert decode_session_cookie(cookie + "x", settings) is None


def test_expired_session_cookie_is_rejected():
    settings = _settings(session_ttl_seconds=-1)
    cookie = create_session_cookie({"sub": "123", "email": "user@example.com"}, settings)
    time.sleep(0.01)
    assert decode_session_cookie(cookie, settings) is None


def test_google_account_allowlist_accepts_domain_or_email():
    settings = _settings(allowed_domains={"example.com"}, allowed_emails={"special@other.com"})
    assert user_is_allowed({"email": "user@example.com"}, settings)
    assert user_is_allowed({"email": "special@other.com"}, settings)
    assert not user_is_allowed({"email": "blocked@other.com"}, settings)


@pytest.mark.asyncio
async def test_auth_middleware_keeps_health_public_and_protects_api_routes():
    app = FastAPI()
    app.add_middleware(GoogleSessionMiddleware, settings=_settings())

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/private")
    async def private():
        return {"status": "ok"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/private")).status_code == 401


@pytest.mark.asyncio
async def test_auth_middleware_accepts_valid_session_cookie():
    settings = _settings()
    app = FastAPI()
    app.add_middleware(GoogleSessionMiddleware, settings=settings)

    @app.get("/private")
    async def private():
        return {"status": "ok"}

    cookie = create_session_cookie({"sub": "123", "email": "user@example.com"}, settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        client.cookies.set(settings.session_cookie, cookie)
        assert (await client.get("/private")).status_code == 200
