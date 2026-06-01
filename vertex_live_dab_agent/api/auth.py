"""Google Identity Services authentication for the operator dashboard."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException, Request, WebSocket
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


def _env_bool(name: str, default: bool) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> set[str]:
    return {item.strip().lower() for item in str(os.environ.get(name, "") or "").split(",") if item.strip()}


@dataclass(frozen=True)
class GoogleAuthSettings:
    enabled: bool
    client_id: str
    session_secret: str
    allowed_domains: set[str]
    allowed_emails: set[str]
    session_cookie: str
    session_ttl_seconds: int
    secure_cookie: bool
    cookie_samesite: str

    @classmethod
    def from_env(cls) -> "GoogleAuthSettings":
        return cls(
            enabled=_env_bool("GOOGLE_AUTH_ENABLED", False),
            client_id=str(os.environ.get("GOOGLE_AUTH_CLIENT_ID", "") or "").strip(),
            session_secret=str(os.environ.get("SESSION_SECRET", "") or "").strip(),
            allowed_domains=_env_csv("GOOGLE_AUTH_ALLOWED_DOMAINS"),
            allowed_emails=_env_csv("GOOGLE_AUTH_ALLOWED_EMAILS"),
            session_cookie=str(os.environ.get("GOOGLE_AUTH_SESSION_COOKIE", "dab_operator_session") or "").strip(),
            session_ttl_seconds=max(300, int(os.environ.get("GOOGLE_AUTH_SESSION_TTL_SECONDS", "43200"))),
            secure_cookie=_env_bool("GOOGLE_AUTH_SECURE_COOKIE", True),
            cookie_samesite=str(os.environ.get("GOOGLE_AUTH_COOKIE_SAMESITE", "lax") or "").strip().lower(),
        )

    def setup_error(self) -> str:
        if not self.enabled:
            return ""
        if not self.client_id:
            return "GOOGLE_AUTH_CLIENT_ID is required when GOOGLE_AUTH_ENABLED=true"
        if len(self.session_secret) < 32:
            return "SESSION_SECRET must contain at least 32 characters when GOOGLE_AUTH_ENABLED=true"
        if self.cookie_samesite not in {"lax", "strict", "none"}:
            return "GOOGLE_AUTH_COOKIE_SAMESITE must be lax, strict, or none"
        if self.cookie_samesite == "none" and not self.secure_cookie:
            return "GOOGLE_AUTH_SECURE_COOKIE must be true when GOOGLE_AUTH_COOKIE_SAMESITE=none"
        return ""


_PUBLIC_PATHS = {
    "/",
    "/app.js",
    "/auth.js",
    "/config.js",
    "/styles.css",
    "/premium.css",
    "/health",
    "/auth/config",
    "/auth/google",
}
def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _signature(payload: str, secret: str) -> str:
    return _b64_encode(hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())


def create_session_cookie(user: dict[str, Any], settings: GoogleAuthSettings) -> str:
    payload = {
        "sub": str(user["sub"]),
        "email": str(user.get("email") or ""),
        "name": str(user.get("name") or ""),
        "picture": str(user.get("picture") or ""),
        "hd": str(user.get("hd") or ""),
        "exp": int(time.time()) + settings.session_ttl_seconds,
    }
    encoded = _b64_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{encoded}.{_signature(encoded, settings.session_secret)}"


def decode_session_cookie(value: str, settings: GoogleAuthSettings) -> Optional[dict[str, Any]]:
    try:
        payload, signature = value.rsplit(".", 1)
        if not hmac.compare_digest(signature, _signature(payload, settings.session_secret)):
            return None
        user = json.loads(_b64_decode(payload))
        if not isinstance(user, dict) or not user.get("sub") or int(user.get("exp") or 0) < int(time.time()):
            return None
        return user
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def user_is_allowed(user: dict[str, Any], settings: GoogleAuthSettings) -> bool:
    email = str(user.get("email") or "").strip().lower()
    domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    if not settings.allowed_domains and not settings.allowed_emails:
        return True
    return email in settings.allowed_emails or domain in settings.allowed_domains


def verify_google_credential(credential: str, settings: GoogleAuthSettings) -> dict[str, Any]:
    if settings.setup_error():
        raise HTTPException(status_code=503, detail=settings.setup_error())
    try:
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2 import id_token

        user = id_token.verify_oauth2_token(credential, GoogleRequest(), settings.client_id)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Google sign-in verification failed: {exc}") from exc
    if not user.get("sub") or not user.get("email") or not user.get("email_verified"):
        raise HTTPException(status_code=403, detail="Google account email is not verified")
    if not user_is_allowed(user, settings):
        raise HTTPException(status_code=403, detail="This Google account is not allowed to use the DAB console")
    return user


def request_user(request: Request, settings: GoogleAuthSettings) -> Optional[dict[str, Any]]:
    if not settings.enabled:
        return {"sub": "auth-disabled", "email": "", "name": "Local operator", "auth_disabled": True}
    return decode_session_cookie(request.cookies.get(settings.session_cookie, ""), settings)


def websocket_user(websocket: WebSocket, settings: GoogleAuthSettings) -> Optional[dict[str, Any]]:
    if not settings.enabled:
        return {"sub": "auth-disabled", "email": "", "name": "Local operator", "auth_disabled": True}
    return decode_session_cookie(websocket.cookies.get(settings.session_cookie, ""), settings)


class GoogleSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: GoogleAuthSettings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        if not self.settings.enabled or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        setup_error = self.settings.setup_error()
        if setup_error:
            return JSONResponse({"detail": setup_error}, status_code=503)
        if request_user(request, self.settings) is None:
            return JSONResponse({"detail": "Google sign-in required"}, status_code=401)
        return await call_next(request)
