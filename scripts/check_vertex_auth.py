#!/usr/bin/env python3
"""Simple Vertex API auth + model call checker.

What it checks:
1) GOOGLE_APPLICATION_CREDENTIALS file exists and is readable JSON
2) ADC can be loaded and token can be refreshed
3) Vertex model call works with generate_content()

Usage:
  python scripts/check_vertex_auth.py \
    --project your-project \
    --location asia-south1 \
        --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import google.auth
from google.auth.transport.requests import Request


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str, code: int = 1) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(code)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Vertex auth and simple model call")
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1"))
    parser.add_argument(
        "--model",
        default=os.getenv("VERTEX_TEST_MODEL") or "gemini-2.5-flash",
    )
    parser.add_argument(
        "--prompt",
        default='Reply with exactly: {"ok": true}',
        help="Small text prompt used for probe call",
    )
    return parser.parse_args()


def check_auth_file() -> None:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not cred_path:
        _warn("GOOGLE_APPLICATION_CREDENTIALS is not set (ADC may still work)")
        return

    p = Path(cred_path)
    if not p.exists():
        _fail(f"Credential file does not exist: {p}")
    if not p.is_file():
        _fail(f"Credential path is not a file: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        _fail(f"Credential file is not valid JSON: {exc}")

    email = data.get("client_email")
    project_id = data.get("project_id")
    _ok(f"Credential file readable: {p}")
    if email:
        _ok(f"Service account email: {email}")
    if project_id:
        _ok(f"Credential project_id: {project_id}")


def check_adc_and_token() -> tuple[object, str | None]:
    try:
        credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as exc:
        _fail(f"Failed to load ADC credentials: {exc}")

    try:
        credentials.refresh(Request())
    except Exception as exc:
        _fail(f"Failed to refresh access token: {exc}")

    token_preview = getattr(credentials, "token", None)
    if token_preview:
        _ok("ADC token refresh successful")
    else:
        _warn("ADC loaded but token is empty")

    qp = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if qp:
        _ok(f"ADC project: {qp}")
    else:
        _warn("No ADC project detected")

    return credentials, project_id


def check_vertex_call(project: str, location: str, model: str, prompt: str) -> None:
    if not project:
        _fail("Missing project. Set --project or GOOGLE_CLOUD_PROJECT")
    if not location:
        _fail("Missing location. Set --location or GOOGLE_CLOUD_LOCATION")

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception as exc:
        _fail(f"vertexai SDK import failed: {exc}. Install: pip install -e '.[vertex]'")

    try:
        vertexai.init(project=project, location=location)
        model_client = GenerativeModel(model)
        response = model_client.generate_content(prompt)
    except Exception as exc:
        _fail(f"Vertex call failed for model '{model}' in {location}: {exc}")

    text = getattr(response, "text", "") or ""
    if text.strip():
        _ok(f"Vertex call succeeded. Response preview: {text[:200]!r}")
    else:
        _ok("Vertex call succeeded (non-text response received)")


def main() -> None:
    args = parse_args()
    print("=== Vertex Auth + API Check ===")
    check_auth_file()
    _, adc_project = check_adc_and_token()

    project = args.project or adc_project or ""
    check_vertex_call(
        project=project,
        location=args.location,
        model=args.model,
        prompt=args.prompt,
    )
    print("=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)
