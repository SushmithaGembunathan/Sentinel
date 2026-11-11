from __future__ import annotations

import secrets

from fastapi import Request


def ensure_csrf_token(request: Request) -> str:
    if "csrf_token" not in request.session:
        request.session["csrf_token"] = secrets.token_urlsafe(32)
    return request.session["csrf_token"]


def require_csrf(request: Request, csrf_token: str) -> None:
    expected = ensure_csrf_token(request)
    if csrf_token != expected:
        raise ValueError("Invalid CSRF token")
