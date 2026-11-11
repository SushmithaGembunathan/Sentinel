from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from sentinel.csrf import ensure_csrf_token, require_csrf
from sentinel.database import get_db
from sentinel.models import AuditLog, User
from sentinel.security import verify_password

router = APIRouter()
templates = Jinja2Templates(directory="sentinel/templates")


@router.get("/")
def landing(request: Request):
    if request.session.get("username"):
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@router.get("/login")
def login_page(request: Request, db: Session = Depends(get_db)):
    ensure_csrf_token(request)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid security token."})
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash, user.salt):
        ensure_csrf_token(request)
        db.add(AuditLog(username=username or "unknown", role="unknown", action="failed_login", entity="users", entity_id=username))
        db.commit()
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password."})
    ensure_csrf_token(request)
    request.session["username"] = user.username
    request.session["role"] = user.role
    db.add(AuditLog(username=user.username, role=user.role, action="login", entity="users", entity_id=user.username, new_value=json.dumps({"status": "success"})))
    db.commit()
    return RedirectResponse(url="/dashboard", status_code=302)


@router.post("/logout")
def logout(request: Request, csrf_token: str = Form(...), db: Session = Depends(get_db)):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        return RedirectResponse(url="/login", status_code=302)
    username = request.session.get("username")
    role = request.session.get("role")
    if username:
        db.add(AuditLog(username=username, role=role or "unknown", action="logout", entity="users", entity_id=username, new_value=json.dumps({"status": "success"})))
        db.commit()
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@router.get("/dashboard")
def dashboard(request: Request, db: Session = Depends(get_db)):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=302)
    user = db.query(User).filter(User.username == username).first()
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})
