from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from sentinel.csrf import require_csrf
from sentinel.database import get_db
from sentinel.deps import log_audit, require_role
from sentinel.models import AuditLog, Camp, ResourceType, ScoringSetting, User
from sentinel.security import hash_password, make_salt

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="sentinel/templates")


@router.get("/users")
def users_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    user_rows = db.query(User).all()
    return templates.TemplateResponse("admin_users.html", {"request": request, "user": user, "users": user_rows})


@router.post("/users")
def create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    camp_id: str | None = Form(None),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    salt = make_salt()
    db.add(User(username=username, password_hash=hash_password(password, salt), salt=salt, role=role, camp_id=camp_id or None))
    db.commit()
    log_audit(db, user.username, user.role, "create_user", "users", username, None, {"role": role, "camp_id": camp_id})
    return RedirectResponse(url="/admin/users", status_code=302)


@router.get("/camps")
def camps_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    camp_rows = db.query(Camp).all()
    coordinators = db.query(User).filter(User.role == "coordinator").all()
    return templates.TemplateResponse("admin_camps.html", {"request": request, "user": user, "camps": camp_rows, "coordinators": coordinators})


@router.post("/camps")
def create_camp(
    request: Request,
    camp_id: str = Form(...),
    name: str = Form(...),
    district: str = Form(...),
    location: str = Form(...),
    headcount: int = Form(...),
    elderly: int = Form(...),
    disabled: int = Form(...),
    maternal: int = Form(...),
    infant: int = Form(...),
    assigned_coordinator_id: int | None = Form(None),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    if elderly + disabled + maternal + infant > headcount:
        raise HTTPException(status_code=400, detail="Priority categories cannot exceed headcount")
    if db.query(Camp).filter(Camp.camp_id == camp_id).first():
        raise HTTPException(status_code=400, detail="Camp ID already exists")
    camp = Camp(
        camp_id=camp_id,
        name=name,
        district=district,
        location=location,
        headcount=headcount,
        elderly=elderly,
        disabled=disabled,
        maternal=maternal,
        infant=infant,
        stock_food=0.0,
        stock_water=0.0,
        stock_medical=0.0,
        stock_infant_supplies=0.0,
        assigned_coordinator_id=assigned_coordinator_id,
        updated_by=user.username,
    )
    db.add(camp)
    db.commit()
    log_audit(db, user.username, user.role, "create_camp", "camps", camp_id, None, {"name": name, "district": district, "location": location})
    return RedirectResponse(url="/admin/camps", status_code=302)


@router.post("/camps/import")
def import_camps(request: Request, file: bytes = Form(...), csrf_token: str = Form(...), db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    text = file.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    errors = []
    created = 0
    for row in reader:
        try:
            camp_id = (row.get("camp_id") or "").strip()
            name = (row.get("name") or "").strip()
            if not camp_id or not name:
                errors.append(f"Missing camp_id or name on row {row}")
                continue
            if db.query(Camp).filter(Camp.camp_id == camp_id).first():
                errors.append(f"Duplicate camp_id {camp_id}")
                continue
            camp = Camp(
                camp_id=camp_id,
                name=name,
                district=(row.get("district") or "").strip(),
                location=(row.get("location") or "").strip(),
                headcount=int(row.get("headcount") or 0),
                elderly=int(row.get("elderly") or 0),
                disabled=int(row.get("disabled") or 0),
                maternal=int(row.get("maternal") or 0),
                infant=int(row.get("infant") or 0),
            )
            db.add(camp)
            created += 1
        except Exception as exc:  # pragma: no cover
            errors.append(f"Row import error: {exc}")
    db.commit()
    log_audit(db, user.username, user.role, "import_camps", "camps", None, None, {"created": created, "errors": errors})
    return RedirectResponse(url="/admin/camps", status_code=302)


@router.get("/resources")
def resources_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    rows = db.query(ResourceType).all()
    return templates.TemplateResponse("admin_resources.html", {"request": request, "user": user, "resources": rows})


@router.post("/resources")
def create_resource(
    request: Request,
    name: str = Form(...),
    unit: str = Form(...),
    nu0: float = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    if db.query(ResourceType).filter(ResourceType.name == name).first():
        raise HTTPException(status_code=400, detail="Resource type already exists")
    db.add(ResourceType(name=name, unit=unit, nu0=nu0))
    db.commit()
    log_audit(db, user.username, user.role, "create_resource", "resources", name, None, {"unit": unit, "nu0": nu0})
    return RedirectResponse(url="/admin/resources", status_code=302)


@router.get("/scoring")
def scoring_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("admin"))):
    settings = db.query(ScoringSetting).order_by(ScoringSetting.id.desc()).first()
    return templates.TemplateResponse("admin_scoring.html", {"request": request, "user": user, "settings": settings})


@router.post("/scoring")
def save_scoring(
    request: Request,
    phi: float = Form(...),
    horizon_days: int = Form(...),
    weights_json: str = Form(...),
    resources_json: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    version = (db.query(ScoringSetting).count() or 0) + 1
    db.add(ScoringSetting(version=version, phi=phi, horizon_days=horizon_days, weights_json=weights_json, resources_json=resources_json, created_by=user.username))
    db.commit()
    log_audit(db, user.username, user.role, "update_scoring", "settings", f"v{version}", None, {"phi": phi, "horizon_days": horizon_days})
    return RedirectResponse(url="/admin/scoring", status_code=302)
