from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from sentinel.csrf import require_csrf
from sentinel.database import get_db
from sentinel.deps import log_audit, require_role
from sentinel.models import AuditLog, Disbursal, FraudFlag, User
from relief_framework.integrity import detect_duplicate_payments, detect_missing_assessment, detect_zscore_outliers

router = APIRouter(prefix="/auditor", tags=["auditor"])
templates = Jinja2Templates(directory="sentinel/templates")


@router.get("/flags")
def flags_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("auditor"))):
    flags = db.query(FraudFlag).order_by(FraudFlag.created_at.desc()).all()
    return templates.TemplateResponse("auditor_flags.html", {"request": request, "user": user, "flags": flags})


@router.post("/flags/review")
def review_flag(
    request: Request,
    flag_id: int = Form(...),
    status: str = Form(...),
    note: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("auditor")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    if not note.strip():
        raise HTTPException(status_code=400, detail="A note is required for the review")
    flag = db.query(FraudFlag).filter(FraudFlag.id == flag_id).first()
    if not flag:
        raise HTTPException(status_code=404, detail="Flag not found")
    flag.status = status
    flag.note = note
    flag.reviewed_by = user.username
    flag.reviewed_at = datetime.utcnow()
    db.commit()
    log_audit(db, user.username, user.role, "review_flag", "fraud_flags", str(flag_id), None, {"status": status, "note": note})
    return RedirectResponse(url="/auditor/flags", status_code=302)


@router.get("/audit-log")
def audit_log(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("auditor"))):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(100).all()
    return templates.TemplateResponse("auditor_log.html", {"request": request, "user": user, "logs": logs})
