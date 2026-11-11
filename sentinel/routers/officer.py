from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from relief_framework import config
from relief_framework.allocation import allocate_baseline, allocate_equity, demand, priority_scores
from relief_framework.metrics import min_coverage, vwc
from sentinel.csrf import require_csrf
from sentinel.database import get_db
from sentinel.deps import log_audit, require_role
from sentinel.models import AllocationRecommendation, Camp, DispatchOrder, SupplyBatch, User
from sentinel.services.allocation_service import run_allocation_for_resource

router = APIRouter(prefix="/officer", tags=["officer"])
templates = Jinja2Templates(directory="sentinel/templates")


@router.get("/allocations")
def allocations_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("officer"))):
    camps = db.query(Camp).all()
    return templates.TemplateResponse("officer_allocations.html", {"request": request, "user": user, "camps": camps, "resources": list(config.RESOURCE_SPECS.keys())})


@router.post("/allocations/run")
def run_allocations(
    request: Request,
    resource: str = Form(...),
    supply_pct: float = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("officer")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    camps = db.query(Camp).all()
    results = run_allocation_for_resource(camps, resource, float(supply_pct))
    batch = SupplyBatch(resource=resource, quantity=float(supply_pct), date=datetime.utcnow(), created_by=user.username)
    db.add(batch)
    db.commit()
    for row in results:
        rec = AllocationRecommendation(
            resource=resource,
            camp_id=row["camp_id"],
            supply_batch_id=batch.id,
            priority_score=row["priority_score"],
            demand=row["demand"],
            recommended_qty=row["recommended_qty"],
            coverage=row["coverage"],
            rank=row["rank"],
            reason=row["reason"],
            headcount_only_qty=row["baseline_qty"],
            settings_version=1,
        )
        db.add(rec)
    db.commit()
    log_audit(db, user.username, user.role, "run_allocation", "allocations", resource, None, {"supply_pct": supply_pct, "records": len(results)})
    return templates.TemplateResponse("officer_allocations.html", {"request": request, "user": user, "results": results, "resources": list(config.RESOURCE_SPECS.keys()), "camps": camps, "resource": resource, "supply_pct": supply_pct})


@router.get("/dispatches")
def dispatches_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("officer"))):
    orders = db.query(DispatchOrder).order_by(DispatchOrder.created_at.desc()).all()
    return templates.TemplateResponse("officer_dispatches.html", {"request": request, "user": user, "orders": orders})


@router.post("/dispatches/approve")
def approve_dispatch(
    request: Request,
    order_id: int = Form(...),
    note: str = Form(""),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("officer")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    order = db.query(DispatchOrder).filter(DispatchOrder.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Dispatch order not found")
    order.status = "approved"
    order.note = note
    order.approved_by = user.username
    order.approved_at = datetime.utcnow()
    db.commit()
    log_audit(db, user.username, user.role, "approve_dispatch", "dispatch_orders", str(order_id), None, {"quantity": order.approved_qty, "note": note})
    return RedirectResponse(url="/officer/dispatches", status_code=302)


@router.post("/dispatches/reject")
def reject_dispatch(
    request: Request,
    order_id: int = Form(...),
    note: str = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("officer")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    if not note.strip():
        raise HTTPException(status_code=400, detail="A note is required for rejection")
    order = db.query(DispatchOrder).filter(DispatchOrder.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Dispatch order not found")
    order.status = "rejected"
    order.note = note
    order.approved_by = user.username
    order.approved_at = datetime.utcnow()
    db.commit()
    log_audit(db, user.username, user.role, "reject_dispatch", "dispatch_orders", str(order_id), None, {"note": note})
    return RedirectResponse(url="/officer/dispatches", status_code=302)
