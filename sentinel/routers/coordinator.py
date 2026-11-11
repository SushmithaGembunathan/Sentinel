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
from sentinel.models import Camp, CampUpdateHistory, User

router = APIRouter(prefix="/coordinator", tags=["coordinator"])
templates = Jinja2Templates(directory="sentinel/templates")


@router.get("/camp")
def camp_page(request: Request, db: Session = Depends(get_db), user: User = Depends(require_role("coordinator"))):
    camp = db.query(Camp).filter(Camp.camp_id == user.camp_id).first()
    if not camp:
        raise HTTPException(status_code=404, detail="No camp assigned")
    history = db.query(CampUpdateHistory).filter(CampUpdateHistory.camp_id == camp.camp_id).order_by(CampUpdateHistory.updated_at.desc()).all()
    return templates.TemplateResponse("coordinator_camp.html", {"request": request, "user": user, "camp": camp, "history": history})


@router.post("/camp/update")
def update_camp(
    request: Request,
    camp_id: str = Form(...),
    headcount: int = Form(...),
    elderly: int = Form(...),
    disabled: int = Form(...),
    maternal: int = Form(...),
    infant: int = Form(...),
    stock_food: float = Form(...),
    stock_water: float = Form(...),
    stock_medical: float = Form(...),
    stock_infant_supplies: float = Form(...),
    csrf_token: str = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("coordinator")),
):
    try:
        require_csrf(request, csrf_token)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    camp = db.query(Camp).filter(Camp.camp_id == camp_id).first()
    if not camp:
        raise HTTPException(status_code=404, detail="Camp not found")
    if user.camp_id != camp.camp_id:
        raise HTTPException(status_code=403, detail="You cannot edit another camp")
    if any(value < 0 for value in [headcount, elderly, disabled, maternal, infant]):
        raise HTTPException(status_code=400, detail="Counts cannot be negative")
    if elderly + disabled + maternal + infant > headcount:
        raise HTTPException(status_code=400, detail="Priority-group totals cannot exceed headcount")
    old_values = {
        "headcount": camp.headcount,
        "elderly": camp.elderly,
        "disabled": camp.disabled,
        "maternal": camp.maternal,
        "infant": camp.infant,
        "stock_food": camp.stock_food,
        "stock_water": camp.stock_water,
        "stock_medical": camp.stock_medical,
        "stock_infant_supplies": camp.stock_infant_supplies,
    }
    camp.headcount = int(headcount)
    camp.elderly = int(elderly)
    camp.disabled = int(disabled)
    camp.maternal = int(maternal)
    camp.infant = int(infant)
    camp.stock_food = float(stock_food)
    camp.stock_water = float(stock_water)
    camp.stock_medical = float(stock_medical)
    camp.stock_infant_supplies = float(stock_infant_supplies)
    camp.updated_by = user.username
    camp.updated_at = datetime.utcnow()
    db.add(CampUpdateHistory(camp_id=camp.camp_id, updated_by=user.username, updated_at=camp.updated_at, old_values=json.dumps(old_values), new_values=json.dumps({
        "headcount": camp.headcount,
        "elderly": camp.elderly,
        "disabled": camp.disabled,
        "maternal": camp.maternal,
        "infant": camp.infant,
        "stock_food": camp.stock_food,
        "stock_water": camp.stock_water,
        "stock_medical": camp.stock_medical,
        "stock_infant_supplies": camp.stock_infant_supplies,
    })))
    db.commit()
    log_audit(db, user.username, user.role, "update_camp", "camps", camp.camp_id, old_values, {"headcount": camp.headcount, "elderly": camp.elderly, "disabled": camp.disabled, "maternal": camp.maternal, "infant": camp.infant})
    return RedirectResponse(url="/coordinator/camp", status_code=302)
