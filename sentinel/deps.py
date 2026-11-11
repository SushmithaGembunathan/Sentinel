from __future__ import annotations

import json
from typing import Iterable

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .database import SessionLocal, get_db
from .models import AuditLog, User


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    username = request.session.get("username")
    if not username:
        return None
    return db.query(User).filter(User.username == username, User.is_active.is_(True)).first()


def require_role(*roles: str):
    def dependency(request: Request, db: Session = Depends(get_db)):
        user = get_current_user(request, db)
        if not user:
            db.add(AuditLog(username="unknown", role="unknown", action="denied_access", entity="auth", entity_id="unknown"))
            db.commit()
            raise HTTPException(status_code=403, detail="Access denied")
        if user.role not in roles:
            db.add(AuditLog(username=user.username, role=user.role, action="denied_access", entity="route", entity_id=user.username))
            db.commit()
            raise HTTPException(status_code=403, detail="Access denied")
        return user

    return dependency


def log_audit(db: Session, username: str, role: str, action: str, entity: str, entity_id: str | None = None, old_value: dict | None = None, new_value: dict | None = None) -> None:
    db.add(AuditLog(username=username, role=role, action=action, entity=entity, entity_id=entity_id, old_value=json.dumps(old_value) if old_value else None, new_value=json.dumps(new_value) if new_value else None))
    db.commit()
