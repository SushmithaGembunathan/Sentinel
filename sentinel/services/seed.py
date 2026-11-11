from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy.orm import Session

from relief_framework import config
from sentinel.models import AuditLog, Camp, ResourceType, ScoringSetting, User
from sentinel.security import hash_password, make_salt


def create_default_resource_types(db: Session) -> None:
    for name, spec in config.RESOURCE_SPECS.items():
        existing = db.query(ResourceType).filter(ResourceType.name == name).first()
        if existing is None:
            db.add(ResourceType(
                name=name,
                unit="units",
                nu0=spec.nu0,
                nu_json=json.dumps(spec.nu),
                rho_json=json.dumps(spec.rho),
            ))
    db.commit()


def create_default_settings(db: Session) -> None:
    existing = db.query(ScoringSetting).first()
    if existing is None:
        db.add(ScoringSetting(
            version=1,
            phi=config.PHI,
            horizon_days=config.HORIZON_DAYS,
            weights_json=json.dumps(config.GROUP_WEIGHTS),
            resources_json=json.dumps({name: {"nu0": spec.nu0, "nu": spec.nu, "rho": spec.rho} for name, spec in config.RESOURCE_SPECS.items()}),
            created_by="system",
        ))
    db.commit()


def ensure_demo_users(db: Session) -> None:
    default_users = [
        ("admin", "admin123", "admin", None),
        ("officer1", "officer1", "officer", None),
        ("coordinator1", "coordinator1", "coordinator", "C001"),
        ("auditor1", "auditor1", "auditor", None),
    ]
    for username, password, role, camp_id in default_users:
        exists = db.query(User).filter(User.username == username).first()
        if exists:
            continue
        salt = make_salt()
        db.add(User(
            username=username,
            password_hash=hash_password(password, salt),
            salt=salt,
            role=role,
            camp_id=camp_id,
            is_active=True,
        ))
    db.commit()


def ensure_sample_camps(db: Session) -> None:
    if db.query(Camp).count() > 0:
        return
    for camp_id in ["C001", "C002", "C003"]:
        db.add(Camp(
            camp_id=camp_id,
            name=f"Camp {camp_id}",
            district="District 1",
            location=f"Location {camp_id}",
            headcount=180 if camp_id == "C001" else 240,
            elderly=40 if camp_id == "C001" else 50,
            disabled=25 if camp_id == "C001" else 30,
            maternal=18 if camp_id == "C001" else 20,
            infant=12 if camp_id == "C001" else 15,
            stock_food=55.0,
            stock_water=70.0,
            stock_medical=20.0,
            stock_infant_supplies=18.0,
            updated_by="system",
            updated_at=datetime.utcnow(),
            assigned_coordinator_id=db.query(User).filter(User.username == "coordinator1").first().id if camp_id == "C001" else None,
        ))
    db.commit()


def bootstrap(db: Session) -> None:
    create_default_settings(db)
    ensure_demo_users(db)
    ensure_sample_camps(db)
    create_default_resource_types(db)
    db.add(AuditLog(username="system", role="admin", action="bootstrap", entity="system", entity_id="startup"))
    db.commit()
