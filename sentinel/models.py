from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    salt: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    camp_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("camps.camp_id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Camp(Base):
    __tablename__ = "camps"

    camp_id: Mapped[str] = mapped_column(String(50), primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    district: Mapped[str] = mapped_column(String(200), default="")
    location: Mapped[str] = mapped_column(String(200), default="")
    headcount: Mapped[int] = mapped_column(Integer, default=0)
    elderly: Mapped[int] = mapped_column(Integer, default=0)
    disabled: Mapped[int] = mapped_column(Integer, default=0)
    maternal: Mapped[int] = mapped_column(Integer, default=0)
    infant: Mapped[int] = mapped_column(Integer, default=0)
    stock_food: Mapped[float] = mapped_column(Float, default=0.0)
    stock_water: Mapped[float] = mapped_column(Float, default=0.0)
    stock_medical: Mapped[float] = mapped_column(Float, default=0.0)
    stock_infant_supplies: Mapped[float] = mapped_column(Float, default=0.0)
    assigned_coordinator_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class ResourceType(Base):
    __tablename__ = "resource_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    unit: Mapped[str] = mapped_column(String(50), default="kg")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    nu0: Mapped[float] = mapped_column(Float, default=1.0)
    nu_json: Mapped[str] = mapped_column(Text, default='{"base": 0.0, "elderly": 0.0, "disabled": 0.0, "maternal": 0.0, "infant": 0.0}')
    rho_json: Mapped[str] = mapped_column(Text, default='{"base": 1.0, "elderly": 1.0, "disabled": 1.0, "maternal": 1.0, "infant": 1.0}')
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ScoringSetting(Base):
    __tablename__ = "scoring_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    phi: Mapped[float] = mapped_column(Float, default=0.20)
    horizon_days: Mapped[int] = mapped_column(Integer, default=3)
    weights_json: Mapped[str] = mapped_column(Text, default='{"base": 1.0, "elderly": 1.5, "disabled": 2.0, "maternal": 2.0, "infant": 1.5}')
    resources_json: Mapped[str] = mapped_column(Text, default='{}')
    created_by: Mapped[str] = mapped_column(String(120), default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SupplyBatch(Base):
    __tablename__ = "supply_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    resource: Mapped[str] = mapped_column(String(120), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AllocationRecommendation(Base):
    __tablename__ = "allocation_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    resource: Mapped[str] = mapped_column(String(120), nullable=False)
    camp_id: Mapped[str] = mapped_column(String(50), ForeignKey("camps.camp_id"), nullable=False)
    supply_batch_id: Mapped[int] = mapped_column(ForeignKey("supply_batches.id"), nullable=True)
    settings_version: Mapped[int] = mapped_column(Integer, default=1)
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)
    demand: Mapped[float] = mapped_column(Float, default=0.0)
    recommended_qty: Mapped[float] = mapped_column(Float, default=0.0)
    coverage: Mapped[float] = mapped_column(Float, default=0.0)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[str] = mapped_column(Text, default="")
    headcount_only_qty: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DispatchOrder(Base):
    __tablename__ = "dispatch_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camp_id: Mapped[str] = mapped_column(String(50), ForeignKey("camps.camp_id"), nullable=False)
    resource: Mapped[str] = mapped_column(String(120), nullable=False)
    approved_qty: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(50), default="approved")
    note: Mapped[str] = mapped_column(Text, default="")
    manual_override: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    received_qty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Disbursal(Base):
    __tablename__ = "disbursals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    beneficiary_id: Mapped[str] = mapped_column(String(120), nullable=False)
    camp_id: Mapped[str] = mapped_column(String(50), ForeignKey("camps.camp_id"), nullable=False)
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    damage_category: Mapped[str] = mapped_column(String(120), nullable=False)
    assessment_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class FraudFlag(Base):
    __tablename__ = "fraud_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    disbursal_id: Mapped[int] = mapped_column(ForeignKey("disbursals.id"), nullable=False)
    payment_id: Mapped[str] = mapped_column(String(120), nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(50), default="medium")
    status: Mapped[str] = mapped_column(String(50), default="open")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class CampUpdateHistory(Base):
    __tablename__ = "camp_update_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camp_id: Mapped[str] = mapped_column(String(50), ForeignKey("camps.camp_id"), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(120), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    old_values: Mapped[str] = mapped_column(Text, default="{}")
    new_values: Mapped[str] = mapped_column(Text, default="{}")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    username: Mapped[str] = mapped_column(String(120), default="system")
    role: Mapped[str] = mapped_column(String(50), default="system")
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity: Mapped[str] = mapped_column(String(120), default="")
    entity_id: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
