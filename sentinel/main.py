from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from sentinel.config import BASE_DIR, SECRET_KEY
from sentinel.csrf import require_csrf
from sentinel.database import SessionLocal, create_tables
from sentinel.models import User
from sentinel.routers import admin, auditor, auth, coordinator, officer
from sentinel.services.seed import bootstrap


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    db = SessionLocal()
    existing_usernames = {user.username for user in db.query(User).all()}
    required_usernames = {"admin", "coordinator1", "officer1", "auditor1"}
    if not existing_usernames or not required_usernames.issubset(existing_usernames):
        bootstrap(db)
    db.close()
    yield


app = FastAPI(title="Sentinel", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "sentinel" / "static")), name="static")
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(coordinator.router)
app.include_router(officer.router)
app.include_router(auditor.router)

templates = Jinja2Templates(directory=str(BASE_DIR / "sentinel" / "templates"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dashboard")
def dashboard_redirect(request: Request):
    if request.session.get("username"):
        user = request.session.get("role")
        if user == "admin":
            return RedirectResponse(url="/admin/users", status_code=302)
        if user == "coordinator":
            return RedirectResponse(url="/coordinator/camp", status_code=302)
        if user == "officer":
            return RedirectResponse(url="/officer/allocations", status_code=302)
        if user == "auditor":
            return RedirectResponse(url="/auditor/flags", status_code=302)
    return RedirectResponse(url="/login", status_code=302)
