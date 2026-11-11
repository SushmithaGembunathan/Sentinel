from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'sentinel.db'}")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SESSION_COOKIE_NAME = "sentinel_session"
