from __future__ import annotations

import argparse

from sentinel.database import SessionLocal, create_tables
from sentinel.models import User
from sentinel.security import hash_password, make_salt


def create_admin(username: str, password: str) -> None:
    create_tables()
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        salt = make_salt()
        db.add(User(username=username, password_hash=hash_password(password, salt), salt=salt, role="admin", is_active=True))
        db.commit()
        print(f"Created admin user: {username}")
    else:
        print(f"Admin user already exists: {username}")
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the first admin user.")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin123")
    args = parser.parse_args()
    create_admin(args.username, args.password)
