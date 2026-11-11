from __future__ import annotations

import bcrypt


def hash_password(password: str, salt: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), salt.encode("utf-8")).decode("utf-8")


def make_salt() -> str:
    return bcrypt.gensalt().decode("utf-8")


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    return bcrypt.hashpw(password.encode("utf-8"), salt.encode("utf-8")) == password_hash.encode("utf-8")
