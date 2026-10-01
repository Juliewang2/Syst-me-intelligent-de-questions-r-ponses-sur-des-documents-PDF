"""
services/auth_service.py
---------------------------
User authentication: password hashing, JWT access tokens, and the
FastAPI dependency that resolves the current user for a request.

- Passwords are never stored in plain text. They are hashed with
  PBKDF2-HMAC-SHA256 (standard library `hashlib`) using a random
  per-user salt and a high iteration count, and compared in constant
  time.
- After login the server issues a signed JWT (HS256, keyed with
  `SECRET_KEY`). The browser keeps it in an HttpOnly cookie (not
  readable by JavaScript, which limits XSS token theft); API clients
  can send it as `Authorization: Bearer <token>` instead.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db
from models import User

settings = get_settings()

ACCESS_TOKEN_COOKIE = "access_token"
_JWT_ALGORITHM = "HS256"
_PBKDF2_ITERATIONS = 390_000

# auto_error=False: a missing header is fine, we fall back to the cookie.
# Declaring it also adds an "Authorize" button to the Swagger UI.
_bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode(),
        base64.b64encode(digest).decode(),
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = stored_hash.split("$")
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), base64.b64decode(salt_b64), int(iterations)
    )
    return hmac.compare_digest(digest, base64.b64decode(digest_b64))


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """Return the user id carried by a valid token, or None."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def authenticate(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user


def _token_from_request(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials]
) -> Optional[str]:
    if credentials is not None and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    return request.cookies.get(ACCESS_TOKEN_COOKIE)


def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    token = _token_from_request(request, credentials)
    if not token:
        return None
    user_id = decode_access_token(token)
    if not user_id:
        return None
    return db.get(User, user_id)


def get_current_user(user: Optional[User] = Depends(get_optional_user)) -> User:
    """FastAPI dependency: the logged-in user, or 401."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
