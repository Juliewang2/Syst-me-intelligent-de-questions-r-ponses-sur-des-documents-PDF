"""
routers/auth.py
-----------------
Account endpoints: register, log in, log out, and "who am I".

Logging in sets the JWT in an HttpOnly cookie for the web UI and also
returns it in the body, so API clients (or the Swagger "Authorize"
button) can use it as a Bearer token.
"""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db
from models import User
from schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from services.auth_service import (
    ACCESS_TOKEN_COOKIE,
    authenticate,
    create_access_token,
    get_current_user,
    hash_password,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])
settings = get_settings()


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        ACCESS_TOKEN_COOKIE,
        token,
        max_age=settings.access_token_expire_minutes * 60,
        httponly=True,  # not readable from JavaScript
        samesite="lax",  # not sent on cross-site POSTs (CSRF mitigation)
        secure=settings.app_env == "production",  # HTTPS-only in production
    )


def _login_response(response: Response, user: User) -> TokenResponse:
    token = create_access_token(user.id)
    _set_auth_cookie(response, token)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user.to_dict()))


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account (and log in)",
)
def register(payload: RegisterRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    invite_code = settings.registration_invite_code
    if invite_code and not hmac.compare_digest(payload.invite_code or "", invite_code):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid invite code.")

    if db.query(User).filter(User.username == payload.username).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username is already taken.")

    user = User(username=payload.username, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return _login_response(response, user)


@router.post("/login", response_model=TokenResponse, summary="Log in")
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    user = authenticate(db, payload.username, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password."
        )
    return _login_response(response, user)


@router.post("/logout", summary="Log out")
def logout(response: Response) -> dict:
    response.delete_cookie(ACCESS_TOKEN_COOKIE)
    return {"logged_out": True}


@router.get("/me", response_model=UserResponse, summary="Get the current user")
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(user.to_dict())
