"""Authentication router — demo-mode login."""

import logging
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm

from backend.schemas import LoginRequest, LoginResponse, UserOut, RefreshRequest
from backend.database import get_user_by_email, get_user_by_id, log_audit, create_refresh_token, get_refresh_token, revoke_refresh_token, replace_refresh_token, hash_refresh_token
from backend.security import verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from backend.config import REFRESH_TOKEN_EXPIRE_DAYS, TURNSTILE_ENABLED
from backend.services.captcha_service import verify_turnstile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    if not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    token = create_access_token(
        {
            "sub": str(user["id"]),
            "email": user["email"],
            "role": user["role"],
            "name": user["name"],
        }
    )

    # Generate refresh token (7 days expiry)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)).isoformat()
    refresh_token = create_refresh_token(user["id"], expires_at)

    log_audit(user["id"], "login", f"email={email}")

    return LoginResponse(
        access_token=token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserOut(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            role=user["role"],
        ),
    )


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Authenticate user and return access + refresh tokens."""
    # Verify CAPTCHA if enabled and provided
    if TURNSTILE_ENABLED and req.captcha_token:
        await verify_turnstile(req.captcha_token)
    elif TURNSTILE_ENABLED and not req.captcha_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA token is required",
        )
    return authenticate_user(req.email, req.password)


@router.post("/login/oauth", response_model=LoginResponse)
async def login_oauth(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """OAuth2 compatible login for Swagger UI (no CAPTCHA for development)."""
    return authenticate_user(form_data.username, form_data.password)


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(req: RefreshRequest):
    """Rotate refresh token and return new access + refresh tokens."""
    # Hash the provided refresh token to look it up
    token_hash = hash_refresh_token(req.refresh_token)
    
    # Look up the token in database
    stored = get_refresh_token(token_hash)
    if not stored:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    # Check if revoked
    if stored["revoked"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
        )
    
    # Check expiry
    expires_at = datetime.fromisoformat(stored["expires_at"].replace('Z', '+00:00'))
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )
    
    # Get user
    user = get_user_by_id(stored["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    # Generate new access token
    new_access_token = create_access_token(
        {
            "sub": str(user["id"]),
            "email": user["email"],
            "role": user["role"],
            "name": user["name"],
        }
    )
    
    # Rotate refresh token
    new_expires_at = (datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)).isoformat()
    new_refresh_token = create_refresh_token(user["id"], new_expires_at)
    
    # Replace old token with new one (rotation)
    new_token_hash = hash_refresh_token(new_refresh_token)
    replace_refresh_token(token_hash, new_token_hash)
    
    log_audit(user["id"], "refresh", "token_rotation")
    
    return LoginResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserOut(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            role=user["role"],
        ),
    )


@router.post("/logout")
async def logout(req: RefreshRequest):
    """Revoke refresh token (logout)."""
    token_hash = hash_refresh_token(req.refresh_token)
    revoke_refresh_token(token_hash)
    return {"message": "Logged out successfully"}