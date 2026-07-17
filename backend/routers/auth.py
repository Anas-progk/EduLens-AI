"""Authentication router — demo-mode login."""

import uuid
import logging
from fastapi import APIRouter, HTTPException
from backend.schemas import LoginRequest, LoginResponse, UserOut
from backend.database import get_user_by_email, log_audit

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """Login with email/password. Demo mode: password stored plaintext."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(401, "Invalid credentials")

    # Demo: plaintext compare (use bcrypt in production)
    if user["password_hash"] != req.password:
        raise HTTPException(401, "Invalid credentials")

    token = f"tok_{uuid.uuid4().hex}"
    log_audit(user["id"], "login", f"email={req.email}")

    return LoginResponse(
        token=token,
        user=UserOut(
            id=user["id"],
            name=user["name"],
            email=user["email"],
            role=user["role"],
        ),
    )


@router.post("/logout")
async def logout():
    return {"message": "Logged out"}
