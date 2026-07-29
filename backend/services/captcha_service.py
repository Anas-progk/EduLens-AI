"""Turnstile CAPTCHA verification service."""

import httpx
from fastapi import HTTPException, status

from backend.config import (
    TURNSTILE_ENABLED,
    TURNSTILE_SECRET_KEY,
    TURNSTILE_VERIFY_URL,
)


async def verify_turnstile(token: str) -> bool:
    """
    Verify a Turnstile token with Cloudflare.
    
    Args:
        token: The Turnstile response token from the frontend.
        
    Returns:
        True if verification succeeds.
        
    Raises:
        HTTPException: If verification fails or service is unavailable.
    """
    if not TURNSTILE_ENABLED:
        return True
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA token is required",
        )
    
    if not TURNSTILE_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CAPTCHA service not configured",
        )
    
    payload = {
        "secret": TURNSTILE_SECRET_KEY,
        "response": token,
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(TURNSTILE_VERIFY_URL, data=payload)
            resp.raise_for_status()
            result = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CAPTCHA verification timeout",
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CAPTCHA service unavailable",
        )
    
    if not result.get("success"):
        error_codes = result.get("error-codes", [])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA verification failed",
        )
    
    return True