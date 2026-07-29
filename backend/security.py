from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import hashlib
import secrets

import bcrypt
from jose import JWTError, jwt

from backend.config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)


# --------------------------
# Password Hashing
# --------------------------

def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    """
    hashed = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    )
    return hashed.decode("utf-8")


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


# --------------------------
# JWT
# --------------------------

def create_access_token(
    data: Dict[str, Any]
) -> str:
    """
    Create a signed JWT access token.
    """

    payload = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    })

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token


def decode_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT.
    Raises JWTError if invalid.
    """

    payload = jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM]
    )

    return payload


# --------------------------
# Refresh Tokens
# --------------------------

def create_refresh_token() -> str:
    """
    Generate a secure random refresh token (opaque, not JWT).
    """
    return secrets.token_urlsafe(64)


def hash_refresh_token(token: str) -> str:
    """
    Hash a refresh token using SHA-256 for storage.
    """
    return hashlib.sha256(token.encode()).hexdigest()


def verify_refresh_token(token: str, token_hash: str) -> bool:
    """
    Verify a refresh token against its stored hash.
    """
    return hash_refresh_token(token) == token_hash


def rotate_refresh_token(old_token: str, old_token_hash: str, user_id: str, expires_at: str) -> tuple[str, str]:
    """
    Rotate a refresh token: generate new token, return (new_token, new_token_hash).
    The database layer handles the actual replacement.
    """
    new_token = create_refresh_token()
    new_token_hash = hash_refresh_token(new_token)
    return new_token, new_token_hash
