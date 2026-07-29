from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from backend.security import decode_token
from backend.database import get_user_by_id, get_session, check_session_ownership

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login/oauth")


def get_current_user(
    token: str = Depends(oauth2_scheme),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    try:
        payload = decode_token(token)
        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = get_user_by_id(user_id)

    if user is None:
        raise credentials_exception

    return user


def require_teacher(
    current_user: dict = Depends(get_current_user),
):
    if current_user["role"] not in ("teacher", "hod", "principal"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher access required",
        )
    return current_user


def require_hod(
    current_user: dict = Depends(get_current_user),
):
    if current_user["role"] not in ("hod", "principal"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="HOD access required",
        )
    return current_user


def require_principal(
    current_user: dict = Depends(get_current_user),
):
    if current_user["role"] != "principal":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Principal access required",
        )
    return current_user


def verify_session_owner(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Verify that the current user owns the session (or is HOD/Principal who can view all)."""
    # HOD and Principal can access all sessions
    if current_user["role"] in ("hod", "principal"):
        session = get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
        return session
    
    # Teacher can only access their own sessions
    session = check_session_ownership(session_id, current_user["id"])
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or access denied",
        )
    return session