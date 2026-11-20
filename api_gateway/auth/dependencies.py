"""FastAPI authentication dependencies."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from api_gateway.auth.jwt_handler import verify_token

_bearer_scheme = HTTPBearer()


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency that enforces JWT authentication.

    Usage in routes:
        @router.get("/protected")
        async def protected_route(user: dict = Depends(require_auth)):
            return {"user": user["sub"]}

    Raises:
        HTTPException: 401 if token is missing or invalid.
    """
    try:
        payload = verify_token(credentials.credentials)
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
