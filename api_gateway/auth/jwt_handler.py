"""JWT token creation and validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from api_gateway.config import APIConfig

_config = APIConfig()


def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        subject: The token subject (typically a user ID).
        expires_delta: Custom expiration. Defaults to config value.

    Returns:
        Encoded JWT string.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=_config.jwt_expire_minutes)
    )
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _config.jwt_secret, algorithm=_config.jwt_algorithm)


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token.

    Args:
        token: The JWT string to verify.

    Returns:
        Decoded payload dictionary.

    Raises:
        JWTError: If the token is invalid or expired.
    """
    return jwt.decode(token, _config.jwt_secret, algorithms=[_config.jwt_algorithm])
