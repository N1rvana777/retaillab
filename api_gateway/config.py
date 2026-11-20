"""API Gateway configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """API configuration loaded from environment variables."""

    jwt_secret: str = "change-me-to-a-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    redis_url: str = "redis://localhost:6379/0"
    model_dir: str = "./models"
    log_level: str = "INFO"
    api_version: str = "v0.1.0"
    rate_limit_per_minute: int = 60

    model_config = {"env_prefix": ""}
