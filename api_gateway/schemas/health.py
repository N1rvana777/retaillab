"""Health check response schema."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    model_loaded: bool = False
    drift_status: str = "unknown"
    drift_message: str = ""
