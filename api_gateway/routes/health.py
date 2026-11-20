"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from api_gateway.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check. Returns model and drift status."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=False,  # TODO: check actual model status
        drift_status="unknown",
    )
