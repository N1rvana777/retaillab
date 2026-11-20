"""Forecast endpoint — exposes demand predictions (not model internals)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_gateway.auth.dependencies import require_auth
from api_gateway.schemas.forecast import ForecastSummary

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/{store_id}/{sku_id}", response_model=ForecastSummary | None)
async def get_forecast(
    store_id: str,
    sku_id: str,
    _user: dict = Depends(require_auth),
) -> ForecastSummary | None:
    """Get the latest demand forecast for a specific SKU in a store.

    Returns quantile predictions and summary statistics.
    Does NOT expose model weights or feature importances.
    """
    # TODO: Wire to quant_engine.service
    return None
