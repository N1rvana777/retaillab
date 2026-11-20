"""Pydantic models for demand forecast responses."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class ForecastSummary(BaseModel):
    """Demand forecast summary for a single SKU."""

    sku_id: str
    store_id: str
    target_date: date
    predicted_mean: float
    predicted_std: float
    quantiles: dict[str, float]
    model_version: str = "v0.1.0"
