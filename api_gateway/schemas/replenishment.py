"""Pydantic models for replenishment order responses.

The API exposes ONLY order quantities and aggregate forecasts.
Model weights, feature importances, and training data are NEVER serialized.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class ReplenishmentLineItem(BaseModel):
    """Single SKU replenishment recommendation."""

    sku_id: str
    sku_name: str = ""
    category: str
    recommended_qty: int = Field(ge=0)
    unit_of_measure: str = "units"
    confidence_lower: float
    confidence_upper: float
    critical_ratio: float = Field(ge=0.0, le=1.0)
    forecast_mean: float
    expected_cost_if_ordered: float
    method: str
    fallback_used: bool = False
    reasoning: str | None = None
    constraints_applied: list[str] = Field(default_factory=list)


class ReplenishmentOrder(BaseModel):
    """Complete replenishment order for a store."""

    order_id: str
    store_id: str
    generated_at: datetime
    delivery_date: date
    line_items: list[ReplenishmentLineItem]
    total_units: int
    total_estimated_cost: float
    model_version: str = "v0.1.0"
    fallback_count: int = 0
    """Number of items where the fallback engine was used."""


class ReplenishmentRequest(BaseModel):
    """Input to trigger replenishment calculation."""

    store_id: str
    delivery_date: date
    sku_ids: list[str] | None = None
    override_safety_stock: float | None = None
