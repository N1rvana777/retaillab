"""Pydantic models for incoming POS data.

These schemas are intentionally PERMISSIVE — they accept dirty data
and let the data_pipeline handle cleaning. Only structurally broken
records (missing required fields) are rejected at the API boundary.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class RawPOSTransaction(BaseModel):
    """Single POS transaction — accepts messy real-world data."""

    transaction_id: str | None = None
    store_id: str
    sku_id: str
    timestamp: datetime
    quantity_sold: float
    unit_price: float
    discount_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_amount: float | None = None
    payment_method: str | None = None
    cashier_id: str | None = None
    category: str | None = None

    @field_validator("quantity_sold", mode="before")
    @classmethod
    def coerce_quantity(cls, v: float) -> float:
        """Accept negative quantities (returns) — cleaning handles them."""
        return float(v)

    @field_validator("unit_price", mode="before")
    @classmethod
    def coerce_price(cls, v: float) -> float:
        """Reject negative prices at the API boundary."""
        v = float(v)
        if v < 0:
            msg = "unit_price cannot be negative"
            raise ValueError(msg)
        return v


class POSDataBatch(BaseModel):
    """Batch upload of POS transactions."""

    store_id: str
    upload_timestamp: datetime
    transactions: list[RawPOSTransaction]
    source_system: str = "unknown"


class DataQualityReport(BaseModel):
    """Returned after data ingestion to inform callers of data quality."""

    total_records: int
    accepted_records: int
    flagged_records: int
    rejected_records: int
    flags: list[str]
    cleaning_applied: list[str]
