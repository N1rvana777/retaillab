"""Replenishment endpoints — the primary business interface.

CRITICAL: All routes wrap quant_engine calls in try-except blocks.
If the ML engine fails for ANY reason, the system degrades gracefully
to safety stock fallback. The API NEVER returns 500 for business logic failures.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from api_gateway.auth.dependencies import require_auth
from api_gateway.schemas.replenishment import (
    ReplenishmentLineItem,
    ReplenishmentOrder,
    ReplenishmentRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/replenish", tags=["replenishment"])


@router.post("", response_model=ReplenishmentOrder)
async def create_replenishment_order(
    request: ReplenishmentRequest,
    _user: dict = Depends(require_auth),
) -> ReplenishmentOrder:
    """Generate a replenishment order for a store.

    This is the main business endpoint. It calls the quant_engine
    to compute optimal order quantities for all (or specified) SKUs.

    If the quant_engine fails, it falls back to safety stock rules.
    The response includes `fallback_used` flags so the frontend can
    highlight degraded recommendations.
    """
    try:
        # TODO: Wire to quant_engine.service once fully integrated
        # For now, return a placeholder that demonstrates the API contract
        line_items = [
            ReplenishmentLineItem(
                sku_id="PLACEHOLDER",
                sku_name="System not yet initialized",
                category="staple",
                recommended_qty=0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                critical_ratio=0.5,
                forecast_mean=0.0,
                expected_cost_if_ordered=0.0,
                method="not_initialized",
                fallback_used=False,
            )
        ]

        return ReplenishmentOrder(
            order_id=str(uuid.uuid4()),
            store_id=request.store_id,
            generated_at=datetime.now(timezone.utc),
            delivery_date=request.delivery_date,
            line_items=line_items,
            total_units=sum(item.recommended_qty for item in line_items),
            total_estimated_cost=sum(item.expected_cost_if_ordered for item in line_items),
            fallback_count=sum(1 for item in line_items if item.fallback_used),
        )

    except Exception:
        logger.exception(
            "Replenishment engine failed for store %s", request.store_id
        )
        # Second-level fallback: return a minimal valid response
        # rather than a 500 error
        return ReplenishmentOrder(
            order_id=str(uuid.uuid4()),
            store_id=request.store_id,
            generated_at=datetime.now(timezone.utc),
            delivery_date=request.delivery_date,
            line_items=[],
            total_units=0,
            total_estimated_cost=0.0,
            fallback_count=0,
            model_version="fallback",
        )


@router.get("/{store_id}/latest", response_model=ReplenishmentOrder | None)
async def get_latest_order(
    store_id: str,
    _user: dict = Depends(require_auth),
) -> ReplenishmentOrder | None:
    """Retrieve the most recent replenishment order for a store.

    Returns None if no orders have been generated yet.
    """
    # TODO: Retrieve from storage (Redis cache or DB)
    return None
