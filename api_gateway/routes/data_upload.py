"""Data upload endpoint for POS transaction data."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api_gateway.auth.dependencies import require_auth
from api_gateway.schemas.pos_data import DataQualityReport, POSDataBatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/pos", response_model=DataQualityReport)
async def upload_pos_data(
    batch: POSDataBatch,
    _user: dict = Depends(require_auth),
) -> DataQualityReport:
    """Accept a batch of raw POS transactions.

    The data is validated at the API boundary (Pydantic),
    then passed to the data_pipeline for cleaning and feature engineering.

    Returns a quality report showing how many records were accepted,
    flagged, or rejected.
    """
    total = len(batch.transactions)

    # TODO: Wire to data_pipeline.orchestrator
    # For now, accept all and return a placeholder report

    logger.info(
        "Received %d transactions from store %s (source: %s)",
        total,
        batch.store_id,
        batch.source_system,
    )

    return DataQualityReport(
        total_records=total,
        accepted_records=total,
        flagged_records=0,
        rejected_records=0,
        flags=[],
        cleaning_applied=["placeholder — pipeline not yet wired"],
    )
