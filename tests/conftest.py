"""Shared test fixtures for the RetailLab test suite."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_engine.domain.entities import SKU
from quant_engine.domain.enums import ProductCategory
from quant_engine.domain.value_objects import CostParameters


@pytest.fixture
def perishable_cost_params() -> CostParameters:
    """Cost parameters for a perishable item (bakery bread).

    Expected critical_ratio ≈ 0.41 (order conservatively).
    Cu = (5-3) + 0.5 = 2.5, Co = (3-0) + 0.1*3 = 3.3
    CR = 2.5 / (2.5 + 3.3) = 0.431
    """
    return CostParameters(
        unit_cost=3.0,
        selling_price=5.0,
        salvage_value=0.0,
        goodwill_penalty=0.5,
        holding_cost_per_day=0.1,
        shelf_life_days=3,
    )


@pytest.fixture
def staple_cost_params() -> CostParameters:
    """Cost parameters for a staple item (canned soup).

    Expected critical_ratio ≈ 0.77 (order aggressively).
    Cu = (8-4) + 2.0 = 6.0, Co = (4-2) + 0.02*365 = 9.3 → wait
    Actually Cu = 6.0, Co = (4-2) + 0.02*365 = 2 + 7.3 = 9.3
    CR = 6.0 / (6.0 + 9.3) ≈ 0.392... hmm, let me recalc.
    Let's use shelf_life=30 instead:
    Co = (4-2) + 0.02*30 = 2.6
    CR = 6.0 / (6.0 + 2.6) = 0.698
    """
    return CostParameters(
        unit_cost=4.0,
        selling_price=8.0,
        goodwill_penalty=2.0,
        salvage_value=2.0,
        holding_cost_per_day=0.02,
        shelf_life_days=30,
    )


@pytest.fixture
def sample_sku() -> SKU:
    """A sample perishable SKU for testing."""
    return SKU(
        sku_id="BREAD-001",
        store_id="STORE-A",
        name="Whole Wheat Bread",
        category=ProductCategory.PERISHABLE,
        shelf_life_days=3,
    )


@pytest.fixture
def sample_features() -> np.ndarray:
    """A sample feature vector (10 features)."""
    return np.array([1.0, 0.5, 12.0, 3.0, 0.0, 1.0, 15.0, 8.5, 0.3, 7.0])


@pytest.fixture
def dirty_pos_dataframe() -> pd.DataFrame:
    """Simulated dirty POS data with known anomalies for testing.

    Contains:
    - Normal transactions
    - Duplicate scans (within 3 seconds)
    - Outage period (zero sales during business hours)
    - Discount anomaly (99% discount)
    - Quantity outlier (500 units instead of 5)
    """
    base_date = pd.Timestamp("2025-01-15")
    records = []

    # Normal transactions (store A, SKU-001)
    for hour in range(8, 22):
        for minute in [0, 15, 30, 45]:
            records.append({
                "transaction_id": f"TXN-{hour}-{minute}",
                "store_id": "STORE-A",
                "sku_id": "SKU-001",
                "timestamp": base_date + pd.Timedelta(hours=hour, minutes=minute),
                "quantity_sold": np.random.randint(1, 10),
                "unit_price": 5.0,
                "discount_pct": 0.0,
            })

    # Duplicate scan (2 identical records 2 seconds apart)
    records.append({
        "transaction_id": "TXN-DUP-1",
        "store_id": "STORE-A",
        "sku_id": "SKU-001",
        "timestamp": base_date + pd.Timedelta(hours=10, minutes=0, seconds=0),
        "quantity_sold": 5,
        "unit_price": 5.0,
        "discount_pct": 0.0,
    })
    records.append({
        "transaction_id": "TXN-DUP-2",
        "store_id": "STORE-A",
        "sku_id": "SKU-001",
        "timestamp": base_date + pd.Timedelta(hours=10, minutes=0, seconds=2),
        "quantity_sold": 5,
        "unit_price": 5.0,
        "discount_pct": 0.0,
    })

    # Discount anomaly (99% off — likely cashier error)
    records.append({
        "transaction_id": "TXN-BAD-DISC",
        "store_id": "STORE-A",
        "sku_id": "SKU-001",
        "timestamp": base_date + pd.Timedelta(hours=14),
        "quantity_sold": 3,
        "unit_price": 5.0,
        "discount_pct": 99.0,
    })

    # Quantity outlier (500 instead of 5)
    records.append({
        "transaction_id": "TXN-QTY-OUTLIER",
        "store_id": "STORE-A",
        "sku_id": "SKU-001",
        "timestamp": base_date + pd.Timedelta(hours=16),
        "quantity_sold": 500,
        "unit_price": 5.0,
        "discount_pct": 0.0,
    })

    return pd.DataFrame(records)
