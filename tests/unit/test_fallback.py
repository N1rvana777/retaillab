"""Tests for the safety stock fallback engine.

Verifies that the fallback produces reasonable orders even when
the primary ML engine is completely unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.domain.enums import OptimizationMethod
from quant_engine.fallback.safety_stock import SafetyStockFallback


@pytest.fixture
def fallback() -> SafetyStockFallback:
    return SafetyStockFallback(window_days=7, safety_factor=1.2)


class TestSafetyStockCompute:
    """Test basic quantity computation."""

    def test_normal_computation(self, fallback):
        """Moving average * safety factor."""
        sales = [10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 13.0]
        qty = fallback.compute(sales)
        expected = np.mean(sales) * 1.2
        assert abs(qty - expected) < 0.01

    def test_uses_last_n_days(self, fallback):
        """Should only use the last window_days of data."""
        sales = [100.0] * 30 + [10.0] * 7  # Old data is high, recent is low
        qty = fallback.compute(sales)
        expected = np.mean([10.0] * 7) * 1.2
        assert abs(qty - expected) < 0.01

    def test_empty_sales_uses_default(self, fallback):
        """With no sales data, fall back to category default."""
        qty = fallback.compute(None, category="staple")
        assert qty > 0

    def test_result_non_negative(self, fallback):
        """Quantity should never be negative."""
        qty = fallback.compute([0.0, 0.0, 0.0])
        assert qty >= 0.0


class TestSafetyStockOrder:
    """Test full ReplenishmentOrder generation from fallback."""

    def test_order_has_fallback_flag(self, fallback):
        """Fallback orders must be clearly marked."""
        order = fallback.compute_order(
            sku_id="TEST-001",
            store_id="STORE-A",
            recent_sales=[10.0, 12.0, 8.0],
            category="perishable",
            error_message="Model file not found",
        )
        assert order.fallback_used is True
        assert order.method == OptimizationMethod.FALLBACK_SAFETY_STOCK
        assert "Model file not found" in (order.reasoning or "")

    def test_order_quantity_positive(self, fallback):
        """Fallback should produce a positive order (unless truly zero demand)."""
        order = fallback.compute_order(
            sku_id="TEST-001",
            store_id="STORE-A",
            recent_sales=[5.0, 8.0, 3.0, 10.0, 7.0],
        )
        assert order.recommended_qty > 0

    def test_order_with_no_sales(self, fallback):
        """Even with no sales data, should produce a valid order."""
        order = fallback.compute_order(
            sku_id="NEW-SKU",
            store_id="STORE-A",
            recent_sales=None,
            category="staple",
        )
        assert order.recommended_qty >= 0
        assert order.fallback_used is True
