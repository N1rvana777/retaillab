"""Tests for the InventoryOptimizer."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quant_engine.domain.entities import SKU
from quant_engine.domain.enums import OptimizationMethod, ProductCategory
from quant_engine.domain.value_objects import CostParameters
from quant_engine.optimization.constraints import OrderConstraints
from quant_engine.optimization.inventory_optimizer import InventoryOptimizer


class MockForecastModel:
    """Mock model that returns predictable values for testing."""

    def __init__(self, base_demand: float = 100.0, spread: float = 20.0):
        self._base = base_demand
        self._spread = spread

    def predict_quantile(self, features: np.ndarray, quantile: float) -> float:
        from scipy import stats
        z = stats.norm.ppf(quantile)
        return max(0, self._base + z * self._spread)

    def predict_mean(self, features: np.ndarray) -> float:
        return self._base


@pytest.fixture
def optimizer() -> InventoryOptimizer:
    return InventoryOptimizer(
        forecast_model=MockForecastModel(base_demand=100, spread=20),
        cold_start_model=MockForecastModel(base_demand=50, spread=10),
        min_history_days=28,
    )


@pytest.fixture
def sku() -> SKU:
    return SKU(
        sku_id="TEST-001",
        store_id="STORE-A",
        name="Test Product",
        category=ProductCategory.STAPLE,
    )


@pytest.fixture
def cost_params() -> CostParameters:
    return CostParameters(
        unit_cost=4.0,
        selling_price=8.0,
        goodwill_penalty=1.0,
        salvage_value=1.0,
        holding_cost_per_day=0.02,
        shelf_life_days=30,
    )


class TestPrimaryPath:
    """Test the main optimization path with sufficient history."""

    def test_returns_valid_order(self, optimizer, sku, cost_params):
        features = np.random.rand(10)
        order = optimizer.optimize(
            sku=sku,
            features=features,
            cost_params=cost_params,
            delivery_date=date(2025, 3, 15),
            history_days=60,
        )
        assert order.sku_id == "TEST-001"
        assert order.recommended_qty > 0
        assert order.method == OptimizationMethod.QUANTILE_DIRECT
        assert not order.fallback_used

    def test_higher_critical_ratio_gives_more_stock(self, optimizer, sku):
        """Higher CR (more underage-averse) should recommend more units."""
        features = np.random.rand(10)

        # Low CR (perishable-like): conservative
        low_cr_params = CostParameters(
            unit_cost=3.0, selling_price=5.0,
            salvage_value=0.0, holding_cost_per_day=0.5, shelf_life_days=3,
        )
        order_low = optimizer.optimize(
            sku=sku, features=features, cost_params=low_cr_params,
            delivery_date=date(2025, 3, 15), history_days=60,
        )

        # High CR (staple-like): aggressive
        high_cr_params = CostParameters(
            unit_cost=3.0, selling_price=10.0,
            goodwill_penalty=5.0, salvage_value=2.0,
            holding_cost_per_day=0.01, shelf_life_days=180,
        )
        order_high = optimizer.optimize(
            sku=sku, features=features, cost_params=high_cr_params,
            delivery_date=date(2025, 3, 15), history_days=60,
        )

        assert order_high.recommended_qty > order_low.recommended_qty


class TestColdStartPath:
    """Test the cold-start path for new products."""

    def test_uses_cold_start_for_new_sku(self, optimizer, sku, cost_params):
        features = np.random.rand(10)
        order = optimizer.optimize(
            sku=sku, features=features, cost_params=cost_params,
            delivery_date=date(2025, 3, 15),
            history_days=5,  # Below min_history_days
        )
        assert order.method == OptimizationMethod.COLD_START
        assert "category-level" in (order.reasoning or "").lower()

    def test_cold_start_is_conservative(self, optimizer, sku, cost_params):
        """Cold start should recommend less than primary (0.8x factor)."""
        features = np.random.rand(10)

        primary_order = optimizer.optimize(
            sku=sku, features=features, cost_params=cost_params,
            delivery_date=date(2025, 3, 15), history_days=60,
        )
        cold_order = optimizer.optimize(
            sku=sku, features=features, cost_params=cost_params,
            delivery_date=date(2025, 3, 15), history_days=5,
        )

        # Cold start uses a different model with lower base demand (50 vs 100)
        # AND applies 0.8x conservative factor
        assert cold_order.recommended_qty < primary_order.recommended_qty


class TestConstraints:
    """Test that operational constraints are applied."""

    def test_moq_applied(self, optimizer, sku, cost_params):
        features = np.random.rand(10)
        constraints = OrderConstraints(moq=24)
        order = optimizer.optimize(
            sku=sku, features=features, cost_params=cost_params,
            delivery_date=date(2025, 3, 15), history_days=60,
            constraints=constraints,
        )
        assert order.recommended_qty % 24 == 0 or order.recommended_qty == 0

    def test_shelf_capacity_respected(self, optimizer, sku, cost_params):
        features = np.random.rand(10)
        constraints = OrderConstraints(shelf_capacity=50)
        order = optimizer.optimize(
            sku=sku, features=features, cost_params=cost_params,
            delivery_date=date(2025, 3, 15), history_days=60,
            constraints=constraints,
        )
        assert order.recommended_qty <= 50
