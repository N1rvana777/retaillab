"""Tests for asymmetric loss functions."""

from __future__ import annotations

import numpy as np
import pytest

from quant_engine.optimization.loss_functions import (
    asymmetric_inventory_cost,
    coverage_rate,
    pinball_loss,
    spoilage_adjusted_cost,
)


class TestPinballLoss:
    """Test the quantile (pinball) loss function."""

    def test_zero_loss_at_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert pinball_loss(y, y, quantile=0.5) == 0.0

    def test_underpredict_penalized_more_at_high_quantile(self):
        """At q=0.9, under-predicting is 9x more costly than over-predicting."""
        y_true = np.array([100.0])
        y_under = np.array([80.0])   # under by 20
        y_over = np.array([120.0])   # over by 20

        loss_under = pinball_loss(y_true, y_under, quantile=0.9)
        loss_over = pinball_loss(y_true, y_over, quantile=0.9)

        assert loss_under > loss_over

    def test_overpredict_penalized_more_at_low_quantile(self):
        """At q=0.1, over-predicting is more costly."""
        y_true = np.array([100.0])
        y_under = np.array([80.0])
        y_over = np.array([120.0])

        loss_under = pinball_loss(y_true, y_under, quantile=0.1)
        loss_over = pinball_loss(y_true, y_over, quantile=0.1)

        assert loss_over > loss_under


class TestAsymmetricCost:
    """Test the Newsvendor cost function."""

    def test_no_cost_when_exact(self):
        demand = np.array([100.0])
        order = np.array([100.0])
        cost = asymmetric_inventory_cost(demand, order, cu=5.0, co=3.0)
        assert cost[0] == 0.0

    def test_underage_cost(self):
        demand = np.array([120.0])
        order = np.array([100.0])
        cost = asymmetric_inventory_cost(demand, order, cu=5.0, co=3.0)
        # 20 units short * Cu=5
        assert cost[0] == pytest.approx(100.0)

    def test_overage_cost(self):
        demand = np.array([80.0])
        order = np.array([100.0])
        cost = asymmetric_inventory_cost(demand, order, cu=5.0, co=3.0)
        # 20 units excess * Co=3
        assert cost[0] == pytest.approx(60.0)

    def test_vectorized(self):
        demand = np.array([80.0, 100.0, 120.0])
        order = np.array([100.0, 100.0, 100.0])
        costs = asymmetric_inventory_cost(demand, order, cu=5.0, co=3.0)
        assert len(costs) == 3
        assert costs[0] == pytest.approx(60.0)   # overage
        assert costs[1] == pytest.approx(0.0)    # exact
        assert costs[2] == pytest.approx(100.0)  # underage


class TestSpoilageAdjustedCost:
    """Test the perishable spoilage cost model."""

    def test_no_leftover_no_extra_cost(self):
        demand = np.array([100.0])
        order = np.array([80.0])  # underage only
        base = asymmetric_inventory_cost(demand, order, cu=5.0, co=3.0)
        adjusted = spoilage_adjusted_cost(demand, order, cu=5.0, co=3.0, shelf_life_days=3)
        # No leftover → no markdown → costs should be equal
        np.testing.assert_array_almost_equal(base, adjusted)

    def test_leftover_increases_cost(self):
        demand = np.array([80.0])
        order = np.array([100.0])
        base = asymmetric_inventory_cost(demand, order, cu=5.0, co=3.0)
        adjusted = spoilage_adjusted_cost(demand, order, cu=5.0, co=3.0, shelf_life_days=3)
        assert adjusted[0] > base[0]


class TestCoverageRate:
    """Test the coverage rate metric."""

    def test_perfect_high_quantile(self):
        """If all predictions are above actuals, coverage = 1.0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])
        assert coverage_rate(y_true, y_pred) == 1.0

    def test_perfect_low_quantile(self):
        """If all predictions are below actuals, coverage = 0.0."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert coverage_rate(y_true, y_pred) == 0.0
