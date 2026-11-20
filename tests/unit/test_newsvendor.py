"""Tests for the Newsvendor model solver.

Verifies mathematical correctness against known analytical solutions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quant_engine.domain.value_objects import CostParameters
from quant_engine.optimization.newsvendor import (
    expected_newsvendor_cost,
    newsvendor_from_quantile_forecast,
    newsvendor_optimal_quantity,
    robust_std_from_quantiles,
)


class TestNewsvendorOptimalQuantity:
    """Test q* = F^{-1}(Cu / (Cu + Co))."""

    def test_symmetric_cost_gives_median(self):
        """When Cu == Co, critical ratio = 0.5, so q* = median."""
        dist = stats.norm(loc=100, scale=20)
        q_star = newsvendor_optimal_quantity(0.5, dist)
        assert abs(q_star - 100.0) < 0.01

    def test_high_underage_shifts_right(self):
        """When Cu >> Co, order aggressively (above median)."""
        dist = stats.norm(loc=100, scale=20)
        q_star = newsvendor_optimal_quantity(0.9, dist)
        assert q_star > 100.0

    def test_high_overage_shifts_left(self):
        """When Co >> Cu, order conservatively (below median)."""
        dist = stats.norm(loc=100, scale=20)
        q_star = newsvendor_optimal_quantity(0.2, dist)
        assert q_star < 100.0

    def test_known_normal_solution(self):
        """Verify against hand-calculated normal quantile."""
        dist = stats.norm(loc=50, scale=10)
        cr = 0.75
        expected = 50 + 10 * stats.norm.ppf(0.75)
        q_star = newsvendor_optimal_quantity(cr, dist)
        assert abs(q_star - expected) < 0.01

    def test_invalid_critical_ratio_raises(self):
        dist = stats.norm(loc=100, scale=20)
        with pytest.raises(ValueError, match="critical_ratio"):
            newsvendor_optimal_quantity(0.0, dist)
        with pytest.raises(ValueError, match="critical_ratio"):
            newsvendor_optimal_quantity(1.0, dist)

    def test_result_is_non_negative(self):
        """Even with very low critical ratio, quantity is >= 0."""
        dist = stats.norm(loc=5, scale=20)  # High variance, low mean
        q_star = newsvendor_optimal_quantity(0.01, dist)
        assert q_star >= 0.0


class TestCostParameters:
    """Test the CostParameters value object and critical ratio calculation."""

    def test_perishable_critical_ratio_below_half(self, perishable_cost_params):
        """Perishables should have CR < 0.5 (order conservatively)."""
        cr = perishable_cost_params.critical_ratio
        assert 0 < cr < 0.5, f"Expected CR < 0.5 for perishable, got {cr}"

    def test_staple_critical_ratio_above_half(self, staple_cost_params):
        """Staples should have CR > 0.5 (order aggressively)."""
        cr = staple_cost_params.critical_ratio
        assert cr > 0.5, f"Expected CR > 0.5 for staple, got {cr}"

    def test_critical_ratio_range(self, perishable_cost_params, staple_cost_params):
        """Critical ratio must always be in (0, 1)."""
        for cp in [perishable_cost_params, staple_cost_params]:
            assert 0 < cp.critical_ratio < 1

    def test_invalid_negative_margin_raises(self):
        """selling_price < unit_cost should raise ValueError."""
        with pytest.raises(ValueError, match="negative margin"):
            CostParameters(unit_cost=10.0, selling_price=5.0)

    def test_salvage_exceeds_cost_raises(self):
        """salvage_value > unit_cost is illogical."""
        with pytest.raises(ValueError, match="salvage"):
            CostParameters(unit_cost=5.0, selling_price=10.0, salvage_value=6.0)


class TestQuantileAdjustment:
    """Test the quantile forecast adjustment function."""

    def test_no_adjustment_when_quantiles_match(self):
        """If predicted quantile == critical ratio, no adjustment."""
        result = newsvendor_from_quantile_forecast(
            predicted_quantile_value=100.0,
            predicted_quantile_level=0.5,
            critical_ratio=0.5,
            demand_std=20.0,
        )
        assert abs(result - 100.0) < 0.01

    def test_shifts_up_for_higher_target(self):
        """If target quantile > predicted quantile, shift up."""
        result = newsvendor_from_quantile_forecast(
            predicted_quantile_value=100.0,
            predicted_quantile_level=0.5,
            critical_ratio=0.8,
            demand_std=20.0,
        )
        assert result > 100.0

    def test_zero_std_returns_original(self):
        """With zero std, return the predicted value as-is."""
        result = newsvendor_from_quantile_forecast(
            predicted_quantile_value=50.0,
            predicted_quantile_level=0.5,
            critical_ratio=0.9,
            demand_std=0.0,
        )
        assert result == 50.0


class TestExpectedCost:
    """Test the analytical expected Newsvendor cost."""

    def test_optimal_quantity_minimizes_cost(self):
        """The Newsvendor optimal quantity should give near-minimum cost."""
        mean, std, cu, co = 100, 20, 5, 3
        cr = cu / (cu + co)
        q_star = mean + std * stats.norm.ppf(cr)

        cost_at_optimal = expected_newsvendor_cost(q_star, mean, std, cu, co)

        # Cost at sub-optimal quantities should be higher
        cost_too_low = expected_newsvendor_cost(q_star - 30, mean, std, cu, co)
        cost_too_high = expected_newsvendor_cost(q_star + 30, mean, std, cu, co)

        assert cost_at_optimal < cost_too_low
        assert cost_at_optimal < cost_too_high

    def test_deterministic_demand(self):
        """With zero std, cost should be exact over/under."""
        cost = expected_newsvendor_cost(
            order_qty=120, demand_mean=100, demand_std=0, cu=5, co=3,
        )
        assert abs(cost - 3 * 20) < 0.01  # 20 units overage * Co=3


class TestRobustStd:
    """Test the robust standard deviation estimator."""

    def test_known_normal(self):
        """For N(100, 20), q10=74.37, q90=125.63, std should be ~20."""
        q10 = 100 + 20 * stats.norm.ppf(0.1)
        q90 = 100 + 20 * stats.norm.ppf(0.9)
        estimated_std = robust_std_from_quantiles(q10, q90)
        assert abs(estimated_std - 20.0) < 0.1

    def test_zero_spread(self):
        """If q_low == q_high, std should be 0."""
        assert robust_std_from_quantiles(50, 50) == 0.0
