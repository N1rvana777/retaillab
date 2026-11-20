"""Tests for the cold-start forecaster."""

from __future__ import annotations

import numpy as np

from quant_engine.forecasting.cold_start import ColdStartForecaster


class TestColdStartForecaster:
    def test_predict_quantile_returns_positive(self):
        forecaster = ColdStartForecaster(default_daily_demand=20.0)
        features = np.zeros(10)
        result = forecaster.predict_quantile(features, quantile=0.5)
        assert result > 0

    def test_higher_quantile_gives_higher_value(self):
        forecaster = ColdStartForecaster(default_daily_demand=20.0)
        features = np.zeros(10)
        low = forecaster.predict_quantile(features, quantile=0.1)
        high = forecaster.predict_quantile(features, quantile=0.9)
        assert high > low

    def test_predict_mean_equals_median(self):
        forecaster = ColdStartForecaster(default_daily_demand=20.0)
        features = np.zeros(10)
        mean = forecaster.predict_mean(features)
        median = forecaster.predict_quantile(features, quantile=0.5)
        assert abs(mean - median) < 0.01
