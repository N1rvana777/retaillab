"""Tests for the data leakage guard.

Verifies that the guard correctly detects look-ahead bias in:
1. Lag features without proper shift
2. Future-dated external data joins
3. Non-temporal train/test splits
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.feature_engineering.leakage_guard import (
    LeakageGuard,
    LookAheadLeakageError,
)


@pytest.fixture
def guard() -> LeakageGuard:
    return LeakageGuard()


class TestLagFeatureValidation:
    """Test detection of lag features that peek at the future."""

    def test_high_correlation_detected(self, guard):
        """A feature that IS the target (correlation ~1) should be flagged."""
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=100),
            "target": range(100),
            "leaked_feature": range(100),  # Same as target = leakage
        })
        with pytest.raises(LookAheadLeakageError, match="LEAKAGE"):
            guard.validate_lag_features(
                df, date_col="date",
                feature_cols=["leaked_feature"],
                target_col="target",
            )

    def test_proper_lag_passes(self, guard):
        """A properly shifted feature (low same-day correlation) should pass."""
        import numpy as np
        np.random.seed(42)
        target = np.random.randn(100).cumsum()  # Random walk
        # A proper lag-7 feature: shifted enough to have low correlation
        lag_7 = [None] * 7 + list(target[:93])
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=100),
            "target": target,
            "proper_lag_7": lag_7,
        })
        # Should NOT raise — correlation between random walk and its lag-7 is < 0.95
        guard.validate_lag_features(
            df, date_col="date",
            feature_cols=["proper_lag_7"],
            target_col="target",
        )


class TestFutureJoinValidation:
    """Test detection of future-dated external data."""

    def test_future_weather_detected(self, guard):
        """Weather data extending beyond sales data = potential leakage."""
        main_df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", "2025-01-31"),
        })
        external_df = pd.DataFrame({
            "forecast_date": pd.date_range("2025-01-01", "2025-03-01"),
        })
        with pytest.raises(LookAheadLeakageError, match="LEAKAGE"):
            guard.validate_no_future_join(
                main_df, "date", external_df, "forecast_date"
            )

    def test_aligned_dates_pass(self, guard):
        """External data that doesn't exceed main data should pass."""
        dates = pd.date_range("2025-01-01", "2025-01-31")
        main_df = pd.DataFrame({"date": dates})
        external_df = pd.DataFrame({"forecast_date": dates})
        # Should NOT raise
        guard.validate_no_future_join(
            main_df, "date", external_df, "forecast_date"
        )


class TestTrainTestSplitValidation:
    """Test detection of non-temporal train/test splits."""

    def test_overlapping_dates_detected(self, guard):
        """Train data that extends into the test period = leakage."""
        train_dates = pd.Series(pd.date_range("2025-01-01", "2025-06-30"))
        test_dates = pd.Series(pd.date_range("2025-04-01", "2025-12-31"))
        with pytest.raises(LookAheadLeakageError, match="LEAKAGE"):
            guard.validate_train_test_split(train_dates, test_dates)

    def test_proper_split_passes(self, guard):
        """Train strictly before test should pass."""
        train_dates = pd.Series(pd.date_range("2025-01-01", "2025-06-30"))
        test_dates = pd.Series(pd.date_range("2025-07-01", "2025-12-31"))
        # Should NOT raise
        guard.validate_train_test_split(train_dates, test_dates)
