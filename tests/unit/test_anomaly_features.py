"""Tests for demand anomaly feature engineering.

Validates that cleaning-stage flags (is_outage, qty_outlier) are correctly
transformed into ML features, and that days_since_last_anomaly tracks
the recovery curve without look-ahead bias.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.feature_engineering.anomaly_features import add_anomaly_features


@pytest.fixture
def daily_df() -> pd.DataFrame:
    """10-day daily data for one store-SKU with anomalies on days 3 and 7."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": "STORE-A",
            "sku_id": "SKU-001",
            "quantity_sold": [10, 12, 11, 0, 8, 9, 10, 50, 11, 10],
            "is_outage": [
                False, False, False, True, False,
                False, False, False, False, False,
            ],
            "qty_outlier": [
                False, False, False, False, False,
                False, False, True, False, False,
            ],
        }
    )


class TestAnomalyFlags:
    """Test binary anomaly flag features."""

    def test_low_anomaly_mirrors_outage(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df)
        assert result["is_demand_anomaly_low"].sum() == 1
        assert result.loc[result["is_outage"], "is_demand_anomaly_low"].iloc[0] == 1

    def test_high_anomaly_mirrors_qty_outlier(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df)
        assert result["is_demand_anomaly_high"].sum() == 1
        assert result.loc[result["qty_outlier"], "is_demand_anomaly_high"].iloc[0] == 1

    def test_no_flags_all_zero(self) -> None:
        """When no cleaning flags exist, anomaly features default to 0."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "store_id": "S1",
                "sku_id": "K1",
                "quantity_sold": [10, 11, 12, 13, 14],
            }
        )
        result = add_anomaly_features(df)
        assert result["is_demand_anomaly_low"].sum() == 0
        assert result["is_demand_anomaly_high"].sum() == 0


class TestDaysSinceAnomaly:
    """Test the days_since_last_anomaly recovery curve feature."""

    def test_days_since_increases_after_anomaly(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df)
        # Day 3 (index 3) is outage → days_since = 0
        # Day 4 (index 4) → days_since = 1
        # Day 5 (index 5) → days_since = 2
        assert result.loc[3, "days_since_last_anomaly"] == 0
        assert result.loc[4, "days_since_last_anomaly"] == 1
        assert result.loc[5, "days_since_last_anomaly"] == 2

    def test_resets_on_new_anomaly(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df)
        # Day 7 (index 7) has qty_outlier → resets to 0
        assert result.loc[7, "days_since_last_anomaly"] == 0
        # Day 8 → 1 again
        assert result.loc[8, "days_since_last_anomaly"] == 1

    def test_before_first_anomaly_is_max(self, daily_df: pd.DataFrame) -> None:
        result = add_anomaly_features(daily_df)
        # Days 0-2 are before the first anomaly → should be capped at 90
        assert result.loc[0, "days_since_last_anomaly"] == 90
        assert result.loc[1, "days_since_last_anomaly"] == 90
        assert result.loc[2, "days_since_last_anomaly"] == 90

    def test_cap_at_90(self) -> None:
        """Values beyond 90 days should be capped."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "store_id": "S1",
                "sku_id": "K1",
                "quantity_sold": range(100),
                "is_outage": [True] + [False] * 99,
                "qty_outlier": [False] * 100,
            }
        )
        result = add_anomaly_features(df)
        assert result["days_since_last_anomaly"].max() == 90

    def test_no_temp_columns_leaked(self, daily_df: pd.DataFrame) -> None:
        """Internal columns should not appear in the output."""
        result = add_anomaly_features(daily_df)
        assert "_any_anomaly" not in result.columns
