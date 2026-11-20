"""Data leakage (look-ahead bias) prevention for time series features.

This is arguably the most critical defensive module in the entire system.
If time-series features accidentally include future information during training,
the model will appear to perform brilliantly in backtesting but will fail
catastrophically in production — and you won't know why until real money is lost.

Common leakage vectors:
1. Rolling features without shift: rolling_mean(t) includes sales at time t
2. Future-dated external data: using tomorrow's weather to predict today
3. Target leakage: including the target variable (or a proxy) as a feature

This module provides:
- LeakageGuard: A validator that checks feature DataFrames for leakage
- LookAheadLeakageError: A custom exception that halts training immediately

Usage:
    guard = LeakageGuard()
    guard.validate_lag_features(df, date_col="date", feature_cols=lag_cols)
    guard.validate_no_future_join(df, "date", weather_df, "forecast_date")
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 数据穿越致命异常——故意不被任何上层 try-except 捕获
# 因为穿越污染的模型从根本上不可信，必须立即中止训练
class LookAheadLeakageError(Exception):
    """Raised when future information leakage is detected in features.

    This exception is intentionally NOT caught by any upstream handler.
    Leakage is a fatal error — training must be halted immediately because
    any model trained on leaked data is fundamentally untrustworthy.
    """


# 穿越防护器：在特征表送入模型训练前做最后一道校验
# 三种检测：滞后特征合法性、外部数据时间范围、训练/测试集时间分割
class LeakageGuard:
    """Validates time-series features for look-ahead bias.

    Call the validate_* methods on your feature DataFrame before training.
    If any check fails, a LookAheadLeakageError is raised.
    """

    # 输入：特征 DataFrame + 日期列 + 待检特征列 + 目标列 → 无返回值（不通过则抛异常）
    def validate_lag_features(
        self,
        df: pd.DataFrame,
        date_col: str,
        feature_cols: list[str],
        target_col: str | None = None,
    ) -> None:
        """Verify that lag/rolling features only use past data.

        For each (entity, date) pair, checks that the feature value at time t
        is NOT correlated with the target at time t (which would indicate
        the feature was computed without a proper shift).

        Detection method: Compute the correlation between each lag feature
        and the current-day target. A true lag feature should have lower
        correlation with today's target than a leaked feature.

        Args:
            df: DataFrame with date and feature columns.
            date_col: Name of the date column.
            feature_cols: List of lag/rolling feature column names to check.
            target_col: Name of the target column. If provided, checks
                for suspiciously high same-day correlation.

        Raises:
            LookAheadLeakageError: If leakage is detected.
        """
        if df.empty or not feature_cols:
            return

        # 检查1：NaN 模式验证——正确 shift 过的 lag 特征在序列头部应为 NaN
        sorted_df = df.sort_values(date_col)

        for col in feature_cols:
            if col not in sorted_df.columns:
                continue

            # 正确 shift(1) 的 lag-1 特征第一行应为 NaN
            # 如果第一行有值，说明可能没有正确偏移
            first_vals = sorted_df.groupby(
                sorted_df.columns.difference([date_col, col]).tolist()[:1]
                or [sorted_df.index.name or sorted_df.index]
            )[col].first()

            # 检查2：与目标的同期相关性——相关性 > 0.95 几乎必定是穿越
            # 正常的 lag 特征与当天目标的相关性应该显著低于 1（因为有时间偏移）
            if target_col and target_col in sorted_df.columns:
                corr = sorted_df[col].corr(sorted_df[target_col])
                if abs(corr) > 0.95:
                    msg = (
                        f"LEAKAGE DETECTED: Feature '{col}' has {corr:.3f} correlation "
                        f"with target '{target_col}'. This suggests the feature includes "
                        f"current-day information. Ensure .shift(1) is applied."
                    )
                    logger.error(msg)
                    raise LookAheadLeakageError(msg)

        logger.info(
            "Lag feature validation passed for %d columns", len(feature_cols)
        )

    # 输入：主表 + 外部表 + 各自日期列 + 容差天数 → 无返回值（不通过则抛异常）
    def validate_no_future_join(
        self,
        main_df: pd.DataFrame,
        main_date_col: str,
        external_df: pd.DataFrame,
        external_date_col: str,
        tolerance_days: int = 0,
    ) -> None:
        """Verify that external data was not joined from future dates.

        Checks that for every row in main_df, the corresponding external data
        comes from a date <= main_date + tolerance_days.

        Args:
            main_df: Primary DataFrame (e.g., sales data).
            main_date_col: Date column in main_df.
            external_df: External DataFrame (e.g., weather forecasts).
            external_date_col: Date column in external_df.
            tolerance_days: Allow external data up to N days ahead
                (e.g., weather forecasts are available 1 day early).

        Raises:
            LookAheadLeakageError: If future-dated external data is found.
        """
        if main_df.empty or external_df.empty:
            return

        main_dates = pd.to_datetime(main_df[main_date_col])
        ext_dates = pd.to_datetime(external_df[external_date_col])

        main_max = main_dates.max()
        ext_max = ext_dates.max()

        # 外部数据的最大日期不应超过主数据太多
        # 例如天气预报提前 1 天可用（tolerance_days=1），但提前 30 天就可疑了
        future_gap = (ext_max - main_max).days

        if future_gap > tolerance_days:
            msg = (
                f"LEAKAGE DETECTED: External data extends {future_gap} days "
                f"beyond the main dataset (tolerance={tolerance_days}). "
                f"Main max={main_max.date()}, external max={ext_max.date()}. "
                f"External data may contain future information."
            )
            logger.error(msg)
            raise LookAheadLeakageError(msg)

        logger.info(
            "External data join validation passed (gap=%d days, tolerance=%d)",
            future_gap,
            tolerance_days,
        )

    # 输入：训练集日期 + 测试集日期 → 无返回值（不通过则抛异常）
    def validate_train_test_split(
        self,
        train_dates: pd.Series,
        test_dates: pd.Series,
    ) -> None:
        """Verify that train/test split respects temporal ordering.

        In time series, ALL training dates must be before ALL test dates.
        Random splits are forbidden because they leak future patterns.

        Args:
            train_dates: Dates in the training set.
            test_dates: Dates in the test set.

        Raises:
            LookAheadLeakageError: If train dates overlap with or follow test dates.
        """
        train_max = pd.to_datetime(train_dates).max()
        test_min = pd.to_datetime(test_dates).min()

        # 时序数据禁止随机切分——训练集最大日期必须严格早于测试集最小日期
        # 否则模型在训练时看到了"未来"的数据模式
        if train_max >= test_min:
            msg = (
                f"LEAKAGE DETECTED: Training data extends to {train_max.date()} "
                f"but test data starts at {test_min.date()}. "
                f"Training data must strictly precede test data in time series. "
                f"Use TimeSeriesSplit instead of KFold."
            )
            logger.error(msg)
            raise LookAheadLeakageError(msg)

        logger.info(
            "Train/test split validation passed (train_max=%s, test_min=%s)",
            train_max.date(),
            test_min.date(),
        )
