"""Lag and rolling window features — the backbone of time-series forecasting.

CRITICAL: ALL lag and rolling features MUST include .shift(1) to prevent
data leakage. Without the shift, today's rolling mean would include today's
sales, which is the very thing we're trying to predict.

    WRONG:  df['rolling_7d_mean'] = df['sales'].rolling(7).mean()
    RIGHT:  df['rolling_7d_mean'] = df['sales'].shift(1).rolling(7).mean()

The shift(1) ensures that at time t, the feature only uses data up to t-1.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# 输入：日粒度 DataFrame → 输出：添加了滞后天数销量特征的 DataFrame
def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "quantity_sold",
    group_cols: list[str] | None = None,
    lag_days: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged demand features.

    Creates features like sales_lag_1 (yesterday's sales),
    sales_lag_7 (same day last week), etc.

    IMPORTANT: Lags are computed AFTER shift(1), meaning lag_1 is
    actually "the day before yesterday" when used for prediction at time t.
    This is by design — lag_1 is safely available at prediction time.

    Args:
        df: Daily aggregated DataFrame, sorted by date.
        target_col: Column to create lags from.
        group_cols: Columns to group by (e.g., ['store_id', 'sku_id']).
        lag_days: Which lag offsets to create. Default: [1, 2, 3, 7, 14, 28].

    Returns:
        DataFrame with lag feature columns.
    """
    df = df.copy()
    # 默认 1-35 天全覆盖，训练后根据特征重要性裁剪无用的
    lag_days = lag_days or list(range(1, 36))
    group_cols = group_cols or ["store_id", "sku_id"]

    # 只用实际存在的列分组，兼容测试数据可能缺少某些列的情况
    available_groups = [c for c in group_cols if c in df.columns]

    for lag in lag_days:
        col_name = f"sales_lag_{lag}"
        if available_groups:
            # shift(lag) 确保在时间 t 只能看到 t-lag 及更早的数据
            # 这是防止数据穿越的核心机制（参见 CLAUDE.md 数学红线）
            df[col_name] = (
                df.groupby(available_groups)[target_col]
                .shift(lag)
            )
        else:
            df[col_name] = df[target_col].shift(lag)

    logger.info("Added %d lag features (days: %s)", len(lag_days), lag_days)
    return df


# 输入：日粒度 DataFrame → 输出：添加了滚动窗口统计特征的 DataFrame
def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "quantity_sold",
    group_cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling window statistics (mean, std, min, max).

    ALL rolling calculations are performed on SHIFTED data (.shift(1))
    to prevent look-ahead bias. At time t, the 7-day rolling mean uses
    data from t-7 to t-1, NOT t-6 to t.

    Args:
        df: Daily aggregated DataFrame, sorted by date.
        target_col: Column to compute rolling stats from.
        group_cols: Grouping columns.
        windows: Window sizes in days. Default: [7, 14, 28].

    Returns:
        DataFrame with rolling feature columns.
    """
    df = df.copy()
    windows = windows or [7, 14, 28]
    group_cols = group_cols or ["store_id", "sku_id"]
    available_groups = [c for c in group_cols if c in df.columns]

    for window in windows:
        prefix = f"rolling_{window}d"

        if available_groups:
            # 关键：先 shift(1) 再 rolling，确保窗口内不包含当天数据
            # 不这样做会导致模型在回测中看起来很好，但上线后预测崩溃
            shifted = df.groupby(available_groups)[target_col].shift(1)
            rolled = shifted.groupby(df[available_groups].apply(tuple, axis=1))
        else:
            shifted = df[target_col].shift(1)
            rolled = shifted

        if available_groups:
            # 分组滚动：先 shift 再按组计算滚动统计
            shifted_series = df.groupby(available_groups)[target_col].shift(1)
            # min_periods=1 允许窗口不满时仍计算（序列开头处）
            df[f"{prefix}_mean"] = shifted_series.rolling(window, min_periods=1).mean()
            # std 至少需要 2 个数据点才有意义
            df[f"{prefix}_std"] = shifted_series.rolling(window, min_periods=2).std()
            df[f"{prefix}_min"] = shifted_series.rolling(window, min_periods=1).min()
            df[f"{prefix}_max"] = shifted_series.rolling(window, min_periods=1).max()
        else:
            df[f"{prefix}_mean"] = shifted.rolling(window, min_periods=1).mean()
            df[f"{prefix}_std"] = shifted.rolling(window, min_periods=2).std()
            df[f"{prefix}_min"] = shifted.rolling(window, min_periods=1).min()
            df[f"{prefix}_max"] = shifted.rolling(window, min_periods=1).max()

    # 每个窗口生成 4 个统计量（mean, std, min, max）
    feature_count = len(windows) * 4
    logger.info(
        "Added %d rolling features (windows: %s × [mean, std, min, max])",
        feature_count,
        windows,
    )
    return df


# 输入：含 lag 和 rolling 特征的 DataFrame → 输出：添加了趋势因子的 DataFrame
def add_trend_features(
    df: pd.DataFrame,
    target_col: str = "quantity_sold",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add trend/momentum features inspired by quantitative trading strategies.

    - Momentum (7d): sales change over 7 days — is demand accelerating or decelerating?
    - Momentum ratio (7d): percentage change — normalized momentum
    - Bollinger position: z-score of yesterday's sales relative to 7-day rolling stats

    All features use shifted data (no look-ahead).

    Args:
        df: DataFrame with lag and rolling features already computed.
        target_col: Sales column.
        group_cols: Grouping columns.

    Returns:
        DataFrame with trend feature columns added.
    """
    df = df.copy()
    group_cols = group_cols or ["store_id", "sku_id"]
    available_groups = [c for c in group_cols if c in df.columns]

    if available_groups:
        grouped = df.groupby(available_groups)[target_col]
        yesterday = grouped.shift(1)
        eight_days_ago = grouped.shift(8)
    else:
        yesterday = df[target_col].shift(1)
        eight_days_ago = df[target_col].shift(8)

    # Momentum：昨天销量 - 8天前销量 = 7天内的绝对变化量
    df["momentum_7d"] = yesterday - eight_days_ago

    # Momentum ratio：7天变化率（除以基期，避免除零用 NA）
    safe_base = eight_days_ago.replace(0, pd.NA)
    df["momentum_7d_ratio"] = (yesterday - eight_days_ago) / safe_base

    # Bollinger position：昨天销量在近7天波动中的相对位置（z-score）
    # 利用已有的 rolling_7d_mean 和 rolling_7d_std（已经是 shifted 的，安全）
    if "rolling_7d_mean" in df.columns and "rolling_7d_std" in df.columns:
        safe_std = df["rolling_7d_std"].replace(0, pd.NA)
        df["bollinger_position"] = (yesterday - df["rolling_7d_mean"]) / safe_std
    else:
        df["bollinger_position"] = pd.NA

    # EMA（指数移动平均）：比 SMA 对近期变化更敏感
    # pandas 的 ewm(span=N) 自动用 α=2/(N+1) 做指数衰减加权
    if available_groups:
        shifted = df.groupby(available_groups)[target_col].shift(1)
    else:
        shifted = df[target_col].shift(1)

    # EMA_7：短期趋势（近一周，对昨天权重 25%）
    df["ema_7"] = shifted.ewm(span=7, min_periods=1).mean()
    # EMA_28：中期趋势（近一个月，对昨天权重 7%）
    # 两者的相对大小 = 金叉/死叉信号，让模型自己学
    df["ema_28"] = shifted.ewm(span=28, min_periods=1).mean()

    feature_count = 5
    logger.info(
        "Added %d trend features (momentum_7d, momentum_7d_ratio, bollinger_position, ema_7, ema_28)",
        feature_count,
    )
    return df


# 输入：含日期列的日粒度 DataFrame → 输出：添加了去年同月同日销量的 DataFrame
def add_yearly_lag(
    df: pd.DataFrame,
    target_col: str = "quantity_sold",
    group_cols: list[str] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add year-over-year lag: same month+day from last year.

    Unlike shift(365) which is off by 1 day in leap years,
    this matches by (month, day) so 2025-03-15 looks up 2024-03-15.

    Special case: Feb 29 (leap year) falls back to Feb 28 of the prior year.

    Args:
        df: Daily aggregated DataFrame with date column.
        target_col: Column to create YoY lag from.
        group_cols: Grouping columns (e.g., ['store_id', 'sku_id']).
        date_col: Name of the date column.

    Returns:
        DataFrame with 'sales_lag_yoy' column.
    """
    df = df.copy()
    group_cols = group_cols or ["store_id", "sku_id"]
    available_groups = [c for c in group_cols if c in df.columns]

    dates = pd.to_datetime(df[date_col])

    # 构建去年同月同日的日期
    # 闰年2月29日 → 去年2月28日（去年没有29号）
    def _same_day_last_year(d: pd.Timestamp) -> pd.Timestamp:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            # 2月29日在非闰年不存在，退到2月28日
            return d.replace(year=d.year - 1, day=28)

    df["_yoy_date"] = dates.map(_same_day_last_year)

    # 构建去年销量的查找表：(store, sku, date) → quantity
    key_cols = available_groups + [date_col]
    lookup = df[key_cols + [target_col]].copy()
    lookup[date_col] = pd.to_datetime(lookup[date_col])

    # 用 merge 匹配：当前行的 _yoy_date = 去年行的 date
    merge_left = available_groups + ["_yoy_date"]
    merge_right = available_groups + [date_col]

    merged = df[merge_left].merge(
        lookup.rename(columns={target_col: "sales_lag_yoy"}),
        left_on=merge_left,
        right_on=merge_right,
        how="left",
    )

    df["sales_lag_yoy"] = merged["sales_lag_yoy"].values
    df = df.drop(columns=["_yoy_date"])

    matched = df["sales_lag_yoy"].notna().sum()
    logger.info(
        "Added YoY lag: %d/%d matched (%.1f%% — first year has no prior data)",
        matched, len(df), matched / max(len(df), 1) * 100,
    )
    return df
