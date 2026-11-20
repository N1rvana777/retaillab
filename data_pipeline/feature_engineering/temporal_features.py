"""Temporal feature engineering — time-based patterns in demand.

Retail demand has strong temporal patterns:
- Day-of-week: Weekends typically see higher traffic
- Month/season: Ice cream peaks in summer, hot pot in winter
- Payday proximity: Sales spike around 15th and end of month
- Week-of-year: Captures annual seasonality
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 输入：含日期列的 DataFrame → 输出：添加了时间特征列的 DataFrame
def add_temporal_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add time-based features to a daily sales DataFrame.

    All features are deterministic (derived from the date alone),
    so there is no risk of data leakage.

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with additional temporal feature columns.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    # 基础日历特征——模型需要知道"今天是星期几、几月"来捕捉周期性
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon, 6=Sun
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    # 周末标志——零售业周末客流和工作日差异显著
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    # 周期性编码：用 sin/cos 将星期几和月份映射到圆环上
    # 这样周日(6)和周一(0)的距离=1天，而不是整数编码下的 6
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 年份——捕捉长期趋势（年度增长或衰退）
    df["year"] = dt.dt.year

    return df
