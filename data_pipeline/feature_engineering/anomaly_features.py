"""Demand anomaly features — converts cleaning flags into ML features.

The cleaning pipeline (outage_detector, misrecord_handler) flags anomalous
records and then imputes/corrects them. But the *fact* that a period was
anomalous is itself valuable information for the model.

Use cases:
- COVID lockdowns: store demand drops to near-zero for days/weeks
- Panic buying: demand spikes 5x+ before a lockdown or typhoon
- Post-anomaly recovery: demand overshoots baseline then normalizes

By surfacing these signals as features (rather than silently cleaning them),
the model learns to distinguish normal vs. disrupted demand regimes without
needing hard-coded date ranges (e.g., "COVID period = 2020-2023").

This approach is general-purpose: it handles any cause of demand disruption
(pandemic, typhoon, road closure, competitor opening/closing) without
requiring manual event annotation.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# 输入：含清洗标志列的日粒度 DataFrame → 输出：添加了异常事件特征的 DataFrame
def add_anomaly_features(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add demand anomaly features derived from cleaning-stage flags.

    Expects the DataFrame to contain boolean columns from prior cleaning:
    - ``is_outage``: periods of abnormally low transaction volume
    - ``qty_outlier``: individual records with extreme quantities

    Generates:
    - ``is_demand_anomaly_low``:  outage flag (already exists, aliased)
    - ``is_demand_anomaly_high``: quantity outlier flag (already exists, aliased)
    - ``days_since_last_anomaly``: recency of the most recent anomaly event,
      capturing the recovery curve after disruptions

    All features use only past data (no look-ahead). ``days_since_last_anomaly``
    is computed per (store, SKU) group and capped at 90 to avoid unbounded
    values for long anomaly-free stretches.

    Args:
        df: Daily aggregated DataFrame with cleaning flags preserved.
        date_col: Date column name.
        group_cols: Grouping columns. Default: ``['store_id', 'sku_id']``.

    Returns:
        DataFrame with anomaly feature columns added.
    """
    df = df.copy()
    group_cols = group_cols or ["store_id", "sku_id"]
    available_groups = [c for c in group_cols if c in df.columns]

    # 将清洗阶段的布尔标志转化为 0/1 整数特征，供模型直接使用
    # 低需求异常（宕机）和高需求异常（数量飙升）分开建模，因为成因和影响不同
    df["is_demand_anomaly_low"] = (
        df["is_outage"].astype(int) if "is_outage" in df.columns else 0
    )
    df["is_demand_anomaly_high"] = (
        df["qty_outlier"].astype(int) if "qty_outlier" in df.columns else 0
    )

    # 合并标志：只要当天有任何类型的异常就算
    df["_any_anomaly"] = (
        df["is_demand_anomaly_low"] | df["is_demand_anomaly_high"]
    ).astype(int)

    # "距上次异常天数"——捕捉异常后的恢复曲线
    # 例如台风后第 1 天需求暴增（补货），第 3 天恢复正常
    # 截断到 90 天，避免长期无异常时特征值无限增长
    max_days = 90

    if available_groups:
        df["days_since_last_anomaly"] = (
            df.sort_values(date_col)
            .groupby(available_groups)["_any_anomaly"]
            .transform(_days_since_last_true)
        )
    else:
        df = df.sort_values(date_col)
        df["days_since_last_anomaly"] = _days_since_last_true(df["_any_anomaly"])

    # 截断 + 填充：首次异常之前的记录填为 max_days（表示"很久没有异常"）
    df["days_since_last_anomaly"] = (
        df["days_since_last_anomaly"].clip(upper=max_days).fillna(max_days).astype(int)
    )

    df = df.drop(columns=["_any_anomaly"])

    feature_count = 3  # low, high, days_since
    logger.info("Added %d anomaly features", feature_count)
    return df


# 输入：布尔/0-1 Series → 输出：每个位置距上一个 True 值的距离
def _days_since_last_true(series: pd.Series) -> pd.Series:
    """Compute days since the last True/1 value in a boolean series.

    Returns NaN if no prior anomaly has been seen yet (filled to max later).
    """
    is_anomaly = series.astype(bool)

    # 全无异常时返回全 NaN，后续统一填充为 max_days
    if not is_anomaly.any():
        return pd.Series(float("nan"), index=series.index)

    # cumsum 技巧：每次遇到 True，组编号 +1
    # 然后在每组内用 cumcount 得到组内第几天 → 即距该次异常的天数
    groups = is_anomaly.cumsum()
    result = groups.groupby(groups).cumcount().astype(float)

    # 第一次异常之前的行没有"上一次异常"参考，标记为 NaN
    first_anomaly_pos = is_anomaly.values.argmax()
    if first_anomaly_pos > 0:
        result.iloc[:first_anomaly_pos] = float("nan")

    return result
