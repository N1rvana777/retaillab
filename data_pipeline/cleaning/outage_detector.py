"""System outage detector — distinguishes real zero demand from missing data.

When a POS system goes down, it records zero sales. But zero sales during
peak hours is NOT the same as zero demand. If we train on these false zeros,
the model will systematically under-predict demand.

Detection algorithm:
1. Compute median hourly transaction count for each store.
2. If a store-hour has < 5% of median transactions, flag as potential outage.
3. If consecutive flagged hours span > 2 hours during business hours, confirm outage.
4. Mark all records during outage windows as is_outage=True.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 输入：带时间戳的交易 DataFrame → 输出：添加 is_outage 标志列的 DataFrame
def detect_outages(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    store_col: str = "store_id",
    min_hourly_ratio: float = 0.05,
    business_hours: tuple[int, int] = (8, 22),
    min_consecutive_hours: int = 2,
) -> pd.DataFrame:
    """Detect system outage periods and flag affected records.

    Args:
        df: Transaction-level DataFrame with timestamps.
        timestamp_col: Name of the timestamp column.
        store_col: Name of the store identifier column.
        min_hourly_ratio: Flag if hourly txns < median * ratio.
        business_hours: (open, close) in 24h format.
        min_consecutive_hours: Minimum gap to classify as outage.

    Returns:
        DataFrame with an added 'is_outage' boolean column.
    """
    df = df.copy()
    df["is_outage"] = False

    if df.empty or timestamp_col not in df.columns:
        return df

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["_date"] = df[timestamp_col].dt.date
    df["_hour"] = df[timestamp_col].dt.hour

    open_h, close_h = business_hours
    total_flagged = 0

    # 按门店逐一检测，因为不同门店的正常交易量差异很大
    for store_id, store_df in df.groupby(store_col):
        # 统计每个（日期，小时）的交易笔数
        hourly_counts = (
            store_df
            .groupby(["_date", "_hour"])
            .size()
            .reset_index(name="txn_count")
        )

        # 只看营业时间内的小时——非营业时间本来就没交易，不算宕机
        biz_mask = (hourly_counts["_hour"] >= open_h) & (hourly_counts["_hour"] < close_h)
        biz_counts = hourly_counts[biz_mask]

        if biz_counts.empty:
            continue

        # 用中位数而非均值作为基准，因为中位数不受个别高峰时段影响
        median_hourly = biz_counts["txn_count"].median()
        # 阈值至少为 1，防止中位数为 0 时把所有小时都标记为宕机
        threshold = max(1, median_hourly * min_hourly_ratio)

        # 交易量低于阈值的小时被标记为"疑似宕机"
        low_hours = biz_counts[biz_counts["txn_count"] < threshold]

        if low_hours.empty:
            continue

        # 按日期分组，找连续的低交易小时段
        # 只有连续低交易 >= min_consecutive_hours 才确认为宕机
        # 零星的低交易可能只是淡时，不能误判
        for d, day_low in low_hours.groupby("_date"):
            hours = sorted(day_low["_hour"].tolist())
            gaps = _find_consecutive_gaps(hours, min_consecutive_hours)

            # 将确认的宕机时段内所有交易记录标记为 is_outage=True
            for gap_start, gap_end in gaps:
                mask = (
                    (df[store_col] == store_id)
                    & (df["_date"] == d)
                    & (df["_hour"] >= gap_start)
                    & (df["_hour"] <= gap_end)
                )
                flagged_count = mask.sum()
                df.loc[mask, "is_outage"] = True
                total_flagged += flagged_count

    # 清理临时列
    df = df.drop(columns=["_date", "_hour"], errors="ignore")

    logger.info(
        "Outage detection: flagged %d records across %d stores",
        total_flagged,
        df[store_col].nunique(),
    )
    return df


# 输入：已排序的小时列表 + 最小连续长度 → 输出：满足长度要求的连续时段列表
def _find_consecutive_gaps(
    hours: list[int], min_length: int
) -> list[tuple[int, int]]:
    """Find runs of consecutive hours that are at least min_length long."""
    if not hours:
        return []

    gaps: list[tuple[int, int]] = []
    start = hours[0]
    prev = hours[0]

    for h in hours[1:]:
        if h == prev + 1:
            # 当前小时与上一小时连续，延伸当前段
            prev = h
        else:
            # 连续段断裂，检查已累积的段是否够长
            if prev - start + 1 >= min_length:
                gaps.append((start, prev))
            start = h
            prev = h

    # 处理最后一段（循环结束时不会自动检查）
    if prev - start + 1 >= min_length:
        gaps.append((start, prev))

    return gaps
