"""Misrecord detection and correction.

Common data entry errors in POS systems:
1. Duplicate scans: Same item scanned twice within seconds
2. Unit-of-measure errors: Scanning a case (24 units) as 1 item, or vice versa
3. Decimal point errors: 50.0 recorded as 500 or 5.0
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 输入：原始交易 DataFrame → 输出：去除重复扫码后的 DataFrame
def remove_duplicate_scans(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    sku_col: str = "sku_id",
    store_col: str = "store_id",
    qty_col: str = "quantity_sold",
    window_seconds: int = 5,
) -> pd.DataFrame:
    """Remove duplicate scan records within a time window.

    If two identical records (same store, SKU, quantity) occur within
    `window_seconds`, keep only the first one.

    Args:
        df: Transaction DataFrame.
        timestamp_col: Timestamp column.
        sku_col: SKU identifier column.
        store_col: Store identifier column.
        qty_col: Quantity column.
        window_seconds: Deduplication window in seconds.

    Returns:
        Deduplicated DataFrame.
    """
    if df.empty:
        return df

    # 按时间排序，确保同门店同SKU的记录按时序排列，这样 diff 才有意义
    df = df.copy().sort_values(timestamp_col)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # 按门店+SKU分组，计算相邻交易的时间差
    group_cols = [store_col, sku_col]
    available_cols = [c for c in group_cols if c in df.columns]

    if not available_cols:
        return df

    df["_time_diff"] = df.groupby(available_cols)[timestamp_col].diff()
    threshold = pd.Timedelta(seconds=window_seconds)

    # 同时检查数量是否相同——时间接近但数量不同的不算重复扫码
    df["_qty_same"] = df.groupby(available_cols)[qty_col].diff().eq(0)

    # 双条件判定：时间窗口内 + 数量完全相同 → 收银员重复扫码
    dup_mask = (df["_time_diff"] <= threshold) & (df["_qty_same"])
    dup_count = dup_mask.sum()

    df = df[~dup_mask].drop(columns=["_time_diff", "_qty_same"])

    logger.info("Removed %d duplicate scans (within %ds window)", dup_count, window_seconds)
    return df


# 输入：交易 DataFrame → 输出：标记并修正了数量异常值的 DataFrame
def detect_quantity_outliers(
    df: pd.DataFrame,
    qty_col: str = "quantity_sold",
    sku_col: str = "sku_id",
    std_multiplier: float = 5.0,
) -> pd.DataFrame:
    """Detect and correct extreme quantity outliers.

    Flags transactions where quantity exceeds mean + N*std for that SKU.
    Replaces outliers with the SKU's median quantity.

    Args:
        df: Transaction DataFrame.
        qty_col: Quantity column.
        sku_col: SKU identifier column.
        std_multiplier: Flag if qty > mean + multiplier * std.

    Returns:
        DataFrame with 'qty_outlier' flag and corrected quantities.
    """
    df = df.copy()
    df["qty_outlier"] = False
    # 保留原始值用于审计追溯
    df["qty_original"] = df[qty_col]

    if df.empty or qty_col not in df.columns:
        return df

    # 向量化计算每个 SKU 的统计量，避免逐 SKU 循环（性能关键）
    grouped = df.groupby(sku_col)[qty_col]
    stats = grouped.agg(["mean", "std", "median", "count"])
    # 样本量 <10 的 SKU 统计不可靠，跳过；std=0 说明所有值相同，无异常可言
    stats = stats[stats["count"] >= 10]
    stats = stats[stats["std"] > 0]
    # 使用 mean + 5σ 作为阈值（5σ 非常宽松，只抓极端异常）
    stats["threshold"] = stats["mean"] + std_multiplier * stats["std"]

    # 把阈值和中位数 join 回主表，避免逐行循环
    df = df.merge(
        stats[["threshold", "median"]].rename(columns={"median": "_sku_median"}),
        left_on=sku_col, right_index=True, how="left",
    )

    # 超过阈值的记录只标记，不修改原始值——让模型自己决定如何处理
    # 修改原始值会误杀真实的大单（促销、节日、团购），且全店性事件会被大量误标
    outlier_mask = df["threshold"].notna() & (df[qty_col] > df["threshold"])
    df.loc[outlier_mask, "qty_outlier"] = True

    df = df.drop(columns=["threshold", "_sku_median"])

    total_outliers = df["qty_outlier"].sum()
    logger.info("Detected and corrected %d quantity outliers", total_outliers)
    return df


# 输入：交易 DataFrame → 输出：标记并修正了小数点错误的 DataFrame
def detect_decimal_errors(
    df: pd.DataFrame,
    qty_col: str = "quantity_sold",
    sku_col: str = "sku_id",
) -> pd.DataFrame:
    """Detect probable decimal point errors in quantities.

    If a quantity is 10x or 0.1x the typical range for that SKU,
    it's likely a decimal shift error. Auto-correct with a flag.

    Args:
        df: Transaction DataFrame.
        qty_col: Quantity column.
        sku_col: SKU identifier column.

    Returns:
        DataFrame with 'decimal_error' flag and corrected quantities.
    """
    df = df.copy()
    df["decimal_error"] = False

    if df.empty or qty_col not in df.columns:
        return df

    # 用四分位数来定义"正常范围"，比均值±标准差更抗极端值干扰
    grouped = df.groupby(sku_col)[qty_col]
    stats = grouped.agg(
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count",
    )
    # 样本量 <20 的 SKU 四分位不稳定；q75=0 说明该 SKU 大部分时间无销售
    stats = stats[stats["count"] >= 20]
    stats = stats[stats["q75"] > 0]

    df = df.merge(
        stats[["q25", "q75"]],
        left_on=sku_col, right_index=True, how="left",
    )

    has_stats = df["q25"].notna()

    # 10 倍错误检测：值远超 q75 的 8 倍，且除以 10 后落入正常区间
    # 说明很可能是 5 打成了 50（收银员多按一个 0）
    too_high = has_stats & (df[qty_col] > df["q75"] * 8)
    div10_in_range = (df[qty_col] / 10 >= df["q25"]) & (df[qty_col] / 10 <= df["q75"])
    high_fix = too_high & div10_in_range
    df.loc[high_fix, "decimal_error"] = True
    df.loc[high_fix, qty_col] = df.loc[high_fix, qty_col] / 10

    # 0.1 倍错误检测：值远低于 q25 的 15%，且乘以 10 后落入正常区间
    # 说明很可能是 50 打成了 5（少了一个 0）
    too_low = has_stats & (df[qty_col] < df["q25"] * 0.15) & (df[qty_col] > 0)
    mul10_in_range = (df[qty_col] * 10 >= df["q25"]) & (df[qty_col] * 10 <= df["q75"])
    low_fix = too_low & mul10_in_range
    df.loc[low_fix, "decimal_error"] = True
    df.loc[low_fix, qty_col] = df.loc[low_fix, qty_col] * 10

    df = df.drop(columns=["q25", "q75"])

    total = df["decimal_error"].sum()
    logger.info("Detected and corrected %d probable decimal errors", total)
    return df
