"""Discount anomaly detection and correction.

POS systems frequently record incorrect discounts:
- Cashier applies 99% discount by mistake
- System glitch produces negative prices
- Promotional discounts applied outside the promo window
- Employee discounts misclassified as regular sales

Detection: Flag discounts that exceed the rolling 30-day median + 3*IQR
for that SKU, then cross-reference with the promo calendar.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 输入：交易 DataFrame + 可选促销日历 → 输出：标记并修正了折扣异常的 DataFrame
def detect_discount_anomalies(
    df: pd.DataFrame,
    discount_col: str = "discount_pct",
    sku_col: str = "sku_id",
    timestamp_col: str = "timestamp",
    iqr_multiplier: float = 3.0,
    max_discount_pct: float = 80.0,
    promo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect and correct anomalous discounts.

    Args:
        df: Transaction DataFrame.
        discount_col: Column containing discount percentage.
        sku_col: SKU identifier column.
        timestamp_col: Timestamp column.
        iqr_multiplier: Flag discounts beyond median + IQR * multiplier.
        max_discount_pct: Hard cap — any discount above this is suspicious.
        promo_df: Optional promo calendar to validate high discounts.

    Returns:
        DataFrame with 'discount_anomaly' flag and corrected discount values.
    """
    df = df.copy()
    df["discount_anomaly"] = False
    # 保留原始折扣值用于审计
    df["discount_original"] = df.get(discount_col, 0.0)

    if discount_col not in df.columns:
        df[discount_col] = 0.0
        return df

    # 将折扣列统一为数值型，非数值填 0（缺失折扣视为无折扣）
    df[discount_col] = pd.to_numeric(df[discount_col], errors="coerce").fillna(0.0)

    # 规则1：硬性上限 — 超过 80% 的折扣几乎不可能是正常促销
    hard_cap_mask = df[discount_col] > max_discount_pct
    df.loc[hard_cap_mask, "discount_anomaly"] = True

    # 规则2：统计异常 — 用 IQR 方法按 SKU 检测离群折扣
    # 不同 SKU 的正常折扣水平不同（如日用品常打折，奢侈品很少打折）
    for sku_id, sku_df in df.groupby(sku_col):
        discounts = sku_df[discount_col]
        # 折扣值全部相同的 SKU 无法计算 IQR，跳过
        if discounts.nunique() <= 1:
            continue

        q25 = discounts.quantile(0.25)
        q75 = discounts.quantile(0.75)
        iqr = q75 - q25
        upper_fence = q75 + iqr_multiplier * iqr

        if upper_fence <= 0:
            continue

        outlier_mask = (df[sku_col] == sku_id) & (df[discount_col] > upper_fence)
        df.loc[outlier_mask, "discount_anomaly"] = True

    # 与促销日历交叉验证——在促销期内的高折扣是正常的，取消异常标记
    if promo_df is not None and not promo_df.empty:
        df = _validate_against_promo(df, promo_df, discount_col, sku_col, timestamp_col)

    # 将异常折扣替换为该 SKU 的中位数折扣（中位数对极端值稳健）
    anomaly_count = df["discount_anomaly"].sum()
    if anomaly_count > 0:
        sku_medians = df.groupby(sku_col)[discount_col].transform("median")
        df.loc[df["discount_anomaly"], discount_col] = sku_medians[df["discount_anomaly"]]

    # 最终防线：检查是否产生负售价（折扣后价格 < 0 说明数据严重异常）
    if "unit_price" in df.columns:
        effective_price = df["unit_price"] * (1 - df[discount_col] / 100)
        negative_price_mask = effective_price < 0
        df.loc[negative_price_mask, "discount_anomaly"] = True
        # 负售价直接归零折扣，宁可保守也不能让负价流入下游
        df.loc[negative_price_mask, discount_col] = 0.0

    logger.info(
        "Discount anomaly detection: %d anomalies found and corrected", anomaly_count
    )
    return df


# 输入：主表 + 促销日历 → 输出：取消了合理促销折扣的异常标记的 DataFrame
def _validate_against_promo(
    df: pd.DataFrame,
    promo_df: pd.DataFrame,
    discount_col: str,
    sku_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Un-flag discounts that match known promotions."""
    for _, promo in promo_df.iterrows():
        # 找到该 SKU 在促销窗口内被标记为异常的记录
        promo_mask = (
            (df[sku_col] == promo["sku_id"])
            & (df[timestamp_col] >= promo["start_date"])
            & (df[timestamp_col] <= promo["end_date"])
            & (df["discount_anomaly"])
        )
        # 如果实际折扣与促销力度的偏差在 10% 以内，说明这是正常促销行为
        if "discount_depth" in promo:
            promo_depth = promo["discount_depth"]
            within_tolerance = abs(df[discount_col] - promo_depth) < promo_depth * 0.1
            valid_promo_mask = promo_mask & within_tolerance
            df.loc[valid_promo_mask, "discount_anomaly"] = False

    return df
