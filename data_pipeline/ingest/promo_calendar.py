"""Promotion calendar ingestion.

Promotions are the single biggest source of demand spikes in retail.
A well-timed promo can 3-5x normal demand. If the forecasting model
doesn't know about upcoming promotions, it will massively under-predict.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


# 促销日历CSV/Excel → 促销记录 DataFrame
def load_promo_calendar(file_path: str) -> pd.DataFrame:
    """Load promotion schedule from CSV/Excel.

    Expected columns:
    - sku_id: Product identifier
    - store_id: Store identifier (or 'ALL' for chain-wide)
    - start_date: Promotion start
    - end_date: Promotion end
    - promo_type: 'discount', 'bogo', 'bundle', 'clearance'
    - discount_depth: Discount percentage (e.g., 20 for 20% off)

    Args:
        file_path: Path to the promo calendar file.

    Returns:
        DataFrame with promotion records.
    """
    df = pd.read_csv(file_path, parse_dates=["start_date", "end_date"])

    # 必须有SKU和时间范围，否则无法与销售数据关联
    required_cols = {"sku_id", "start_date", "end_date"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns in promo calendar: {missing}"
        raise ValueError(msg)

    logger.info("Loaded %d promotion records", len(df))
    return df


# 销售数据 + 促销日历 → 附加促销特征的销售数据
def generate_promo_features(
    sales_df: pd.DataFrame,
    promo_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Generate promotion features for each (sku, store, date).

    Features:
    - is_on_promo: Binary flag
    - promo_depth: Discount percentage (0 if no promo)
    - days_since_promo_start: How many days into the promo
    - days_until_promo_end: Days remaining in the promo
    - promo_type_encoded: One-hot encoding of promo type

    Args:
        sales_df: Daily sales DataFrame with sku_id, store_id, date.
        promo_df: Promotion calendar.
        date_col: Name of the date column in sales_df.

    Returns:
        sales_df with additional promo feature columns.
    """
    df = sales_df.copy()
    # 初始化所有促销特征列为默认值（无促销状态）
    df["is_on_promo"] = False
    df["promo_depth"] = 0.0
    df["days_into_promo"] = 0
    df["days_until_promo_end"] = 0

    if promo_df.empty:
        return df

    # 逐条促销记录匹配销售数据：找出在促销时间范围内的对应SKU行
    # 用 iterrows 而非 merge，因为促销记录通常不多（百~千条），
    # 而且需要处理 store_id='ALL' 的全门店促销
    for _, promo in promo_df.iterrows():
        mask = (
            (df["sku_id"] == promo["sku_id"])
            & (df[date_col] >= promo["start_date"])
            & (df[date_col] <= promo["end_date"])
        )
        # store_id='ALL' 表示全门店促销，不需要额外过滤门店
        if "store_id" in promo and promo.get("store_id") != "ALL":
            mask = mask & (df["store_id"] == promo["store_id"])

        df.loc[mask, "is_on_promo"] = True
        df.loc[mask, "promo_depth"] = promo.get("discount_depth", 0)

        # 计算促销进度特征：已进行天数和剩余天数
        # 这两个特征很重要，因为促销效果通常呈"前高后低"的衰减曲线
        matched_dates = df.loc[mask, date_col]
        if not matched_dates.empty:
            start = pd.to_datetime(promo["start_date"])
            end = pd.to_datetime(promo["end_date"])
            df.loc[mask, "days_into_promo"] = (
                pd.to_datetime(matched_dates) - start
            ).dt.days
            df.loc[mask, "days_until_promo_end"] = (
                end - pd.to_datetime(matched_dates)
            ).dt.days

    promo_count = df["is_on_promo"].sum()
    logger.info("Marked %d rows as on-promotion", promo_count)
    return df
