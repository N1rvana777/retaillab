"""Missing data imputation for outage periods and gaps.

After outage detection flags false-zero periods, this module fills in
estimated demand values so the forecasting model can train on a complete
dataset without systematic underestimation bias.

Imputation strategy: Use the median of the same day-of-week from the
previous N weeks (default 4). This captures weekly seasonality while
being robust to outliers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 输入：含宕机标志的 DataFrame → 输出：宕机期间销量被历史中位数填补的 DataFrame
def impute_outage_periods(
    df: pd.DataFrame,
    sales_col: str = "quantity_sold",
    date_col: str = "timestamp",
    store_col: str = "store_id",
    sku_col: str = "sku_id",
    outage_col: str = "is_outage",
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """Impute sales during detected outage periods.

    For each outage record, replaces the (presumably zero) quantity
    with the median sales for the same (store, SKU, day-of-week)
    from the previous `lookback_weeks` non-outage weeks.

    Args:
        df: DataFrame with outage flags.
        sales_col: Column to impute.
        date_col: Timestamp column.
        store_col: Store identifier.
        sku_col: SKU identifier.
        outage_col: Boolean flag column (True = outage).
        lookback_weeks: Number of historical weeks to look back.

    Returns:
        DataFrame with imputed values and 'is_imputed' flag.
    """
    df = df.copy()
    df["is_imputed"] = False

    # 没有宕机记录就直接返回，无需填补
    if outage_col not in df.columns or not df[outage_col].any():
        logger.info("No outage records to impute")
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # 提取星期几——用于匹配"同星期"的历史数据（周末和工作日需求模式差异大）
    df["_dow"] = df[date_col].dt.dayofweek
    df["_date"] = df[date_col].dt.date

    # 只用非宕机的记录作为参考基准，避免用假零值去填补其他假零值
    clean_df = df[~df[outage_col]].copy()

    impute_count = 0

    for idx in df[df[outage_col]].index:
        row = df.loc[idx]
        store = row[store_col]
        sku = row[sku_col]
        dow = row["_dow"]
        row_date = row[date_col]

        # 在过去 N 周内找同门店、同 SKU、同星期几的正常销量记录
        # 这样既捕捉了周期性（周几效应），又限制了时间范围（避免过时数据）
        lookback_start = row_date - pd.Timedelta(weeks=lookback_weeks)
        reference = clean_df[
            (clean_df[store_col] == store)
            & (clean_df[sku_col] == sku)
            & (clean_df["_dow"] == dow)
            & (clean_df[date_col] >= lookback_start)
            & (clean_df[date_col] < row_date)
        ]

        # 用中位数而非均值，因为中位数对偶发异常值更稳健
        if not reference.empty:
            imputed_value = reference[sales_col].median()
            df.loc[idx, sales_col] = imputed_value
            df.loc[idx, "is_imputed"] = True
            impute_count += 1

    df = df.drop(columns=["_dow", "_date"], errors="ignore")

    logger.info(
        "Imputed %d outage records using %d-week same-DoW median",
        impute_count,
        lookback_weeks,
    )
    return df
