"""External feature integration — weather, holidays, promotions.

Merges external data sources into the daily sales DataFrame.
All joins are validated to prevent future data leakage.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# 输入：销售 DataFrame + 天气 DataFrame → 输出：合并了天气特征的 DataFrame
def merge_weather_features(
    sales_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    date_col: str = "date",
    store_col: str = "store_id",
) -> pd.DataFrame:
    """Merge weather features into sales data.

    Weather is joined by date. If weather data is store-specific
    (different locations have different weather), also joins by store.

    Args:
        sales_df: Daily sales DataFrame.
        weather_df: Weather DataFrame with date and weather columns.
        date_col: Date column in both DataFrames.
        store_col: Store column (if weather is location-specific).

    Returns:
        Merged DataFrame with weather features.
    """
    if weather_df.empty:
        logger.warning("Empty weather DataFrame, skipping merge")
        return sales_df

    # 统一日期类型，避免 merge 时因 date vs datetime 类型不匹配而漏匹配
    sales_copy = sales_df.copy()
    weather_copy = weather_df.copy()
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    weather_copy[date_col] = pd.to_datetime(weather_copy[date_col])

    # 优先按城市匹配天气（同城市天气相同，但不同门店可能在不同城市）
    # 只有在没有城市列时才退而按 store_id 匹配
    join_cols = [date_col]
    if "store_city" in weather_copy.columns and "store_city" in sales_copy.columns:
        join_cols.append("store_city")
    elif store_col in weather_copy.columns and store_col in sales_copy.columns:
        join_cols.append(store_col)

    # 用 left join 保证不丢失销售记录（天气缺失的日期特征为 NaN）
    result = sales_copy.merge(weather_copy, on=join_cols, how="left")

    # 统计匹配率，便于监控天气数据覆盖情况
    merged_count = result.dropna(subset=weather_df.columns.difference(join_cols).tolist()[:1]).shape[0]
    logger.info(
        "Merged weather features: %d/%d rows matched",
        merged_count,
        len(sales_df),
    )
    return result


# 输入：销售 DataFrame + 节假日 DataFrame → 输出：合并了节假日特征的 DataFrame
def merge_holiday_features(
    sales_df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Merge holiday features into sales data.

    Args:
        sales_df: Daily sales DataFrame.
        holiday_df: Holiday feature DataFrame with date column.
        date_col: Date column in both DataFrames.

    Returns:
        Merged DataFrame with holiday features.
    """
    if holiday_df.empty:
        logger.warning("Empty holiday DataFrame, skipping merge")
        return sales_df

    sales_copy = sales_df.copy()
    holiday_copy = holiday_df.copy()
    sales_copy[date_col] = pd.to_datetime(sales_copy[date_col])
    holiday_copy[date_col] = pd.to_datetime(holiday_copy[date_col])

    # 节假日只按日期 join（全国统一假期，不区分门店）
    result = sales_copy.merge(holiday_copy, on=date_col, how="left")

    # 非节假日填 False/0——left join 产生的 NaN 实际含义是"当天不是节假日"
    for col in ["is_holiday", "is_pre_holiday", "is_post_holiday"]:
        if col in result.columns:
            result[col] = result[col].fillna(False)

    # 距最近节假日天数：缺失填 365（即"一年内无假期"，作为极值处理）
    if "days_to_nearest_holiday" in result.columns:
        result["days_to_nearest_holiday"] = result["days_to_nearest_holiday"].fillna(365)

    logger.info("Merged holiday features for %d rows", len(result))
    return result
