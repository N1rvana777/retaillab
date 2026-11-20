"""Weather data client — fetches and caches weather for demand features.

Weather significantly affects demand for certain categories:
- Hot days → ice cream, beverages spike
- Rain → umbrella sales up, foot traffic down
- Cold snap → hot pot ingredients, instant noodles up

This module provides a simple interface that returns daily weather features
for a given location and date range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


# 单日天气观测数据的结构化表示
@dataclass
class DailyWeather:
    """Weather observations for a single day."""

    date: date
    temp_max_c: float
    temp_min_c: float
    temp_mean_c: float
    precipitation_mm: float
    humidity_pct: float
    is_rainy: bool
    is_extreme_heat: bool  # > 35°C
    is_extreme_cold: bool  # < 0°C


# 天气CSV文件 → 带衍生特征的 DataFrame
def load_weather_csv(file_path: str) -> pd.DataFrame:
    """Load weather data from a local CSV file.

    Expected columns: date, temp_max, temp_min, precipitation, humidity.
    Additional weather features are derived from these.

    Args:
        file_path: Path to the weather CSV.

    Returns:
        DataFrame with weather features.
    """
    df = pd.read_csv(file_path, parse_dates=["date"])

    # 从最高/最低温度派生均温、温差和极端天气标记
    # 温差大说明天气不稳定，会影响出行意愿和购物行为
    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_mean"] = (df["temp_max"] + df["temp_min"]) / 2
        df["temp_range"] = df["temp_max"] - df["temp_min"]
        # 35°C 以上为极端高温（东北地区夏季偶发），影响冷饮/啤酒需求
        df["is_extreme_heat"] = df["temp_max"] > 35
        # 0°C 以下为极端低温（东北冬季常见），影响热食/方便食品需求
        df["is_extreme_cold"] = df["temp_min"] < 0

    if "precipitation" in df.columns:
        # 降水量 > 1mm 才算有效降雨，微量降水对购物行为影响不大
        df["is_rainy"] = df["precipitation"] > 1.0

    logger.info("Loaded %d days of weather data", len(df))
    return df


# 原始天气数据 → ML可用的滞后/滚动特征
def generate_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Generate ML-ready features from raw weather data.

    Creates lag features (yesterday's weather affects today's shopping)
    and rolling averages for temperature trends.

    Args:
        weather_df: Raw weather DataFrame with date column.

    Returns:
        DataFrame with additional weather feature columns.
    """
    df = weather_df.copy().sort_values("date")

    # 滞后特征：用昨天的天气预测今天的需求
    # shift(1) 保证不使用当天数据，防止数据穿越（这是项目的数学红线）
    for col in ["temp_mean", "precipitation", "humidity"]:
        if col in df.columns:
            df[f"weather_{col}_lag1"] = df[col].shift(1)

    # 3日滚动均温：捕捉温度趋势（连续降温/升温对需求的影响不同于单日变化）
    # 先 shift(1) 再 rolling(3)，确保只使用 t-1, t-2, t-3 的数据
    if "temp_mean" in df.columns:
        df["weather_temp_3d_avg"] = df["temp_mean"].shift(1).rolling(3).mean()

    return df
