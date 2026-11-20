"""Fetch historical weather data from Open-Meteo API for store locations.

Open-Meteo is free, no API key required, and has historical data back to 1940.

Usage:
    python -m tools.fetch_weather
    python -m tools.fetch_weather --start 2024-01-01 --end 2026-03-19
    python -m tools.fetch_weather --output data/raw/weather.csv

Store locations (Liaoning, China):
    - 城市A (City A): 坐标已在公开版本中移除
    - 城市B (City B): 坐标已在公开版本中移除
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

# Store locations
_STORE_LOCATIONS: list[dict[str, str | float]] = [
    {"city": "city_a", "city_cn": "城市A", "lat": 0.0, "lon": 0.0},  # 公开版本已移除真实坐标
    {"city": "city_b", "city_cn": "城市B", "lat": 0.0, "lon": 0.0},  # 公开版本已移除真实坐标
]

# Open-Meteo API parameters
_DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
]


@dataclass
class WeatherFetchConfig:
    """Configuration for weather data fetching."""

    start_date: str = "2024-01-01"
    end_date: str = "2026-03-19"
    output_path: str = "data/raw/weather.csv"
    timezone: str = "Asia/Shanghai"


def fetch_weather_for_location(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Fetch daily weather data from Open-Meteo Archive API.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        timezone: Timezone for the data.

    Returns:
        DataFrame with daily weather data.
    """
    try:
        import requests
    except ImportError:
        logger.error("requests library not installed. Run: pip install requests")
        raise

    # Use archive API for historical data, forecast API for recent/future
    # Open-Meteo archive covers up to ~5 days ago; forecast covers last 3 months + 16 days ahead
    # We'll use both and merge
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(_DAILY_VARIABLES),
        "timezone": timezone,
    }

    logger.info("Fetching weather for (%.2f, %.2f) from %s to %s", lat, lon, start_date, end_date)

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        logger.warning("No daily data returned for (%.2f, %.2f)", lat, lon)
        return pd.DataFrame()

    daily = data["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_max": daily.get("temperature_2m_max"),
        "temp_min": daily.get("temperature_2m_min"),
        "temp_mean": daily.get("temperature_2m_mean"),
        "precipitation": daily.get("precipitation_sum"),
        "humidity": daily.get("relative_humidity_2m_mean"),
        "wind_speed_max": daily.get("wind_speed_10m_max"),
    })

    logger.info("Fetched %d days of weather data", len(df))
    return df


def fetch_all_stores(config: WeatherFetchConfig) -> pd.DataFrame:
    """Fetch weather for all store locations and combine.

    If stores are in the same city, they share weather data.
    If stores are in different cities, weather is fetched per city
    and a store_city column is added.

    Args:
        config: Fetch configuration.

    Returns:
        Combined weather DataFrame for all locations.
    """
    all_dfs: list[pd.DataFrame] = []

    for loc in _STORE_LOCATIONS:
        df = fetch_weather_for_location(
            lat=float(loc["lat"]),
            lon=float(loc["lon"]),
            start_date=config.start_date,
            end_date=config.end_date,
            timezone=config.timezone,
        )
        if not df.empty:
            df["store_city"] = loc["city"]
            df["store_city_cn"] = loc["city_cn"]
            all_dfs.append(df)

        # Be respectful to the free API
        time.sleep(1)

    if not all_dfs:
        logger.error("No weather data fetched for any location")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Derive additional features
    combined["temp_range"] = combined["temp_max"] - combined["temp_min"]
    combined["is_rainy"] = combined["precipitation"] > 1.0
    combined["is_extreme_heat"] = combined["temp_max"] > 35
    combined["is_extreme_cold"] = combined["temp_min"] < -10  # 东北标准：-10°C

    logger.info(
        "Total weather records: %d (%d cities × %d days)",
        len(combined),
        len(_STORE_LOCATIONS),
        len(combined) // max(len(_STORE_LOCATIONS), 1),
    )
    return combined


def main() -> None:
    """CLI entry point for fetching weather data."""
    parser = argparse.ArgumentParser(
        description="从 Open-Meteo 获取城市A/城市B历史天气数据",
    )
    parser.add_argument(
        "--start", default="2024-01-01", help="起始日期 (默认: 2024-01-01)",
    )
    parser.add_argument(
        "--end", default=str(date.today()), help="结束日期 (默认: 今天)",
    )
    parser.add_argument(
        "--output", default="data/raw/weather.csv", help="输出路径 (默认: data/raw/weather.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = WeatherFetchConfig(
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
    )

    df = fetch_all_stores(config)
    if df.empty:
        print("未能获取任何天气数据！")
        sys.exit(1)

    # Save to CSV
    from pathlib import Path
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary = (
        f"\nWeather data saved to {output}\n"
        f"  Date range: {df['date'].min().date()} ~ {df['date'].max().date()}\n"
        f"  Cities: {', '.join(df['store_city'].unique())}\n"
        f"  Total records: {len(df):,}\n"
        f"  Temp range: {df['temp_min'].min():.1f}C ~ {df['temp_max'].max():.1f}C\n"
        f"  Rainy days: {df['is_rainy'].sum()}\n"
        f"  Extreme cold days (<-10C): {df['is_extreme_cold'].sum()}\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
