"""Step 1: 构建极度真实的 12 个月单店模拟数据集.

包含:
- 时间维度: 周末效应、中国法定节假日
- 天气维度: 下雨/极端天气导致客流锐减
- 两种典型 SKU: 鲜奶(短保生鲜) + 矿泉水(长保标品)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── 随机种子 ──────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── SKU 成本参数 ─────────────────────────────────────────────────
SKU_PARAMS: dict[str, dict] = {
    "鲜奶": {
        "sell_price": 15.0,
        "cost_price": 10.0,
        "salvage_value": 0.0,   # 过期即报废
        "base_demand_mean": 80,  # 日均基础需求
        "base_demand_std": 18,
        "shelf_life_days": 3,
    },
    "矿泉水": {
        "sell_price": 3.0,
        "cost_price": 1.0,
        "salvage_value": 0.8,   # 可打折出清
        "base_demand_mean": 200,
        "base_demand_std": 40,
        "shelf_life_days": 365,
    },
}

# ── 中国 2025 法定节假日 (映射到模拟年) ─────────────────────────
HOLIDAYS_MMDD: set[str] = {
    "01-01", "01-28", "01-29", "01-30", "01-31", "02-01", "02-02", "02-03",
    "04-04", "04-05", "04-06",
    "05-01", "05-02", "05-03", "05-04", "05-05",
    "06-22",
    "09-29", "09-30", "10-01", "10-02", "10-03", "10-04", "10-05", "10-06",
}


def _is_holiday(dt: pd.Timestamp) -> bool:
    return dt.strftime("%m-%d") in HOLIDAYS_MMDD


def _weather_multiplier(n: int) -> np.ndarray:
    """生成 n 天的天气因子: 晴天1.0, 小雨0.85, 大雨0.65, 极端0.45."""
    weather_codes = RNG.choice(
        [0, 1, 2, 3], size=n, p=[0.60, 0.22, 0.12, 0.06]
    )
    mapping = {0: 1.0, 1: 0.85, 2: 0.65, 3: 0.45}
    return np.array([mapping[c] for c in weather_codes])


def _weekday_multiplier(day_of_week: np.ndarray) -> np.ndarray:
    """周末客流量比工作日高 30-50%."""
    mult = np.ones_like(day_of_week, dtype=float)
    mult[day_of_week == 5] = 1.35  # 周六
    mult[day_of_week == 6] = 1.50  # 周日
    mult[day_of_week == 4] = 1.10  # 周五晚效应
    return mult


def _holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """节假日期间需求激增 40-80%."""
    mult = np.ones(len(dates))
    for i, dt in enumerate(dates):
        if _is_holiday(dt):
            mult[i] = RNG.uniform(1.4, 1.8)
    return mult


def _seasonal_trend(dates: pd.DatetimeIndex) -> np.ndarray:
    """季节性趋势: 夏季矿泉水需求↑, 冬季↓; 鲜奶相对稳定."""
    day_of_year = dates.dayofyear.values
    # 正弦波: 夏季(第180天)达峰值
    return 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)


def generate_dataset(
    start_date: str = "2025-01-01",
    days: int = 365,
) -> pd.DataFrame:
    """生成完整的模拟零售数据集.

    Returns:
        DataFrame with columns:
            date, sku, base_demand, weather_factor, weekday_factor,
            holiday_factor, seasonal_factor, actual_demand,
            sell_price, cost_price, salvage_value
    """
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    weather = _weather_multiplier(days)
    weekday = _weekday_multiplier(dates.dayofweek.values)
    holiday = _holiday_multiplier(dates)
    seasonal = _seasonal_trend(dates)

    rows: list[dict] = []
    for sku_name, params in SKU_PARAMS.items():
        base_demand = RNG.normal(
            params["base_demand_mean"], params["base_demand_std"], size=days
        )
        base_demand = np.maximum(base_demand, 5)  # 最低保底需求

        # 矿泉水受季节影响大, 鲜奶较稳定
        sku_seasonal = seasonal if sku_name == "矿泉水" else np.ones(days)

        actual = (
            base_demand * weather * weekday * holiday * sku_seasonal
            + RNG.normal(0, params["base_demand_std"] * 0.3, size=days)
        )
        actual = np.maximum(np.round(actual), 0).astype(int)

        for i, dt in enumerate(dates):
            rows.append({
                "date": dt,
                "sku": sku_name,
                "base_demand": round(base_demand[i], 1),
                "weather_factor": round(weather[i], 2),
                "weekday_factor": round(weekday[i], 2),
                "holiday_factor": round(holiday[i], 2),
                "seasonal_factor": round(sku_seasonal[i], 3),
                "actual_demand": actual[i],
                "sell_price": params["sell_price"],
                "cost_price": params["cost_price"],
                "salvage_value": params["salvage_value"],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "sku"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"生成数据: {len(df)} 行, SKU: {df['sku'].unique()}")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print("\n各SKU需求统计:")
    print(df.groupby("sku")["actual_demand"].describe().round(1))
    df.to_parquet("data/mock_retail_365d.parquet", index=False)
    print("\n✓ 已保存到 data/mock_retail_365d.parquet")
