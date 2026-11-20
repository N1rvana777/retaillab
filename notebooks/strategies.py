"""Step 2: 两种博弈策略 — 人类店长 vs Quant EV 最大化.

策略 A: 人类店长经验模型
    "明日进货量 = 上周同星期的真实销量 × 1.1"

策略 B: Newsvendor + ML 需求预测
    基于滚动历史拟合正态分布, 叠加报童模型临界比率动态计算最优进货量.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════
# 策略 A: 人类店长 — "上周同星期 × 1.1"
# ═══════════════════════════════════════════════════════════════════

def strategy_human_manager(
    df: pd.DataFrame,
    safety_pad: float = 1.10,
) -> pd.DataFrame:
    """模拟真实店长: 明日进货 = 上周同星期真实销量 × safety_pad.

    前 7 天没有参考数据, 使用前 N 天均值作为冷启动.
    """
    results: list[dict] = []

    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].sort_values("date").reset_index(drop=True)
        demands = sku_df["actual_demand"].values

        for i in range(len(sku_df)):
            if i < 7:
                # 冷启动: 用已有天数的均值
                ref = np.mean(demands[: max(1, i)]) if i > 0 else demands[0]
            else:
                # 上周同一天 (7天前) 的真实销量
                ref = demands[i - 7]

            order_qty = int(np.ceil(ref * safety_pad))
            order_qty = max(order_qty, 1)

            results.append({
                "date": sku_df.loc[i, "date"],
                "sku": sku,
                "order_qty": order_qty,
                "actual_demand": demands[i],
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# 策略 B: Newsvendor + 滚动需求预测
# ═══════════════════════════════════════════════════════════════════

def _compute_critical_ratio(
    sell_price: float,
    cost_price: float,
    salvage_value: float,
) -> float:
    """Cu / (Cu + Co) — 报童模型临界比率.

    Cu = 缺货成本 = sell_price - cost_price (少卖一件损失的毛利)
    Co = 超额成本 = cost_price - salvage_value (多进一件的报废损失)
    """
    cu = sell_price - cost_price    # underage cost
    co = cost_price - salvage_value  # overage cost
    return cu / (cu + co)


def strategy_newsvendor_quant(
    df: pd.DataFrame,
    lookback_window: int = 28,
) -> pd.DataFrame:
    """Newsvendor 最优进货量 = F^{-1}(CR), F ~ N(μ, σ).

    使用滚动窗口估计需求的均值和标准差,
    并按每个 SKU 的成本结构计算不同的临界比率.

    鲜奶: CR ≈ 0.33 → 偏保守 (超额成本高, 报废 = 全额亏损)
    矿泉水: CR ≈ 0.91 → 偏激进 (超额成本低, 有残值)
    """
    results: list[dict] = []

    for sku in df["sku"].unique():
        sku_df = df[df["sku"] == sku].sort_values("date").reset_index(drop=True)
        demands = sku_df["actual_demand"].values

        # 获取成本参数, 计算临界比率
        sell_price = sku_df["sell_price"].iloc[0]
        cost_price = sku_df["cost_price"].iloc[0]
        salvage = sku_df["salvage_value"].iloc[0]
        cr = _compute_critical_ratio(sell_price, cost_price, salvage)

        # 同时利用星期几信息做分层预测
        day_of_week = sku_df["date"].dt.dayofweek.values

        for i in range(len(sku_df)):
            if i < 14:
                # 冷启动: 用已有数据估算
                window = demands[: max(1, i)]
                mu = np.mean(window)
                sigma = max(np.std(window), mu * 0.15)  # 最小波动率15%
            else:
                # 滚动窗口 + 星期几分层
                start = max(0, i - lookback_window)
                window = demands[start:i]

                # 找同星期几的历史数据
                same_dow_mask = day_of_week[start:i] == day_of_week[i]
                if same_dow_mask.sum() >= 3:
                    # 有足够同星期样本: 加权融合
                    mu_dow = np.mean(window[same_dow_mask])
                    mu_all = np.mean(window)
                    mu = 0.6 * mu_dow + 0.4 * mu_all  # 偏重同星期
                    sigma = max(np.std(window), mu * 0.10)
                else:
                    mu = np.mean(window)
                    sigma = max(np.std(window), mu * 0.10)

            # Newsvendor 最优: q* = F^{-1}(CR) where F ~ N(mu, sigma)
            q_star = stats.norm.ppf(cr, loc=mu, scale=sigma)
            order_qty = max(1, int(np.round(q_star)))

            results.append({
                "date": sku_df.loc[i, "date"],
                "sku": sku,
                "order_qty": order_qty,
                "actual_demand": demands[i],
                "mu": round(mu, 1),
                "sigma": round(sigma, 1),
                "critical_ratio": round(cr, 4),
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════════════════════

def backtest(strategy_df: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    """对策略结果进行逐日利润回测.

    计算每一天的:
    - 实际销售量 = min(order_qty, actual_demand)
    - 收入 = 实际销售量 × sell_price
    - 进货成本 = order_qty × cost_price
    - 报废量 = max(order_qty - actual_demand, 0)
    - 报废残值回收 = 报废量 × salvage_value
    - 缺货量 = max(actual_demand - order_qty, 0)
    - 缺货损失 = 缺货量 × (sell_price - cost_price)
    - 净利润 = 收入 - 进货成本 + 残值回收
    """
    # 合并成本参数
    cost_params = cost_df.drop_duplicates("sku")[
        ["sku", "sell_price", "cost_price", "salvage_value"]
    ]
    merged = strategy_df.merge(cost_params, on="sku", how="left")

    merged["sold"] = np.minimum(merged["order_qty"], merged["actual_demand"])
    merged["revenue"] = merged["sold"] * merged["sell_price"]
    merged["cogs"] = merged["order_qty"] * merged["cost_price"]

    merged["waste_qty"] = np.maximum(merged["order_qty"] - merged["actual_demand"], 0)
    merged["waste_cost"] = merged["waste_qty"] * (
        merged["cost_price"] - merged["salvage_value"]
    )
    merged["salvage_income"] = merged["waste_qty"] * merged["salvage_value"]

    merged["stockout_qty"] = np.maximum(
        merged["actual_demand"] - merged["order_qty"], 0
    )
    merged["lost_sales"] = merged["stockout_qty"] * (
        merged["sell_price"] - merged["cost_price"]
    )

    merged["net_profit"] = merged["revenue"] - merged["cogs"] + merged["salvage_income"]

    return merged


if __name__ == "__main__":
    from mock_retail_data import generate_dataset

    df = generate_dataset()

    print("=" * 60)
    print("策略 A: 人类店长 (上周同星期 × 1.1)")
    a = strategy_human_manager(df)
    bt_a = backtest(a, df)

    print("策略 B: Newsvendor Quant")
    b = strategy_newsvendor_quant(df)
    bt_b = backtest(b, df)

    for name, bt in [("人类店长", bt_a), ("Quant", bt_b)]:
        print(f"\n── {name} ──")
        for sku in bt["sku"].unique():
            s = bt[bt["sku"] == sku]
            print(f"  {sku}: 净利润={s['net_profit'].sum():,.0f}, "
                  f"报废={s['waste_cost'].sum():,.0f}, "
                  f"缺货损失={s['lost_sales'].sum():,.0f}")
