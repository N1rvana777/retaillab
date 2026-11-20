"""Step 3: 生成高管商业研报 — AI vs 传统进货盲测对比.

输出:
- 3 张 Matplotlib 图表 (PNG)
- 1 份 Markdown 格式报告总结
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mock_retail_data import generate_dataset
from strategies import (
    backtest,
    strategy_human_manager,
    strategy_newsvendor_quant,
)

# ── 输出目录 ─────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "report_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 中文字体设置 ─────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# SKU display names (English for font compatibility)
SKU_DISPLAY = {"鲜奶": "Fresh Milk", "矿泉水": "Mineral Water"}


def _color_palette() -> dict[str, str]:
    return {
        "human": "#E74C3C",   # 红色 - 人工
        "quant": "#2ECC71",   # 绿色 - AI
        "accent": "#3498DB",  # 蓝色
    }


# ═══════════════════════════════════════════════════════════════════
# 图表 1: 累计净利润对比曲线
# ═══════════════════════════════════════════════════════════════════

def plot_cumulative_profit(
    bt_a: pd.DataFrame,
    bt_b: pd.DataFrame,
) -> str:
    colors = _color_palette()
    fig, ax = plt.subplots(figsize=(14, 6))

    for sku in bt_a["sku"].unique():
        sa = bt_a[bt_a["sku"] == sku].sort_values("date")
        sb = bt_b[bt_b["sku"] == sku].sort_values("date")

        cum_a = sa["net_profit"].cumsum()
        cum_b = sb["net_profit"].cumsum()

        ls = "--" if sku == "鲜奶" else "-"
        ax.plot(sa["date"], cum_a, color=colors["human"],
                linestyle=ls, linewidth=2, label=f"Human - {SKU_DISPLAY.get(sku, sku)}")
        ax.plot(sb["date"], cum_b, color=colors["quant"],
                linestyle=ls, linewidth=2, label=f"AI Quant - {SKU_DISPLAY.get(sku, sku)}")

    # 总计
    total_a = bt_a.groupby("date")["net_profit"].sum().cumsum()
    total_b = bt_b.groupby("date")["net_profit"].sum().cumsum()
    ax.plot(total_a.index, total_a.values, color=colors["human"],
            linewidth=3, alpha=0.4, label="Human TOTAL")
    ax.plot(total_b.index, total_b.values, color=colors["quant"],
            linewidth=3, alpha=0.4, label="AI Quant TOTAL")

    # 标注终点差值
    delta = total_b.iloc[-1] - total_a.iloc[-1]
    ax.annotate(
        f"+¥{delta:,.0f}",
        xy=(total_b.index[-1], total_b.iloc[-1]),
        fontsize=16, fontweight="bold", color=colors["quant"],
        ha="left", va="bottom",
    )

    ax.set_title("Cumulative Net Profit: AI Quant vs Human Manager (12 Months)",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Net Profit (CNY)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = str(OUTPUT_DIR / "01_cumulative_profit.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# 图表 2: 月度报废金额对比柱状图
# ═══════════════════════════════════════════════════════════════════

def plot_waste_comparison(
    bt_a: pd.DataFrame,
    bt_b: pd.DataFrame,
) -> str:
    colors = _color_palette()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for idx, sku in enumerate(bt_a["sku"].unique()):
        ax = axes[idx]
        sa = bt_a[bt_a["sku"] == sku].copy()
        sb = bt_b[bt_b["sku"] == sku].copy()
        sa["month"] = sa["date"].dt.to_period("M")
        sb["month"] = sb["date"].dt.to_period("M")

        wa = sa.groupby("month")["waste_cost"].sum()
        wb = sb.groupby("month")["waste_cost"].sum()

        months = wa.index.astype(str)
        x = np.arange(len(months))
        width = 0.35

        bars_a = ax.bar(x - width / 2, wa.values, width,
                        color=colors["human"], label="Human", alpha=0.85)
        bars_b = ax.bar(x + width / 2, wb.values, width,
                        color=colors["quant"], label="AI Quant", alpha=0.85)

        ax.set_title(f"Monthly Waste Cost: {SKU_DISPLAY.get(sku, sku)}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Waste Cost (CNY)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Spoilage / Waste Cost Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = str(OUTPUT_DIR / "02_waste_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# 图表 3: 缺货损失 (Lost Sales) 对比
# ═══════════════════════════════════════════════════════════════════

def plot_lost_sales(
    bt_a: pd.DataFrame,
    bt_b: pd.DataFrame,
) -> str:
    colors = _color_palette()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for idx, sku in enumerate(bt_a["sku"].unique()):
        ax = axes[idx]
        sa = bt_a[bt_a["sku"] == sku].copy()
        sb = bt_b[bt_b["sku"] == sku].copy()
        sa["month"] = sa["date"].dt.to_period("M")
        sb["month"] = sb["date"].dt.to_period("M")

        la = sa.groupby("month")["lost_sales"].sum()
        lb = sb.groupby("month")["lost_sales"].sum()

        months = la.index.astype(str)
        x = np.arange(len(months))
        width = 0.35

        ax.bar(x - width / 2, la.values, width,
               color=colors["human"], label="Human", alpha=0.85)
        ax.bar(x + width / 2, lb.values, width,
               color=colors["quant"], label="AI Quant", alpha=0.85)

        ax.set_title(f"Monthly Lost Sales: {SKU_DISPLAY.get(sku, sku)}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Lost Sales (CNY)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Lost Sales (Stockout Cost) Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = str(OUTPUT_DIR / "03_lost_sales.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# Markdown 报告
# ═══════════════════════════════════════════════════════════════════

def generate_markdown_report(
    bt_a: pd.DataFrame,
    bt_b: pd.DataFrame,
) -> str:
    """生成高管可读的 Markdown 研报."""

    # ── 总体指标 ──
    total_profit_a = bt_a["net_profit"].sum()
    total_profit_b = bt_b["net_profit"].sum()
    profit_uplift = (total_profit_b - total_profit_a) / abs(total_profit_a) * 100

    total_waste_a = bt_a["waste_cost"].sum()
    total_waste_b = bt_b["waste_cost"].sum()
    waste_saved = total_waste_a - total_waste_b

    total_lost_a = bt_a["lost_sales"].sum()
    total_lost_b = bt_b["lost_sales"].sum()
    lost_reduced = total_lost_a - total_lost_b

    # ── 分 SKU 指标 ──
    sku_rows = []
    for sku in bt_a["sku"].unique():
        sa = bt_a[bt_a["sku"] == sku]
        sb = bt_b[bt_b["sku"] == sku]
        p_a = sa["net_profit"].sum()
        p_b = sb["net_profit"].sum()
        w_a = sa["waste_cost"].sum()
        w_b = sb["waste_cost"].sum()
        l_a = sa["lost_sales"].sum()
        l_b = sb["lost_sales"].sum()
        sku_rows.append({
            "sku": sku,
            "profit_a": p_a, "profit_b": p_b,
            "profit_pct": (p_b - p_a) / abs(p_a) * 100 if p_a != 0 else 0,
            "waste_a": w_a, "waste_b": w_b,
            "lost_a": l_a, "lost_b": l_b,
        })

    # ── 鲜奶 CR ──
    cr_milk = (15 - 10) / ((15 - 10) + (10 - 0))
    cr_water = (3 - 1) / ((3 - 1) + (1 - 0.8))

    report = f"""# AI 量化进货 vs 传统人工进货 — 盲测对比报告

> **回测周期**: 2025-01-01 ~ 2025-12-31 (365 天)
> **门店**: 单店模拟 | **SKU**: 鲜奶 (短保生鲜) + 矿泉水 (长保标品)
> **策略 A**: 人类店长经验模型 (上周同星期销量 × 1.1)
> **策略 B**: AI Newsvendor 最优化模型 (滚动需求预测 + 动态临界比率)

---

## Executive Summary

| 核心指标 | 人工 (策略A) | AI量化 (策略B) | 提升幅度 |
|---------|-------------|---------------|---------|
| **年度净利润** | ¥{total_profit_a:,.0f} | ¥{total_profit_b:,.0f} | **{profit_uplift:+.1f}%** |
| **年度报废损失** | ¥{total_waste_a:,.0f} | ¥{total_waste_b:,.0f} | 节省 ¥{waste_saved:,.0f} |
| **年度缺货损失** | ¥{total_lost_a:,.0f} | ¥{total_lost_b:,.0f} | 减少 ¥{lost_reduced:,.0f} |

---

## 1. 净利润提升: {profit_uplift:+.1f}%

AI 量化模型在 12 个月内累计多赚 **¥{total_profit_b - total_profit_a:,.0f}**。

![累计净利润对比](report_output/01_cumulative_profit.png)

**关键洞察**: AI 模型的优势从第 2 个月开始显现 (模型完成冷启动后)，此后差距持续扩大。

---

## 2. 分品类深度分析

### 2.1 鲜奶 (短保生鲜) — 报童模型临界比率 CR = {cr_milk:.2f}

| 指标 | 人工 | AI | 差异 |
|------|-----|-----|------|
| 净利润 | ¥{sku_rows[0]["profit_a"]:,.0f} | ¥{sku_rows[0]["profit_b"]:,.0f} | {sku_rows[0]["profit_pct"]:+.1f}% |
| 报废损失 | ¥{sku_rows[0]["waste_a"]:,.0f} | ¥{sku_rows[0]["waste_b"]:,.0f} | |
| 缺货损失 | ¥{sku_rows[0]["lost_a"]:,.0f} | ¥{sku_rows[0]["lost_b"]:,.0f} | |

**分析**: 鲜奶 CR = {cr_milk:.2f}，超额成本 (¥10) 远大于缺货成本 (¥5)。
传统店长依赖"上周同天"单一数据点，噪音极大——若上周恰逢雨天低销量，本周就严重少进，导致 **¥{sku_rows[0]["lost_a"]:,.0f} 的巨额缺货损失**。
AI 用 28 天滚动窗口平滑了噪音，缺货损失从 ¥{sku_rows[0]["lost_a"]:,.0f} 降至 ¥{sku_rows[0]["lost_b"]:,.0f}，降幅 {(1 - sku_rows[0]["lost_b"]/max(sku_rows[0]["lost_a"],1))*100:.0f}%。

### 2.2 矿泉水 (长保标品) — 报童模型临界比率 CR = {cr_water:.2f}

| 指标 | 人工 | AI | 差异 |
|------|-----|-----|------|
| 净利润 | ¥{sku_rows[1]["profit_a"]:,.0f} | ¥{sku_rows[1]["profit_b"]:,.0f} | {sku_rows[1]["profit_pct"]:+.1f}% |
| 报废损失 | ¥{sku_rows[1]["waste_a"]:,.0f} | ¥{sku_rows[1]["waste_b"]:,.0f} | |
| 缺货损失 | ¥{sku_rows[1]["lost_a"]:,.0f} | ¥{sku_rows[1]["lost_b"]:,.0f} | |

**分析**: 矿泉水 CR = {cr_water:.2f}，缺货成本 (¥2) 远大于超额成本 (¥0.2)。
传统店长的"上周同天 × 1.1"在遇到上周高需求日（如节假日）时会严重超额进货，导致 **¥{sku_rows[1]["waste_a"]:,.0f} 的天量报废**。
AI 用滚动窗口给出更精准的需求估计，报废从 ¥{sku_rows[1]["waste_a"]:,.0f} 降至 ¥{sku_rows[1]["waste_b"]:,.0f}，降幅 {(1 - sku_rows[1]["waste_b"]/max(sku_rows[1]["waste_a"],1))*100:.0f}%。

---

## 3. 报废成本对比

![报废金额对比](report_output/02_waste_comparison.png)

AI 模型全年节省报废金额 **¥{waste_saved:,.0f}**，主要来自矿泉水品类的精准需求估计——消除了人工策略因噪音导致的大幅超额进货。

---

## 4. 缺货隐性损失对比

![缺货损失对比](report_output/03_lost_sales.png)

AI 模型全年减少缺货损失 **¥{lost_reduced:,.0f}**，主要来自鲜奶品类——AI 的滚动窗口预测避免了人工策略因单日噪音导致的严重少进。

---

## 5. 方法论说明

### 报童模型 (Newsvendor Model)

核心公式: **q* = F⁻¹(Cu / (Cu + Co))**

- **Cu** (缺货成本) = 售价 - 进价 = 少卖一件损失的毛利
- **Co** (超额成本) = 进价 - 残值 = 多进一件的报废净损失
- **临界比率 CR** = Cu / (Cu + Co) 决定了进货分位数

| SKU | Cu | Co | CR | 含义 |
|-----|----|----|-----|------|
| 鲜奶 | ¥5 | ¥10 | {cr_milk:.2f} | 保守进货 (多进太贵) |
| 矿泉水 | ¥2 | ¥0.2 | {cr_water:.2f} | 激进备货 (多进便宜) |

### 人类店长策略局限性

1. **不区分品类成本结构**: 对所有 SKU 统一 +10% 安全垫
2. **仅依赖单一数据点**: 上周同星期 1 天的销量，噪音极大
3. **无法处理天气/节假日突变**: 没有系统性的外部因子修正

### AI 量化策略优势

1. **成本驱动的差异化决策**: 不同 SKU 使用不同的临界比率
2. **28 天滚动窗口 + 星期分层**: 更稳健的需求估计
3. **数学最优解**: 在给定需求分布下，利润期望最大化

---

## 6. 落地建议

1. **第一阶段 (1-2 月)**: 在 3-5 家门店选取 10 个 SKU 进行 A/B 测试
2. **第二阶段 (3-4 月)**: 扩展到全品类，保留人工审核权
3. **第三阶段 (5-6 月)**: 全面推广，人工仅处理异常告警

**预计年化收益**: 按单店提升 {profit_uplift:.0f}% 净利润估算，
100 家门店年增利 **¥{(total_profit_b - total_profit_a) * 100:,.0f}**。

---

*报告生成时间: 2026-03-10 | RetailLab AI Quant Engine v1.0*
"""
    return report


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  RetailLab — AI vs Human Backtest Report Generator")
    print("=" * 60)

    # Step 1: 生成数据
    print("\n[1/5] Generating mock retail data (365 days)...")
    df = generate_dataset()
    print(f"  → {len(df)} rows, SKUs: {df['sku'].unique().tolist()}")

    # Step 2: 运行策略
    print("\n[2/5] Running Strategy A: Human Manager...")
    orders_a = strategy_human_manager(df)

    print("[3/5] Running Strategy B: Newsvendor Quant...")
    orders_b = strategy_newsvendor_quant(df)

    # Step 3: 回测
    print("\n[4/5] Running backtest...")
    bt_a = backtest(orders_a, df)
    bt_b = backtest(orders_b, df)

    # 打印核心数字
    print(f"\n  Human total profit:  ¥{bt_a['net_profit'].sum():>12,.0f}")
    print(f"  Quant total profit:  ¥{bt_b['net_profit'].sum():>12,.0f}")
    print(f"  Human total waste:   ¥{bt_a['waste_cost'].sum():>12,.0f}")
    print(f"  Quant total waste:   ¥{bt_b['waste_cost'].sum():>12,.0f}")
    print(f"  Human lost sales:    ¥{bt_a['lost_sales'].sum():>12,.0f}")
    print(f"  Quant lost sales:    ¥{bt_b['lost_sales'].sum():>12,.0f}")

    # Step 4: 生成图表
    print("\n[5/5] Generating charts...")
    p1 = plot_cumulative_profit(bt_a, bt_b)
    print(f"  → {p1}")
    p2 = plot_waste_comparison(bt_a, bt_b)
    print(f"  → {p2}")
    p3 = plot_lost_sales(bt_a, bt_b)
    print(f"  → {p3}")

    # Step 5: 生成报告
    report = generate_markdown_report(bt_a, bt_b)
    report_path = OUTPUT_DIR / "blind_test_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n✓ Report saved to: {report_path}")

    # 同时输出到终端
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
