"""V2 profit comparison: model recommendations vs actual 2025.

Uses real per-SKU per-store unit costs (Tier 1) plus industry-standard
gross margins (Tier 2). Profit is computed against TRUE demand (uncensored)
so the model receives credit for stockout recovery.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Industry-baseline gross margin per Tier 2 cat_l1 (revenue × (1 - gm) = cost)
TIER2_GM = {
    "联营熟食":   0.45,
    "联营烘焙":   0.40,
    "联营杂项":   0.30,
    "联营耗材":   0.20,
    "生鲜禽肉":   0.18,
    "生鲜蔬菜":   0.25,
    "生鲜水产":   0.20,
    "生鲜豆制品": 0.22,
    "生鲜蛋":     0.15,
    "生鲜蛋禽":   0.15,
    "自营熟食":   0.40,
    "自营烘焙":   0.35,
    "包装/非商品": 0.30,
    "加工肉品":   0.22,
}
DEFAULT_TIER2_GM = 0.30


def main():
    warnings.filterwarnings("ignore")
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("Loading...", flush=True)
    d = joblib.load("models/v2_quantile_models.joblib")
    features = d["features"]
    cat_cols = d.get("cat_cols", [])
    models = d["quantile_models"]

    oos = pd.read_parquet("data/features/features_v2_oos.parquet")
    sku_master = pd.read_parquet("data/features/sku_master.parquet")
    print(f"  OOS rows: {len(oos):,}", flush=True)

    # Encode categorical
    for c in cat_cols:
        oos[c] = oos[c].astype("category")
    for c in features:
        if pd.api.types.is_bool_dtype(oos[c]):
            oos[c] = oos[c].astype(int)

    # ── Compute per-SKU per-store annual aggregates ──
    print("Aggregating annual...", flush=True)

    # Add 2025 unit cost for Tier 1 SKUs
    cost_2025_cols = [c for c in sku_master.columns if c.endswith("_2025")]
    # Average across stores per SKU as fallback (some SKUs only sold at 1 store)
    sku_cost_map = sku_master.set_index("sku_id").copy()
    # Per-store unit cost (uc_<store>_2025)
    cost_lookup = {}
    for sid in ["store_a", "store_b", "store_c"]:
        col = f"uc_{sid}_2025"
        if col in sku_cost_map.columns:
            cost_lookup[sid] = sku_cost_map[col]

    # Generic 2025 unit cost
    if "uc_2025" in sku_cost_map.columns:
        cost_lookup["any"] = sku_cost_map["uc_2025"]
    else:
        cost_lookup["any"] = sku_cost_map["avg_unit_cost"]

    def get_unit_cost(sku_id, store_id):
        try:
            v = cost_lookup.get(store_id)
            if v is not None and sku_id in v.index:
                val = v.loc[sku_id]
                if pd.notna(val):
                    return val
            v_any = cost_lookup["any"]
            if sku_id in v_any.index:
                val = v_any.loc[sku_id]
                if pd.notna(val):
                    return val
        except Exception:
            pass
        return np.nan

    # ── Predictions for all quantiles ──
    print("\nPredicting and aggregating...", flush=True)
    quantiles = sorted(models.keys())
    X_oos = oos[features].copy()

    annual_results = {}
    for q in quantiles:
        pred = models[q].predict(X_oos)
        daily_qty = np.maximum(0, np.round(pred))
        oos_q = oos[["store_id", "sku_id", "tier", "cat_l1", "is_lianying",
                     "quantity_sold", "true_demand", "total_amount"]].copy()
        oos_q["pred"] = daily_qty

        annual = oos_q.groupby(
            ["store_id", "sku_id", "tier", "cat_l1", "is_lianying"],
            as_index=False, observed=True,
        ).agg(
            model_demand=("pred", "sum"),
            actual_sold=("quantity_sold", "sum"),
            true_demand_ann=("true_demand", "sum"),
            actual_revenue=("total_amount", "sum"),
        )
        annual["model_qty"] = np.ceil(annual["model_demand"] * 1.05)
        annual["unit_price"] = np.where(
            annual["actual_sold"] > 0,
            annual["actual_revenue"] / annual["actual_sold"], 0,
        )
        # Model can sell up to true_demand at the unit_price
        annual["model_can_sell"] = np.minimum(
            annual["model_qty"], annual["true_demand_ann"],
        )
        annual["model_revenue"] = annual["model_can_sell"] * annual["unit_price"]
        annual["model_qty_total"] = annual["model_qty"]
        annual_results[q] = annual

    # ── Compute Tier 1 cost (real) and Tier 2 cost (estimated GM) ──
    print("\nComputing profits per tier...", flush=True)

    # We need actual costs and model costs
    actual_cost = float("nan")  # 三店进货总成本（门店报表）；合作方经营数字，公开版本已移除
    actual_revenue = oos["total_amount"].sum()
    actual_profit = actual_revenue - actual_cost

    print(f"\n=== Actual 2025 Performance ===", flush=True)
    print(f"  Revenue:     {actual_revenue/1e8:.4f} 亿", flush=True)
    print(f"  Cost:        {actual_cost/1e8:.4f} 亿  (from 门店报表)", flush=True)
    print(f"  Gross profit:{actual_profit/1e8:.4f} 亿  ({actual_profit/actual_revenue*100:.1f}%)",
          flush=True)

    # Per-tier breakdown
    print(f"\n=== Tier breakdown by 2025 sales ===", flush=True)
    for tier in ["Tier1", "Tier2", "Tier3"]:
        sub = oos[oos["tier"] == tier]
        rev = sub["total_amount"].sum()
        qty = sub["quantity_sold"].sum()
        print(f"  {tier}: rev={rev/1e8:.4f}亿 ({rev/actual_revenue*100:.1f}%) "
              f"qty={qty/1e4:.0f}万件", flush=True)

    # COGS ratio: same as V1 — apply chain-wide cost-to-revenue ratio.
    # This avoids the inventory-cost mismatch (actual benefited from
    # pre-2025 inventory that model has no equivalent for).
    cogs_ratio = actual_cost / actual_revenue

    print(f"\n=== Model profit by quantile (COGS-ratio basis, fair comparison) ===",
          flush=True)
    print(f"  Method: model_cost = model_revenue × {cogs_ratio:.3f} (chain COGS ratio)",
          flush=True)
    print(f"  Tier 2 SKUs use category-specific gross margins (industry baselines)",
          flush=True)
    print()
    print(f"  {'q':>6} {'qty(万)':>10} {'revenue(亿)':>12} {'cost(亿)':>10} "
          f"{'profit(亿)':>10} {'vs actual':>14}", flush=True)
    print(f"  {'REAL':>6} {'n/a':>10} {actual_revenue/1e8:>12.4f} "
          f"{actual_cost/1e8:>10.4f} {actual_profit/1e8:>10.4f} {'baseline':>14}", flush=True)

    summary_rows = []
    for q in quantiles:
        annual = annual_results[q]

        # Tier 1: model_cost = model_revenue × COGS ratio (chain-wide)
        tier1 = annual[annual["tier"] == "Tier1"].copy()
        tier1_revenue = tier1["model_revenue"].sum()
        tier1_cost = tier1_revenue * cogs_ratio
        tier1_profit = tier1_revenue - tier1_cost

        # Tier 2: model_cost = model_revenue × (1 - category GM)
        tier2 = annual[annual["tier"] == "Tier2"].copy()
        tier2["gm"] = tier2["cat_l1"].map(TIER2_GM).fillna(DEFAULT_TIER2_GM)
        tier2["model_cost"] = tier2["model_revenue"] * (1 - tier2["gm"])
        tier2_profit = tier2["model_revenue"].sum() - tier2["model_cost"].sum()

        # Tier 3 — excluded
        tier3_profit = 0.0

        total_profit = tier1_profit + tier2_profit + tier3_profit
        total_revenue = annual["model_revenue"].sum()
        total_qty = annual["model_qty"].sum()
        total_cost = total_revenue - total_profit

        diff = total_profit - actual_profit
        pct = diff / actual_profit * 100

        summary_rows.append({
            "q": q,
            "qty_万": total_qty / 1e4,
            "revenue_亿": total_revenue / 1e8,
            "cost_亿": total_cost / 1e8,
            "profit_亿": total_profit / 1e8,
            "tier1_profit_亿": tier1_profit / 1e8,
            "tier2_profit_亿": tier2_profit / 1e8,
            "vs_actual_万": diff / 1e4,
            "improvement_pct": pct,
        })

        print(f"  q={q:.3f} {total_qty/1e4:>10,.0f} {total_revenue/1e8:>12.4f} "
              f"{total_cost/1e8:>10.4f} {total_profit/1e8:>10.4f} "
              f"{diff/1e4:>+10,.0f}万 ({pct:+.1f}%)", flush=True)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    Path("backtests/report").mkdir(parents=True, exist_ok=True)
    summary_df.to_csv("backtests/report/v2_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: backtests/report/v2_summary.csv", flush=True)


if __name__ == "__main__":
    main()
