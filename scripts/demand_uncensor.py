"""Demand uncensoring + fair profit comparison.

Stockout days suppress observed sales. This script:
1. Detects stockout days per SKU (active SKU with 0 sales)
2. Estimates true demand on those days using non-stockout averages
3. Computes model profit using uncensored demand (model gets credit for
   filling stockouts that the actual store couldn't)

Usage:
    python -u scripts/demand_uncensor.py
"""
from __future__ import annotations

import sys
import io
import time

import joblib
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def detect_stockouts(df: pd.DataFrame) -> pd.DataFrame:
    """Detect stockout days per SKU-store.

    Two types of stockout:
    1. Full stockout: sales == 0 on a day where the SKU is normally active
    2. Partial stockout: sales < 50% of the SKU's typical daily sales
       (shelf ran low partway through the day)

    Active = SKU sold on at least 30% of its observed days.

    Returns df with added columns:
    - is_stockout: bool (full stockout)
    - is_partial_stockout: bool (partial stockout)
    - sell_rate: fraction of days with positive sales
    - typical_daily_sales: median daily sales on positive days
    """
    # Per SKU-store: how many days have positive sales?
    sku_stats = df.groupby(["store_id", "sku_id"]).agg(
        n_days=("quantity_sold", "count"),
        n_positive=("quantity_sold", lambda x: (x > 0).sum()),
        median_when_positive=("quantity_sold", lambda x: x[x > 0].median() if (x > 0).any() else 0),
    ).reset_index()
    sku_stats["sell_rate"] = sku_stats["n_positive"] / sku_stats["n_days"]

    df = df.merge(
        sku_stats[["store_id", "sku_id", "sell_rate", "median_when_positive"]],
        on=["store_id", "sku_id"], how="left",
    )

    # Full stockout = active SKU with 0 sales
    df["is_stockout"] = (df["sell_rate"] >= 0.30) & (df["quantity_sold"] == 0)

    # Partial stockout = active SKU with sales < 50% of typical
    # (only for SKUs that typically sell >= 3/day, otherwise noise dominates)
    df["is_partial_stockout"] = (
        (df["sell_rate"] >= 0.30)
        & (df["median_when_positive"] >= 3)
        & (df["quantity_sold"] > 0)
        & (df["quantity_sold"] < df["median_when_positive"] * 0.5)
    )

    return df


def uncensor_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate true demand on stockout days.

    Method: for each SKU-store, use the average daily sales on non-stockout days
    (with same day-of-week weighting for seasonality).

    Also uses demand uncensoring formula:
    true_demand = actual_sales × (total_days / on_shelf_days)
    for annual aggregation.
    """
    # Per SKU-store-dayofweek: average sales on non-stockout days
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek

    # Non-stockout average by SKU-store-DOW
    non_stockout = df[~df["is_stockout"] & (df["quantity_sold"] > 0)]
    dow_avg = non_stockout.groupby(["store_id", "sku_id", "dayofweek"])["quantity_sold"].mean()
    dow_avg.name = "dow_avg_sales"

    # Fallback: overall non-stockout average by SKU-store
    overall_avg = non_stockout.groupby(["store_id", "sku_id"])["quantity_sold"].mean()
    overall_avg.name = "overall_avg_sales"

    df = df.merge(dow_avg, on=["store_id", "sku_id", "dayofweek"], how="left")
    df = df.merge(overall_avg, on=["store_id", "sku_id"], how="left")

    # True demand = actual sales on normal days, estimated on stockout days
    df["true_demand"] = df["quantity_sold"].astype(float).copy()

    # Unit price from non-stockout days
    sku_price = non_stockout.copy()
    sku_price["unit_price"] = sku_price["total_amount"] / sku_price["quantity_sold"]
    avg_price = sku_price.groupby(["store_id", "sku_id"])["unit_price"].mean()
    avg_price.name = "avg_unit_price"
    df = df.merge(avg_price, on=["store_id", "sku_id"], how="left")

    # Full stockout: replace 0 with estimated demand
    full_mask = df["is_stockout"]
    estimated_full = df.loc[full_mask, "dow_avg_sales"].fillna(
        df.loc[full_mask, "overall_avg_sales"]
    ).fillna(0)
    df.loc[full_mask, "true_demand"] = estimated_full

    # Partial stockout: replace low sales with estimated demand
    partial_mask = df["is_partial_stockout"]
    estimated_partial = df.loc[partial_mask, "dow_avg_sales"].fillna(
        df.loc[partial_mask, "overall_avg_sales"]
    ).fillna(df.loc[partial_mask, "quantity_sold"])
    # Only replace if estimate > actual (don't reduce demand)
    actual_partial = df.loc[partial_mask, "quantity_sold"]
    df.loc[partial_mask, "true_demand"] = np.maximum(estimated_partial, actual_partial)

    # True revenue
    df["true_revenue"] = df["total_amount"].copy()
    # Full stockout: all revenue is estimated
    df.loc[full_mask, "true_revenue"] = (
        df.loc[full_mask, "true_demand"] * df.loc[full_mask, "avg_unit_price"]
    ).fillna(0)
    # Partial stockout: add the incremental revenue
    df.loc[partial_mask, "true_revenue"] = (
        df.loc[partial_mask, "true_demand"] * df.loc[partial_mask, "avg_unit_price"]
    ).fillna(df.loc[partial_mask, "total_amount"])

    return df


def main() -> None:
    t0 = time.time()

    # Load model
    d = joblib.load("models/step3_results.joblib")
    features = d["features"]
    models = d["quantile_models"]

    # Load OOS data
    oos = pd.read_parquet("data/features/features_oos.parquet")
    for col in features:
        if col in oos.columns and oos[col].dtype == "bool":
            oos[col] = oos[col].astype(int)

    # Known actuals
    actual_revenue = oos["total_amount"].sum()
    actual_cost = float("nan")  # 合作方经营数字，公开版本已移除
    actual_profit = actual_revenue - actual_cost
    cogs_ratio = actual_cost / actual_revenue

    print("=" * 65)
    print("  DEMAND UNCENSORING + PROFIT COMPARISON")
    print("=" * 65)
    print()

    # Step 1: Detect stockouts
    print("Step 1: Detecting stockouts...", flush=True)
    analysis = oos[["store_id", "sku_id", "date", "quantity_sold", "total_amount"]].copy()
    analysis = detect_stockouts(analysis)

    n_full = analysis["is_stockout"].sum()
    n_partial = analysis["is_partial_stockout"].sum()
    n_total = len(analysis)
    n_active_skus = analysis[analysis["sell_rate"] >= 0.30]["sku_id"].nunique()
    print(f"  Total SKU-days: {n_total:,}")
    print(f"  Active SKUs (sell_rate>=30%): {n_active_skus:,}")
    print(f"  Full stockout days (sales=0): {n_full:,} ({n_full/n_total*100:.1f}%)")
    print(f"  Partial stockout days (sales<50% typical): {n_partial:,} ({n_partial/n_total*100:.1f}%)")
    print(f"  Total affected days: {n_full+n_partial:,} ({(n_full+n_partial)/n_total*100:.1f}%)")
    print()

    # Step 2: Uncensor demand
    print("Step 2: Estimating true demand on stockout days...", flush=True)
    analysis = uncensor_demand(analysis)

    censored_demand = analysis["quantity_sold"].sum()
    uncensored_demand = analysis["true_demand"].sum()
    demand_uplift = uncensored_demand - censored_demand
    censored_revenue = analysis["total_amount"].sum()
    uncensored_revenue = analysis["true_revenue"].sum()
    revenue_uplift = uncensored_revenue - censored_revenue

    print(f"  Censored (observed) demand: {censored_demand/1e4:.0f}wan")
    print(f"  Uncensored (true) demand:   {uncensored_demand/1e4:.0f}wan")
    print(f"  Hidden demand (stockouts):  +{demand_uplift/1e4:.0f}wan (+{demand_uplift/censored_demand*100:.1f}%)")
    print()
    print(f"  Censored revenue:   {censored_revenue/1e8:.4f}B")
    print(f"  Uncensored revenue: {uncensored_revenue/1e8:.4f}B")
    print(f"  Hidden revenue:     +{revenue_uplift/1e4:.0f}wan (+{revenue_uplift/censored_revenue*100:.1f}%)")
    print()

    # Step 3: Annual aggregation with uncensored demand
    print("Step 3: Annual aggregation...", flush=True)
    annual = analysis.groupby(["store_id", "sku_id"]).agg(
        actual_sold=("quantity_sold", "sum"),          # censored
        true_demand=("true_demand", "sum"),             # uncensored
        actual_revenue=("total_amount", "sum"),         # censored
        true_revenue=("true_revenue", "sum"),           # uncensored
        n_stockout_days=("is_stockout", "sum"),
        n_days=("is_stockout", "count"),
    ).reset_index()

    annual["unit_price"] = np.where(
        annual["actual_sold"] > 0,
        annual["actual_revenue"] / annual["actual_sold"],
        0,
    )
    # Also compute uncensored unit price (should be similar)
    annual["true_unit_price"] = np.where(
        annual["true_demand"] > 0,
        annual["true_revenue"] / annual["true_demand"],
        0,
    )

    print(f"  SKU-store combos: {len(annual):,}")
    print(f"  With stockouts: {(annual['n_stockout_days']>0).sum():,} ({(annual['n_stockout_days']>0).mean()*100:.1f}%)")
    print()

    # Step 4: Model profit with uncensored demand
    print("Step 4: Computing model profits...", flush=True)
    print()
    print(f"{'':>4} {'q':>5} | {'model_qty':>10} | {'vs_demand':>10} | {'revenue':>10} | {'cost':>10} | {'profit':>10} | {'vs_actual':>12}")
    print("  " + "-" * 85)

    # Actual profit (known)
    print(f"{'':>4} {'REAL':>5} | {'n/a':>10} | {'':>10} | {actual_revenue/1e8:>9.4f}B | {actual_cost/1e8:>9.4f}B | {actual_profit/1e8:>9.4f}B | {'baseline':>12}")

    for q in sorted(models.keys()):
        model = models[q]
        pred = model.predict(oos[features].values)
        daily_qty = np.maximum(0, np.round(pred))

        oos_pred = oos[["store_id", "sku_id"]].copy()
        oos_pred["pred"] = daily_qty

        annual_pred = oos_pred.groupby(["store_id", "sku_id"])["pred"].sum().reset_index()
        annual_pred.columns = ["store_id", "sku_id", "model_demand"]

        merged = annual.merge(annual_pred, on=["store_id", "sku_id"], how="left")
        merged["model_demand"] = merged["model_demand"].fillna(0)
        merged["model_qty"] = np.ceil(merged["model_demand"] * 1.05)

        # KEY CHANGE: model revenue uses TRUE demand, not censored demand
        # Model can sell up to min(model_qty, true_demand)
        # Revenue = min(model_qty, true_demand) × unit_price
        merged["model_can_sell"] = np.minimum(merged["model_qty"], merged["true_demand"])
        merged["model_revenue"] = merged["model_can_sell"] * merged["unit_price"]

        model_total_qty = merged["model_qty"].sum()
        model_revenue = merged["model_revenue"].sum()
        model_cost = model_revenue * cogs_ratio
        model_profit = model_revenue - model_cost

        diff = model_profit - actual_profit
        pct = diff / actual_profit * 100
        demand_ratio = model_total_qty / uncensored_demand

        print(f"{'':>4} q={q:.3f} | {model_total_qty/1e4:>8.0f}wan | {demand_ratio:>9.1%} | {model_revenue/1e8:>9.4f}B | {model_cost/1e8:>9.4f}B | {model_profit/1e8:>9.4f}B | {diff/1e4:>+8.0f}wan ({pct:>+.1f}%)")

    print()

    # Bonus: breakdown of where model profit comes from
    print("=" * 65)
    print("  PROFIT BREAKDOWN (q=0.55)")
    print("=" * 65)
    model = models[0.55]
    pred = model.predict(oos[features].values)
    daily_qty = np.maximum(0, np.round(pred))

    oos_pred = oos[["store_id", "sku_id"]].copy()
    oos_pred["pred"] = daily_qty
    annual_pred = oos_pred.groupby(["store_id", "sku_id"])["pred"].sum().reset_index()
    annual_pred.columns = ["store_id", "sku_id", "model_demand"]

    merged = annual.merge(annual_pred, on=["store_id", "sku_id"], how="left")
    merged["model_demand"] = merged["model_demand"].fillna(0)
    merged["model_qty"] = np.ceil(merged["model_demand"] * 1.05)
    merged["model_can_sell"] = np.minimum(merged["model_qty"], merged["true_demand"])
    merged["model_revenue"] = merged["model_can_sell"] * merged["unit_price"]

    # Revenue from normal sales (same as actual)
    normal_rev = (np.minimum(merged["model_qty"], merged["actual_sold"]) * merged["unit_price"]).sum()
    # Revenue from stockout recovery (extra sales model enables)
    stockout_rev = merged["model_revenue"].sum() - normal_rev

    print(f"  Normal revenue (= actual):  {normal_rev/1e8:.4f}B")
    print(f"  Stockout recovery revenue:  +{stockout_rev/1e4:.0f}wan (+{stockout_rev/normal_rev*100:.1f}%)")
    print(f"  Total model revenue:        {merged['model_revenue'].sum()/1e8:.4f}B")
    print()

    # How many SKUs benefited from stockout recovery?
    merged["recovery"] = merged["model_can_sell"] - np.minimum(merged["model_qty"], merged["actual_sold"])
    recovered_skus = merged[merged["recovery"] > 0]
    print(f"  SKUs with stockout recovery: {len(recovered_skus):,} / {len(merged):,}")
    print(f"  Avg recovery per SKU: {recovered_skus['recovery'].mean():.1f} units")
    print(f"  Top 10 SKUs by recovery revenue:")
    recovered_skus = recovered_skus.copy()
    recovered_skus["recovery_revenue"] = recovered_skus["recovery"] * recovered_skus["unit_price"]
    top10 = recovered_skus.nlargest(10, "recovery_revenue")
    for _, row in top10.iterrows():
        print(f"    {row['sku_id']}: +{row['recovery']:.0f} units, +{row['recovery_revenue']:.0f} yuan ({row['n_stockout_days']:.0f} stockout days)")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
