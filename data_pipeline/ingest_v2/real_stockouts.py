"""Detect TRUE stockouts using reconstructed daily inventory.

Replaces the heuristic stockout detector (which guessed based on
sales-rate thresholds) with one based on reconstructed inventory
levels from purchase + snapshot data.

Stockout definition:
  - "out_of_stock" = reconstructed inv_qty == 0 at end of day
  - "real_stockout" = out_of_stock AND the SKU is "active"
                      (sold on >=20% of days in surrounding 30-day window)
  - "partial_stockout" = inv at start of day is small (<= median daily sales / 2)
                         AND sales are below typical (probably ran out mid-day)

Output: data/features/stockouts.parquet
    date, store_id, 商品编码, sku_id, quantity_sold, inv_qty,
    is_stockout, is_partial_stockout, true_demand
"""
from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import numpy as np


def main():
    warnings.filterwarnings("ignore")
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("Loading inputs...", flush=True)
    inv = pd.read_parquet("data/features/daily_inventory.parquet")
    inv["date"] = pd.to_datetime(inv["date"])

    sku_master = pd.read_parquet("data/features/sku_master.parquet")
    code_to_sku_id = sku_master.dropna(subset=["商品编码"]).set_index(
        "商品编码"
    )["sku_id"].to_dict()

    sales_train = pd.read_parquet(
        "data/features/features_train.parquet",
        columns=["date", "store_id", "sku_id", "quantity_sold"],
    )
    sales_oos = pd.read_parquet(
        "data/features/features_oos.parquet",
        columns=["date", "store_id", "sku_id", "quantity_sold"],
    )
    sales = pd.concat([sales_train, sales_oos], ignore_index=True)
    sales["date"] = pd.to_datetime(sales["date"])
    print(f"  Sales rows: {len(sales):,}", flush=True)

    # Map sku_id → 商品编码
    sku_id_to_code = sku_master.dropna(subset=["商品编码"]).set_index(
        "sku_id"
    )["商品编码"].to_dict()
    sales["商品编码"] = sales["sku_id"].map(sku_id_to_code)
    sales = sales[sales["商品编码"].notna()].copy()
    sales["quantity_sold"] = sales["quantity_sold"].fillna(0).clip(lower=0)
    print(f"  Sales mapped to 商品编码: {len(sales):,}", flush=True)

    # ── Merge sales with reconstructed inventory ──
    print("\nMerging sales with reconstructed inventory...", flush=True)
    sales_inv = sales.merge(
        inv, on=["date", "store_id", "商品编码"], how="left"
    )
    # Forward-fill inv from previous activity day within (store, code)
    sales_inv = sales_inv.sort_values(["store_id", "商品编码", "date"])
    sales_inv["inv_qty"] = sales_inv.groupby(["store_id", "商品编码"])[
        "inv_qty"
    ].ffill()
    sales_inv["inv_qty"] = sales_inv["inv_qty"].fillna(0)

    # Compute "start of day" inventory = inv at end of previous day
    sales_inv["inv_start_of_day"] = sales_inv.groupby(
        ["store_id", "商品编码"]
    )["inv_qty"].shift(1).fillna(0)
    print(f"  Merged rows: {len(sales_inv):,}", flush=True)

    # ── Compute SKU activity (sell-rate over rolling 30 days) ──
    print("\nComputing SKU activity rates...", flush=True)
    # Per (store, code): fraction of days in observation window with positive sales
    sku_stats = sales_inv.groupby(["store_id", "商品编码"]).agg(
        n_days=("quantity_sold", "count"),
        n_positive=("quantity_sold", lambda x: (x > 0).sum()),
        median_when_positive=("quantity_sold",
                              lambda x: x[x > 0].median() if (x > 0).any() else 0),
    ).reset_index()
    sku_stats["sell_rate"] = sku_stats["n_positive"] / sku_stats["n_days"]
    sku_stats["is_active"] = sku_stats["sell_rate"] >= 0.20  # active = sells >=20% of days
    print(f"  SKU-store pairs: {len(sku_stats):,}, active: "
          f"{sku_stats['is_active'].sum():,} ({sku_stats['is_active'].mean()*100:.1f}%)",
          flush=True)

    sales_inv = sales_inv.merge(
        sku_stats[["store_id", "商品编码", "sell_rate", "median_when_positive",
                   "is_active"]],
        on=["store_id", "商品编码"],
    )

    # ── Detect stockouts ──
    print("\nDetecting stockouts (real, based on reconstructed inv)...", flush=True)

    # Helper: mark "actively-selling window" — has sales in surrounding ±30 days.
    # This filters out SKUs that were delisted (no sales for months) so we
    # don't count them as stocked-out forever.
    sales_inv = sales_inv.sort_values(["store_id", "商品编码", "date"]).reset_index(drop=True)
    sales_inv["has_sale"] = (sales_inv["quantity_sold"] > 0).astype(int)
    # Rolling window: any sale in ±30 days
    grp = sales_inv.groupby(["store_id", "商品编码"])
    sales_inv["sales_in_30d_before"] = grp["has_sale"].transform(
        lambda x: x.rolling(window=30, min_periods=1).sum().shift(1).fillna(0)
    )
    sales_inv["sales_in_30d_after"] = grp["has_sale"].transform(
        lambda x: x.iloc[::-1].rolling(window=30, min_periods=1).sum().iloc[::-1].shift(-1).fillna(0)
    )
    sales_inv["in_active_window"] = (
        (sales_inv["sales_in_30d_before"] > 0) | (sales_inv["sales_in_30d_after"] > 0)
    )

    # Filter: only trust reconstruction when units appear to match between
    # purchase and sales records. Heuristic: the maximum reconstructed
    # inventory should be at least a typical day's sales worth.
    # If max_inv < median_when_positive (one day's sales), reconstruction
    # is broken (likely case-vs-piece unit mismatch in packaged goods).
    max_inv_per_pair = sales_inv.groupby(["store_id", "商品编码"])[
        "inv_qty"
    ].transform("max")
    sales_inv["recon_trustworthy"] = (
        max_inv_per_pair >= sales_inv["median_when_positive"]
    ) & (max_inv_per_pair > 0)

    # Real stockout: inv_start_of_day == 0, SKU is active, in active window,
    # AND reconstruction is trustworthy.
    sales_inv["is_stockout"] = (
        (sales_inv["inv_start_of_day"] <= 0) &
        sales_inv["is_active"] &
        sales_inv["in_active_window"] &
        sales_inv["recon_trustworthy"]
    )

    # Partial stockout: inv was low at start-of-day (less than half daily median)
    # AND we sold less than half the typical amount
    sales_inv["is_partial_stockout"] = (
        (sales_inv["inv_start_of_day"] > 0) &
        (sales_inv["inv_start_of_day"] < sales_inv["median_when_positive"] * 0.5) &
        (sales_inv["quantity_sold"] > 0) &
        (sales_inv["quantity_sold"] < sales_inv["median_when_positive"] * 0.5) &
        sales_inv["is_active"] &
        (sales_inv["median_when_positive"] >= 3)
    )

    # ── Estimate true demand on stockout days ──
    print("Estimating true demand on stockout/partial days...", flush=True)
    sales_inv["dow"] = sales_inv["date"].dt.dayofweek

    # DOW average sales (non-stockout days only)
    non_stockout = sales_inv[
        ~sales_inv["is_stockout"] & ~sales_inv["is_partial_stockout"] &
        (sales_inv["quantity_sold"] > 0)
    ]
    dow_mean = non_stockout.groupby(["store_id", "商品编码", "dow"])[
        "quantity_sold"
    ].mean().rename("dow_mean")
    overall_mean = non_stockout.groupby(["store_id", "商品编码"])[
        "quantity_sold"
    ].mean().rename("overall_mean")

    sales_inv = sales_inv.merge(
        dow_mean, on=["store_id", "商品编码", "dow"], how="left"
    )
    sales_inv = sales_inv.merge(
        overall_mean, on=["store_id", "商品编码"], how="left"
    )

    # True demand = MAX(actual, imputed) — never decreases observed sales.
    sales_inv["true_demand"] = sales_inv["quantity_sold"].astype(float)

    # Full stockout: imputed = DOW mean (else overall mean else 0)
    full_mask = sales_inv["is_stockout"]
    imputed = sales_inv.loc[full_mask, "dow_mean"].fillna(
        sales_inv.loc[full_mask, "overall_mean"]
    ).fillna(0)
    # Always keep max of actual and imputed (cases where partial delivery
    # happened mid-day → actual sales > 0 even though inv started at 0)
    sales_inv.loc[full_mask, "true_demand"] = np.maximum(
        sales_inv.loc[full_mask, "quantity_sold"], imputed,
    )

    # Partial stockout: same max() rule
    partial_mask = sales_inv["is_partial_stockout"]
    imputed_partial = sales_inv.loc[partial_mask, "dow_mean"].fillna(
        sales_inv.loc[partial_mask, "overall_mean"]
    ).fillna(sales_inv.loc[partial_mask, "quantity_sold"])
    sales_inv.loc[partial_mask, "true_demand"] = np.maximum(
        sales_inv.loc[partial_mask, "quantity_sold"], imputed_partial,
    )

    # ── Summary ──
    n_full = sales_inv["is_stockout"].sum()
    n_partial = sales_inv["is_partial_stockout"].sum()
    n_total = len(sales_inv)
    censored_q = sales_inv["quantity_sold"].sum()
    uncensored_q = sales_inv["true_demand"].sum()

    print()
    print("=" * 60)
    print("  REAL STOCKOUT SUMMARY")
    print("=" * 60)
    print(f"Total SKU-days:         {n_total:,}")
    print(f"Full stockouts:         {n_full:,} ({n_full/n_total*100:.2f}%)")
    print(f"Partial stockouts:      {n_partial:,} ({n_partial/n_total*100:.2f}%)")
    print(f"Total affected:         {n_full+n_partial:,} "
          f"({(n_full+n_partial)/n_total*100:.2f}%)")
    print()
    print(f"Censored (observed) demand:  {censored_q/1e4:>10,.0f}万 件")
    print(f"Uncensored (true) demand:    {uncensored_q/1e4:>10,.0f}万 件")
    print(f"Hidden demand (lost sales):  +{(uncensored_q-censored_q)/1e4:>10,.0f}万 件 "
          f"(+{(uncensored_q-censored_q)/censored_q*100:.1f}%)")

    # ── Save ──
    output = sales_inv[[
        "date", "store_id", "sku_id", "商品编码", "quantity_sold",
        "inv_qty", "inv_start_of_day", "is_stockout", "is_partial_stockout",
        "true_demand",
    ]].copy()

    Path("data/features").mkdir(parents=True, exist_ok=True)
    output.to_parquet("data/features/stockouts.parquet")
    print(f"\nSaved: data/features/stockouts.parquet ({len(output):,} rows)",
          flush=True)


if __name__ == "__main__":
    main()
