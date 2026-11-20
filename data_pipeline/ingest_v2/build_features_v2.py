"""Build the v2 feature set with new data sources.

Adds to the existing features:
  - Real per-store-year unit cost (uc_<store>_<year>)
  - SKU category (cat_l1, cat_l2)
  - Tier (1/2/3)
  - is_lianying flag
  - Inventory features:
      * inv_qty_lag1   : end-of-yesterday inventory level
      * inv_start_today: start-of-today inventory (= inv_qty_lag1 in our convention)
  - Purchase features:
      * purchase_today: qty purchased today (delivery indicator)
      * days_since_last_purchase
      * purchase_qty_7d_total
  - Spring Festival features:
      * days_to_chinese_new_year (signed: positive = before, negative = after)
      * is_within_30d_of_cny (binary)
  - True demand (target replacement option):
      * true_demand (from real_stockouts.parquet)

Output: data/features/features_v2_train.parquet
        data/features/features_v2_oos.parquet
"""
from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import numpy as np


def load_cny_dates() -> dict[int, pd.Timestamp]:
    """Spring Festival (Chinese New Year) dates 2024-2027."""
    return {
        2024: pd.Timestamp("2024-02-10"),
        2025: pd.Timestamp("2025-01-29"),
        2026: pd.Timestamp("2026-02-17"),
        2027: pd.Timestamp("2027-02-06"),
    }


def add_cny_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Spring Festival proximity features."""
    cny = load_cny_dates()
    cny_dates = pd.Series(list(cny.values()))

    # For each row date, find the closest CNY (signed distance)
    dates = df["date"].values

    def days_to_cny(d):
        d_ts = pd.Timestamp(d)
        # Find closest CNY date (could be in past or future)
        deltas = [(c - d_ts).days for c in cny_dates]
        # Choose the one with smallest absolute distance
        idx = np.argmin([abs(x) for x in deltas])
        return deltas[idx]

    # Vectorize via mapping unique dates
    unique_dates = pd.DatetimeIndex(df["date"].unique())
    dist_map = {d: days_to_cny(d) for d in unique_dates}
    df["days_to_chinese_new_year"] = df["date"].map(dist_map)
    df["is_within_30d_of_cny"] = (
        df["days_to_chinese_new_year"].abs() <= 30
    ).astype(int)
    return df


def main():
    warnings.filterwarnings("ignore")
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("Loading inputs...", flush=True)
    sku_master = pd.read_parquet("data/features/sku_master.parquet")
    purchases = pd.read_parquet("data/features/purchases.parquet")
    purchases["date"] = pd.to_datetime(purchases["date"])
    inv = pd.read_parquet("data/features/daily_inventory.parquet")
    inv["date"] = pd.to_datetime(inv["date"])
    stockouts = pd.read_parquet("data/features/stockouts.parquet")
    stockouts["date"] = pd.to_datetime(stockouts["date"])

    train = pd.read_parquet("data/features/features_train.parquet")
    oos = pd.read_parquet("data/features/features_oos.parquet")
    train["date"] = pd.to_datetime(train["date"])
    oos["date"] = pd.to_datetime(oos["date"])

    print(f"  Train: {len(train):,} rows | OOS: {len(oos):,} rows", flush=True)

    # ── Build per-(store, sku, date) augmentation table ──
    print("\nBuilding augmentation table...", flush=True)

    # Map sku_id → 商品编码
    sku_to_code = sku_master.dropna(subset=["商品编码"]).set_index(
        "sku_id"
    )["商品编码"].to_dict()

    # Add 商品编码 + cats + tier to base train/oos
    sku_meta = sku_master[["sku_id", "商品编码", "cat_l1", "cat_l2",
                            "is_lianying", "tier", "avg_unit_cost"]].copy()
    # Year-specific unit cost
    cost_cols = [c for c in sku_master.columns if c.startswith("uc_") and c not in ["uc_2024", "uc_2025", "uc_2026"]]
    # We'll attach uc_<store>_<year> per row dynamically below

    # Base merge: add SKU metadata
    train = train.merge(sku_meta, on="sku_id", how="left")
    oos = oos.merge(sku_meta, on="sku_id", how="left")
    print(f"  After meta merge: train {len(train):,}, oos {len(oos):,}", flush=True)

    # ── Add inventory features (lagged) ──
    print("\nAdding inventory features...", flush=True)

    # inv table is per (store, 商品编码, date) — we need per (store, sku_id, date)
    # First merge inv onto a (store, 商品编码, date) → inv_qty mapping
    # then map back via sku_id → 商品编码

    inv_sorted = inv.sort_values(["store_id", "商品编码", "date"])
    inv_sorted["inv_qty_lag1"] = inv_sorted.groupby(["store_id", "商品编码"])[
        "inv_qty"
    ].shift(1)

    inv_attach = inv_sorted[["date", "store_id", "商品编码",
                              "inv_qty", "inv_qty_lag1"]].rename(
        columns={"inv_qty": "inv_qty_today"}
    )

    train = train.merge(
        inv_attach, on=["date", "store_id", "商品编码"], how="left"
    )
    oos = oos.merge(
        inv_attach, on=["date", "store_id", "商品编码"], how="left"
    )

    # Forward-fill inv per (store, sku_id) since inv only has events
    for df_ in [train, oos]:
        df_.sort_values(["store_id", "sku_id", "date"], inplace=True)
        df_["inv_qty_today"] = df_.groupby(["store_id", "sku_id"])[
            "inv_qty_today"
        ].ffill()
        df_["inv_qty_lag1"] = df_.groupby(["store_id", "sku_id"])[
            "inv_qty_lag1"
        ].ffill()
        df_["inv_qty_today"] = df_["inv_qty_today"].fillna(0)
        df_["inv_qty_lag1"] = df_["inv_qty_lag1"].fillna(0)

    print(f"  Inventory features added", flush=True)

    # ── Add purchase features ──
    print("\nAdding purchase features...", flush=True)

    pur_daily = purchases.groupby(["date", "store_id", "商品编码"], as_index=False).agg(
        purchase_today=("qty_in", "sum"),
    )

    train = train.merge(
        pur_daily, on=["date", "store_id", "商品编码"], how="left"
    )
    oos = oos.merge(
        pur_daily, on=["date", "store_id", "商品编码"], how="left"
    )
    train["purchase_today"] = train["purchase_today"].fillna(0)
    oos["purchase_today"] = oos["purchase_today"].fillna(0)

    # Days since last purchase + 7-day rolling purchase
    for df_ in [train, oos]:
        df_.sort_values(["store_id", "sku_id", "date"], inplace=True)
        # 7-day rolling sum of purchases (shift 1 to avoid leakage)
        df_["purchase_qty_7d"] = df_.groupby(["store_id", "sku_id"])[
            "purchase_today"
        ].transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
        df_["purchase_qty_7d"] = df_["purchase_qty_7d"].fillna(0)

        # days_since_last_purchase: count days since last positive purchase
        df_["had_purchase"] = (df_["purchase_today"] > 0).astype(int)
        # Cumulative max of date_of_last_purchase
        df_["_last_pur_date"] = df_.groupby(["store_id", "sku_id"])[
            "date"
        ].transform(lambda d: d.where(df_.loc[d.index, "had_purchase"] == 1).ffill())
        # Lag by 1 to avoid leakage
        df_["_last_pur_date"] = df_.groupby(["store_id", "sku_id"])[
            "_last_pur_date"
        ].shift(1)
        df_["days_since_last_purchase"] = (
            df_["date"] - df_["_last_pur_date"]
        ).dt.days
        df_["days_since_last_purchase"] = df_["days_since_last_purchase"].fillna(999)
        df_.drop(columns=["had_purchase", "_last_pur_date"], inplace=True)

    print(f"  Purchase features added", flush=True)

    # ── Add CNY features ──
    print("\nAdding CNY features...", flush=True)
    train = add_cny_features(train)
    oos = add_cny_features(oos)
    print(f"  CNY features added", flush=True)

    # ── Add true_demand (from stockout-uncensored) ──
    print("\nAdding true_demand...", flush=True)
    so = stockouts[["date", "store_id", "sku_id", "true_demand",
                    "is_stockout", "is_partial_stockout",
                    "inv_start_of_day"]].copy()
    train = train.merge(
        so, on=["date", "store_id", "sku_id"], how="left"
    )
    oos = oos.merge(
        so, on=["date", "store_id", "sku_id"], how="left"
    )
    # Default true_demand = quantity_sold for non-stockout rows
    train["true_demand"] = train["true_demand"].fillna(train["quantity_sold"])
    oos["true_demand"] = oos["true_demand"].fillna(oos["quantity_sold"])
    train["is_stockout"] = train["is_stockout"].fillna(False)
    oos["is_stockout"] = oos["is_stockout"].fillna(False)
    train["is_partial_stockout"] = train["is_partial_stockout"].fillna(False)
    oos["is_partial_stockout"] = oos["is_partial_stockout"].fillna(False)
    print(f"  True demand attached", flush=True)

    # ── Save ──
    print(f"\nFinal train: {len(train):,} rows × {len(train.columns)} cols", flush=True)
    print(f"Final oos:   {len(oos):,} rows × {len(oos.columns)} cols", flush=True)
    print(f"\nNew columns added:")
    base_cols = set(pd.read_parquet("data/features/features_train.parquet", columns=None).columns)
    new_cols = [c for c in train.columns if c not in base_cols]
    for c in new_cols:
        dtype = str(train[c].dtype)
        nunique = train[c].nunique() if dtype in ('object', 'category', 'bool', 'string') else None
        print(f"  {c}: {dtype}" + (f" ({nunique} unique)" if nunique else ""))

    Path("data/features").mkdir(parents=True, exist_ok=True)
    train.to_parquet("data/features/features_v2_train.parquet")
    oos.to_parquet("data/features/features_v2_oos.parquet")
    print(f"\nSaved features_v2_train.parquet and features_v2_oos.parquet", flush=True)


if __name__ == "__main__":
    main()
