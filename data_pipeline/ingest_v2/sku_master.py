"""Build the unified SKU master table.

Combines:
  - POS sku_id ↔ barcode (from raw_pos.parquet)
  - Purchase 商品编码 ↔ barcode ↔ 商品名称 (from purchases.parquet)
  - Per-store per-year average unit cost (from purchases.parquet)
  - SKU classification (from sku_categories.classify_sku)
  - Tier assignment (Tier 1/2/3)

Output: data/features/sku_master.parquet
    sku_id (POS) | 商品编码 | barcode | 商品名称 | sku_name (POS)
    cat_l1 | cat_l2 | is_lianying | tier
    avg_unit_cost (overall) | uc_2024 | uc_2025 | uc_2026
    uc_<store>_<year> for per-store per-year unit costs
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

from data_pipeline.ingest_v2.sku_categories import classify_sku


LIANYING_CATEGORIES = {"联营熟食", "联营烘焙", "联营杂项", "联营耗材"}


def strip_pos_barcode(bc: str | float | None) -> str | None:
    """Strip alphabetic suffix from POS barcode (variant codes like '1234A')."""
    if pd.isna(bc):
        return None
    bc_str = str(bc).strip()
    m = re.match(r"^(\d{8,})[a-zA-Z]+$", bc_str)
    return m.group(1) if m else bc_str


def assign_tier(row) -> str:
    if row["cat_l1"] == "其他":
        return "Tier3"
    if row["is_lianying"] or row["cat_l1"] in LIANYING_CATEGORIES:
        return "Tier2"
    if pd.notna(row["商品编码"]):
        return "Tier1"
    # POS-only with classifiable cat → Tier 2 (estimated)
    return "Tier2"


def build_sku_master() -> pd.DataFrame:
    print("Building SKU master...", flush=True)

    # 1. POS-side master
    raw = pd.read_parquet(
        "data/raw/raw_pos.parquet",
        columns=["sku_id", "sku_name", "barcode"],
    )
    pos = raw.drop_duplicates("sku_id").copy()
    pos["bc_clean"] = pos["barcode"].apply(strip_pos_barcode)
    pos = pos.rename(columns={"barcode": "pos_barcode"})
    print(f"  POS unique SKUs: {len(pos):,}", flush=True)

    # 2. Purchase-side master with cost data
    purchases = pd.read_parquet("data/features/purchases.parquet")
    purchases["year"] = purchases["date"].dt.year

    # Purchase master: one row per 商品编码 with name + barcode
    pur_master = purchases.drop_duplicates("商品编码")[
        ["商品编码", "barcode", "商品名称"]
    ].copy()
    print(f"  Purchase unique SKUs: {len(pur_master):,}", flush=True)

    # 3. Per-store per-year average unit cost (cost_in / qty_in, weighted by volume)
    cost_per_store_year = purchases.groupby(["商品编码", "store_id", "year"]).agg(
        qty_in=("qty_in", "sum"),
        cost_in=("cost_in", "sum"),
    ).reset_index()
    cost_per_store_year["unit_cost"] = (
        cost_per_store_year["cost_in"] / cost_per_store_year["qty_in"].replace(0, np.nan)
    )
    cost_per_store_year = cost_per_store_year.dropna(subset=["unit_cost"])

    # Per-year only (across stores)
    cost_per_year = purchases.groupby(["商品编码", "year"]).agg(
        qty_in=("qty_in", "sum"),
        cost_in=("cost_in", "sum"),
    ).reset_index()
    cost_per_year["unit_cost"] = (
        cost_per_year["cost_in"] / cost_per_year["qty_in"].replace(0, np.nan)
    )
    cost_per_year = cost_per_year.dropna(subset=["unit_cost"])

    # Pivot: 商品编码 × year → unit_cost
    yearly_cost = cost_per_year.pivot(
        index="商品编码", columns="year", values="unit_cost"
    ).rename(columns=lambda y: f"uc_{int(y)}")

    # Per-store-year pivot
    cost_per_store_year["store_year"] = (
        cost_per_store_year["store_id"].astype(str) + "_" +
        cost_per_store_year["year"].astype(str)
    )
    store_year_cost = cost_per_store_year.pivot(
        index="商品编码", columns="store_year", values="unit_cost"
    ).rename(columns=lambda s: f"uc_{s}")

    # Overall unit cost (all years, all stores combined)
    overall_cost = purchases.groupby("商品编码").apply(
        lambda x: x["cost_in"].sum() / max(x["qty_in"].sum(), 1),
        include_groups=False,
    ).rename("avg_unit_cost").reset_index()

    # 4. Merge POS master with purchase master via barcode
    pur_bc_map = pur_master.set_index("barcode")["商品编码"].to_dict()
    # Some POS barcodes have alternative formats — try direct match first
    pos["商品编码"] = pos["bc_clean"].map(pur_bc_map)

    # Pull purchase 商品名称 onto matched rows
    code_to_name = pur_master.set_index("商品编码")["商品名称"].to_dict()
    pos["商品名称"] = pos["商品编码"].map(code_to_name)

    # 5. Merge cost data
    pos = pos.merge(overall_cost, on="商品编码", how="left")
    pos = pos.merge(yearly_cost, on="商品编码", how="left")
    pos = pos.merge(store_year_cost, on="商品编码", how="left")

    # 6. Classify
    pos[["cat_l1", "cat_l2", "is_lianying", "is_pos_only_non_product"]] = (
        pos.apply(
            lambda r: pd.Series(classify_sku(r["商品编码"], r["sku_name"])),
            axis=1,
        )
    )

    # 7. Tier
    pos["tier"] = pos.apply(assign_tier, axis=1)

    # 8. Reorder columns
    cols = ["sku_id", "商品编码", "pos_barcode", "bc_clean",
            "sku_name", "商品名称",
            "cat_l1", "cat_l2", "is_lianying", "tier",
            "avg_unit_cost"]
    cols += [c for c in pos.columns if c.startswith("uc_") and c != "avg_unit_cost"]
    cols = [c for c in cols if c in pos.columns]
    pos = pos[cols]

    return pos


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    master = build_sku_master()
    print(f"\nMaster rows: {len(master):,}", flush=True)
    print(f"Tier distribution:")
    print(master["tier"].value_counts().to_string())
    print()
    print(f"Cost coverage:")
    print(f"  avg_unit_cost present: {master['avg_unit_cost'].notna().sum():,}")
    print(f"  uc_2024 present:       {master['uc_2024'].notna().sum():,}")
    print(f"  uc_2025 present:       {master['uc_2025'].notna().sum():,}")
    print(f"  uc_2026 present:       {master['uc_2026'].notna().sum():,}")

    Path("data/features").mkdir(parents=True, exist_ok=True)
    master.to_parquet("data/features/sku_master.parquet")
    print(f"\nSaved: data/features/sku_master.parquet", flush=True)
