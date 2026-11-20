"""Load all 80 purchase files into a clean daily per-store per-SKU table.

Output schema:
    date (datetime), store_id (str), 商品编码 (str), barcode (str),
    商品名称 (str), qty_in (float, 入库数量, includes 进货+调入+其他入库),
    qty_out (float, 出库数量, includes 退厂+调出),
    qty_net (float, qty_in - qty_out),
    cost_in (float, 入库成本金额), cost_out (float, 出库成本金额),
    cost_net (float, cost_in - cost_out),
    unit_cost (float, cost_net / max(qty_net, 1))
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import pandas as pd
import numpy as np


STORE_FOLDER_TO_ID = {
    "门店A": "store_a",
    "门店B": "store_b",
    "门店C": "store_c",
}

# Two known schemas for purchase files
SCHEMA_A_COLS = ["商品编码", "国际条码", "日期", "商品名称",
                 "进货数量", "进货成本金额", "调入数量", "调入成本金额",
                 "其它入库数量", "其它入库成本金额", "退厂数量", "退厂成本金额",
                 "调出数量", "调出成本金额"]

SCHEMA_B_COLS = ["商品编码", "商品条码", "日期", "商品名称",
                 "验收数量", "验收成本金额", "调入数量", "调入成本金额",
                 "其它入库数量", "其它入库成本金额",
                 "采购退货数量", "采购退货成本金额",
                 "调出数量", "调出成本金额"]


def load_one_file(path: str, store_id: str) -> pd.DataFrame:
    """Load one purchase file, returning a normalized DataFrame.

    Both Schema A (国际条码 + 进货数量) and Schema B (商品条码 + 验收数量) are handled.
    """
    df = pd.read_excel(path, header=0)

    # Detect schema
    has_a = "国际条码" in df.columns and "进货数量" in df.columns
    has_b = "商品条码" in df.columns and "验收数量" in df.columns

    if has_a:
        bc_col = "国际条码"
        in_qty_col = "进货数量"
        in_cost_col = "进货成本金额"
        ret_qty_col = "退厂数量"
        ret_cost_col = "退厂成本金额"
    elif has_b:
        bc_col = "商品条码"
        in_qty_col = "验收数量"
        in_cost_col = "验收成本金额"
        ret_qty_col = "采购退货数量"
        ret_cost_col = "采购退货成本金额"
    else:
        # Unknown schema — skip
        return pd.DataFrame()

    # Filter blank rows
    df = df[df["商品编码"].notna() & df["商品名称"].notna()].copy()

    # Numeric coercion
    for c in [in_qty_col, in_cost_col, "调入数量", "调入成本金额",
              "其它入库数量", "其它入库成本金额",
              ret_qty_col, ret_cost_col, "调出数量", "调出成本金额"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # Combine columns
    df["qty_in"] = df[in_qty_col] + df["调入数量"] + df["其它入库数量"]
    df["cost_in"] = df[in_cost_col] + df["调入成本金额"] + df["其它入库成本金额"]
    df["qty_out"] = df[ret_qty_col] + df["调出数量"]
    df["cost_out"] = df[ret_cost_col] + df["调出成本金额"]

    df["qty_net"] = df["qty_in"] - df["qty_out"]
    df["cost_net"] = df["cost_in"] - df["cost_out"]

    # Date
    df["date"] = pd.to_datetime(df["日期"], errors="coerce")

    # Standardize 商品编码 (string, no .0, with leading zero preserved)
    df["商品编码"] = df["商品编码"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Standardize barcode
    df["barcode"] = pd.to_numeric(df[bc_col], errors="coerce").apply(
        lambda x: str(int(x)) if pd.notna(x) else None,
    )

    df["store_id"] = store_id

    # Skip rows with no purchase activity (all zeros)
    has_activity = (df["qty_in"] != 0) | (df["qty_out"] != 0) | \
                   (df["cost_in"] != 0) | (df["cost_out"] != 0)
    df = df[has_activity & df["date"].notna()]

    cols = ["date", "store_id", "商品编码", "barcode", "商品名称",
            "qty_in", "qty_out", "qty_net", "cost_in", "cost_out", "cost_net"]
    return df[cols]


def load_all_purchases(verbose: bool = True) -> pd.DataFrame:
    """Load all purchase files from data/raw/进货/ into one DataFrame."""
    warnings.filterwarnings("ignore")

    all_dfs = []
    for folder, store_id in STORE_FOLDER_TO_ID.items():
        # Windows glob is case-insensitive — *.xls and *.XLS would double-load.
        patterns = [
            f"data/raw/进货/{folder}/{folder}/*.xls",
            f"data/raw/进货/{folder}/{folder}/*.XLS",
            f"data/raw/进货/{folder}/*.xls",
            f"data/raw/进货/{folder}/*.XLS",
        ]
        seen = set()
        files = []
        for p in patterns:
            for f in glob.glob(p):
                key = str(Path(f).resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    files.append(f)
        files = sorted(files)
        if verbose:
            print(f"  {store_id} ({folder}): {len(files)} files", flush=True)

        for f in files:
            try:
                df = load_one_file(f, store_id)
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                if verbose:
                    print(f"    ERR {f}: {e}", flush=True)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Aggregate by (date, store_id, 商品编码) — same SKU same day same store may appear in multiple rows
    agg = combined.groupby(["date", "store_id", "商品编码"], as_index=False).agg(
        barcode=("barcode", "first"),
        商品名称=("商品名称", "first"),
        qty_in=("qty_in", "sum"),
        qty_out=("qty_out", "sum"),
        qty_net=("qty_net", "sum"),
        cost_in=("cost_in", "sum"),
        cost_out=("cost_out", "sum"),
        cost_net=("cost_net", "sum"),
    )

    return agg


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("Loading all purchase files...", flush=True)
    df = load_all_purchases()
    print(f"\nTotal rows: {len(df):,}", flush=True)
    print(f"Stores: {df['store_id'].unique()}", flush=True)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}", flush=True)
    print(f"Unique SKUs: {df['商品编码'].nunique():,}", flush=True)
    print(f"Total qty_in: {df['qty_in'].sum():,.0f} 件", flush=True)
    print(f"Total cost_in: {df['cost_in'].sum() / 1e8:.4f} 亿", flush=True)
    print(f"Net (qty_in - qty_out): {df['qty_net'].sum():,.0f}", flush=True)
    print(f"Net cost: {df['cost_net'].sum() / 1e8:.4f} 亿", flush=True)

    # Per-year totals
    df["year"] = df["date"].dt.year
    print("\nBy year and store (qty_net):")
    pivot = df.pivot_table(index="store_id", columns="year",
                           values="qty_net", aggfunc="sum") / 1e4
    print(pivot.round(0))

    print("\nBy year and store (cost_net 亿):")
    pivot = df.pivot_table(index="store_id", columns="year",
                           values="cost_net", aggfunc="sum") / 1e8
    print(pivot.round(4))

    # Save
    Path("data/features").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/features/purchases.parquet")
    print(f"\nSaved: data/features/purchases.parquet ({len(df):,} rows)", flush=True)
