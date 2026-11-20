"""Load all inventory snapshot files (库存) into a clean table.

We have 6 snapshot dates × 3 stores = 18 files. Each file contains
per-SKU inventory at that date.

Output schema:
    snapshot_date (datetime), store_id (str), 商品编码 (str), barcode (str),
    商品名称 (str), inv_qty (float, 结存数量), inv_cost (float, 结存成本金额)
"""
from __future__ import annotations

import glob
import re
import warnings
from pathlib import Path

import pandas as pd


# Map filename keywords to store_id
FILENAME_TO_STORE = {
    "门店A": "store_a",
    "门店B": "store_b",
    "城市B": "store_c",
}


def parse_snapshot_date_from_filename(fname: str) -> pd.Timestamp | None:
    """Extract snapshot date from filename like '2025.01.01门店A.XLS'."""
    m = re.match(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})", fname)
    if not m:
        return None
    y, mo, d = m.groups()
    try:
        return pd.Timestamp(f"{y}-{mo}-{d}")
    except Exception:
        return None


def parse_store_from_filename(fname: str) -> str | None:
    for kw, store_id in FILENAME_TO_STORE.items():
        if kw in fname:
            return store_id
    return None


def load_one_inventory_file(path: str) -> pd.DataFrame:
    """Load one inventory snapshot file."""
    fname = Path(path).name
    snap_date = parse_snapshot_date_from_filename(fname)
    store_id = parse_store_from_filename(fname)
    if snap_date is None or store_id is None:
        return pd.DataFrame()

    df = pd.read_excel(path, header=0)

    # Expected schema: # | 管库标志 | 日期 | 机构名称 | 商品编码 | 商品条码 | 商品名称 | 结存数量 | 结存成本金额
    if not {"商品编码", "商品名称", "结存数量"}.issubset(df.columns):
        return pd.DataFrame()

    # Drop summary/blank rows
    df = df[df["商品编码"].notna() & df["商品名称"].notna()].copy()

    # Standardize 商品编码 (preserve leading zeros where present in source)
    df["商品编码"] = df["商品编码"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Barcode column
    bc_col = "商品条码" if "商品条码" in df.columns else None
    if bc_col:
        df["barcode"] = pd.to_numeric(df[bc_col], errors="coerce").apply(
            lambda x: str(int(x)) if pd.notna(x) else None,
        )
    else:
        df["barcode"] = None

    # Numeric
    df["inv_qty"] = pd.to_numeric(df["结存数量"], errors="coerce").fillna(0.0)
    df["inv_cost"] = pd.to_numeric(df.get("结存成本金额", 0), errors="coerce").fillna(0.0)

    df["snapshot_date"] = snap_date
    df["store_id"] = store_id

    cols = ["snapshot_date", "store_id", "商品编码", "barcode",
            "商品名称", "inv_qty", "inv_cost"]
    return df[cols]


def load_all_inventory(verbose: bool = True) -> pd.DataFrame:
    warnings.filterwarnings("ignore")

    # Windows glob is case-insensitive — using both *.xls and *.XLS
    # would double-count files. Use only one and dedupe by lowercase path.
    files_raw = (
        glob.glob("data/raw/库存/库存/*.xls") +
        glob.glob("data/raw/库存/库存/*.XLS")
    )
    seen = set()
    files = []
    for f in files_raw:
        key = str(Path(f).resolve()).lower()
        if key not in seen:
            seen.add(key)
            files.append(f)
    files = sorted(files)
    if verbose:
        print(f"  Inventory files: {len(files)}", flush=True)

    all_dfs = []
    for f in files:
        try:
            df = load_one_inventory_file(f)
            if len(df) > 0:
                all_dfs.append(df)
                if verbose:
                    fname = Path(f).name
                    print(f"    {fname}: {len(df):,} SKUs, "
                          f"qty={df['inv_qty'].sum():,.0f}, "
                          f"cost={df['inv_cost'].sum()/1e6:.2f}百万",
                          flush=True)
        except Exception as e:
            if verbose:
                print(f"    ERR {f}: {e}", flush=True)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("Loading all inventory snapshots...", flush=True)
    df = load_all_inventory()
    print(f"\nTotal rows: {len(df):,}", flush=True)
    print(f"Snapshot dates: {sorted(df['snapshot_date'].unique())}", flush=True)
    print(f"Stores: {df['store_id'].unique()}", flush=True)

    # Per-snapshot summary
    print("\nPer-snapshot summary:")
    pivot = df.pivot_table(
        index=["snapshot_date"],
        columns="store_id",
        values="inv_cost",
        aggfunc="sum",
    ) / 1e6
    print(pivot.round(2))

    print("\nPer-snapshot SKU counts:")
    pivot_skus = df.pivot_table(
        index=["snapshot_date"],
        columns="store_id",
        values="商品编码",
        aggfunc="count",
    )
    print(pivot_skus)

    Path("data/features").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/features/inventory_snapshots.parquet")
    print(f"\nSaved: data/features/inventory_snapshots.parquet", flush=True)
