"""Reconstruct daily inventory per (store, SKU) from snapshots + flows.

Vectorized algorithm:
1. Build event log: every (store, code, date, flow) from purchases/sales,
   plus snapshot anchor rows.
2. Within each (store, code), cumsum the flows.
3. At each snapshot date, compute the offset between snapshot value and
   current cumsum; forward-fill the offset until the next snapshot.
4. Inventory = cumsum + offset.

We only compute inventory on event days (purchase OR sale OR snapshot).
For training feature attachment, the consumer can forward-fill onto
arbitrary days.

Output: data/features/daily_inventory.parquet
    date, store_id, 商品编码, inv_qty (end-of-day, reconstructed)
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
    purchases = pd.read_parquet("data/features/purchases.parquet")
    purchases["date"] = pd.to_datetime(purchases["date"])

    snapshots = pd.read_parquet("data/features/inventory_snapshots.parquet")
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])

    sku_master = pd.read_parquet("data/features/sku_master.parquet")
    code_map = sku_master.dropna(subset=["商品编码"]).set_index(
        "sku_id"
    )["商品编码"].to_dict()
    print(f"  Code map: {len(code_map):,} POS sku_id → 商品编码", flush=True)

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
    sales["商品编码"] = sales["sku_id"].map(code_map)
    sales = sales[sales["商品编码"].notna()]
    sales = sales.groupby(["date", "store_id", "商品编码"], as_index=False)[
        "quantity_sold"
    ].sum()
    print(f"  Sales rows: {len(sales):,}", flush=True)

    # ── Build event log ──
    print("\nBuilding event log...", flush=True)
    pur_ev = purchases[["date", "store_id", "商品编码", "qty_in"]].rename(
        columns={"qty_in": "flow"}
    )
    sal_ev = sales.copy()
    sal_ev["flow"] = -sal_ev["quantity_sold"]
    sal_ev = sal_ev[["date", "store_id", "商品编码", "flow"]]

    flows = pd.concat([pur_ev, sal_ev], ignore_index=True)
    flows = flows.groupby(["date", "store_id", "商品编码"], as_index=False)[
        "flow"
    ].sum()
    print(f"  Flow rows: {len(flows):,}", flush=True)

    # Snapshot rows as anchors
    snap_ev = snapshots.rename(
        columns={"snapshot_date": "date", "inv_qty": "snapshot_value"}
    )[["date", "store_id", "商品编码", "snapshot_value"]]
    print(f"  Snapshot rows: {len(snap_ev):,}", flush=True)

    # Merge: outer join so we have rows for both flow days and snapshot days
    events = flows.merge(
        snap_ev, on=["date", "store_id", "商品编码"], how="outer"
    )
    events["flow"] = events["flow"].fillna(0.0)
    events = events.sort_values(["store_id", "商品编码", "date"]).reset_index(drop=True)
    print(f"  Total events: {len(events):,}", flush=True)

    # ── Vectorized reconstruction ──
    print("\nVectorized cumsum + snapshot correction...", flush=True)
    # For each (store, code) group, cumsum the flow
    events["cum_flow"] = events.groupby(["store_id", "商品编码"])["flow"].cumsum()

    # At each snapshot, the desired inv = snapshot_value
    # Currently inv_provisional = cum_flow (assumes start value 0)
    # We need offset such that inv = cum_flow + offset
    # At snapshot date: snapshot_value = cum_flow_at_snap + offset
    # → offset = snapshot_value - cum_flow_at_snap

    events["offset_change"] = np.where(
        events["snapshot_value"].notna(),
        events["snapshot_value"] - events["cum_flow"],
        np.nan,
    )

    # Forward-fill the offset within (store, code) — start at NaN (treated as 0
    # for SKUs without a snapshot anchor at the beginning)
    events["offset"] = events.groupby(["store_id", "商品编码"])[
        "offset_change"
    ].ffill().fillna(0)

    # IMPORTANT: When a NEW snapshot appears, the offset should JUMP to the new
    # value (replacing any prior offset). The above ffill achieves this because
    # we explicitly set offset_change at each snapshot row.
    # But we also need the offset_change to apply ONLY from that snapshot date
    # onward — already correct via ffill.

    # However: between snapshot N and snapshot N+1, offset stays at offset_N.
    # At snapshot N+1, offset jumps to offset_(N+1). Correct.

    events["inv_qty"] = events["cum_flow"] + events["offset"]
    events["inv_qty"] = events["inv_qty"].clip(lower=0)

    # Drop snapshot-only rows? We want to KEEP all rows for completeness
    # (including snapshot dates), but for output we'll keep only flow-or-snapshot rows.
    # Actually, all rows have either flow or snapshot info — keep all.

    output = events[["date", "store_id", "商品编码", "inv_qty"]].copy()
    print(f"  Output rows: {len(output):,}", flush=True)
    print(f"  Stockout rate (inv_qty == 0): "
          f"{(output['inv_qty'] == 0).mean() * 100:.2f}%", flush=True)

    # ── Sanity check: verify snapshot values ──
    print("\nSanity check: do snapshot values match reconstructed inv?", flush=True)
    check = events.merge(
        snap_ev, on=["date", "store_id", "商品编码"], how="inner",
        suffixes=("", "_actual_snap"),
    )
    if len(check) > 0:
        check["snap_actual"] = check["snapshot_value_actual_snap"].fillna(
            check["snapshot_value"]
        )
        check["error"] = check["inv_qty"] - check["snap_actual"]
        print(f"  Snapshot rows checked: {len(check):,}")
        print(f"  Mean abs error: {check['error'].abs().mean():.4f}")
        print(f"  Max abs error:  {check['error'].abs().max():.4f}")

    Path("data/features").mkdir(parents=True, exist_ok=True)
    output.to_parquet("data/features/daily_inventory.parquet")
    print(f"\nSaved: data/features/daily_inventory.parquet", flush=True)


if __name__ == "__main__":
    main()
