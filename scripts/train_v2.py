"""Train v2 quantile models on the new feature set.

Target: true_demand (uncensored). Sample weights: log1p(true_demand) × 2 if
within 30 days of Chinese New Year.

Quantiles trained: 0.50, 0.55, 0.60, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


# Columns to exclude from feature set (target, IDs, leakage candidates)
EXCLUDE_COLS = {
    # IDs / metadata
    "date", "sku_id", "store_id", "商品编码", "barcode", "sku_name",
    "商品名称", "spec", "product_name", "category", "store_city_cn",
    # Targets and direct leakage
    "quantity_sold", "true_demand", "total_amount", "discount_amount",
    "unit_price", "is_promotional_gift", "is_return",
    # Same-day inventory (leakage — we know it AFTER selling)
    "inv_qty_today", "inv_start_of_day",
    "is_stockout", "is_partial_stockout",
    # Same-day purchase (could be lead-leakage; we'd know if a delivery came
    # before predicting; safer to rely on lagged purchase_qty_7d only)
    "purchase_today",
    # Other constants / meta we don't want as features
    "store_city", "is_outage", "is_imputed",
    # Cost columns (we use these in profit calc, not as model features)
    "avg_unit_cost",
    # String categorical not yet encoded
    "tier",
}


def select_features(df: pd.DataFrame) -> list[str]:
    """Pick columns suitable as model features."""
    candidates = []
    cat_cols = ["cat_l1", "cat_l2"]
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if c in cat_cols:
            candidates.append(c)
            continue
        # Skip uc_<store>_<year> columns — not meant for prediction features,
        # they're for downstream profit calculation.
        if c.startswith("uc_"):
            continue
        # Numeric or boolean only
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            candidates.append(c)
        elif pd.api.types.is_string_dtype(df[c]):
            # skip arbitrary strings
            pass
    return candidates


def main():
    warnings.filterwarnings("ignore")
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    t0 = time.time()
    print("Loading v2 features...", flush=True)
    train = pd.read_parquet("data/features/features_v2_train.parquet")
    oos = pd.read_parquet("data/features/features_v2_oos.parquet")
    print(f"  train: {len(train):,} rows, oos: {len(oos):,} rows", flush=True)

    # Pick features
    features = select_features(train)
    print(f"\nFeatures selected: {len(features)}", flush=True)

    # Encode categorical columns
    cat_cols_used = [c for c in features if c in ("cat_l1", "cat_l2")]
    for c in cat_cols_used:
        train[c] = train[c].astype("category")
        oos[c] = oos[c].astype("category")

    # Boolean → int
    for c in features:
        if pd.api.types.is_bool_dtype(train[c]):
            train[c] = train[c].astype(int)
            oos[c] = oos[c].astype(int)

    # Target: true_demand (uncensored)
    y_train = train["true_demand"].astype(float).values
    # Drop NaN target rows
    valid_train = ~np.isnan(y_train)
    print(f"  Valid train rows: {valid_train.sum():,} / {len(y_train):,}", flush=True)
    train_v = train.iloc[valid_train].copy()
    y_train = y_train[valid_train]

    # Sample weights
    base_weight = np.log1p(np.maximum(y_train, 0))
    cny_boost = np.where(train_v["is_within_30d_of_cny"].values == 1, 2.0, 1.0)
    sample_weight = base_weight * cny_boost
    print(f"  CNY-boosted samples: {(cny_boost > 1).sum():,} "
          f"({(cny_boost > 1).sum() / len(cny_boost) * 100:.1f}%)",
          flush=True)

    X_train = train_v[features]
    X_oos = oos[features].copy()

    # Train 9 quantiles
    quantiles = [0.50, 0.55, 0.60, 0.625, 0.75, 0.80, 0.85, 0.90, 0.95]
    models = {}
    for q in quantiles:
        t1 = time.time()
        print(f"\n[q={q:.3f}] Training...", flush=True)
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train, sample_weight=sample_weight,
            categorical_feature=cat_cols_used,
        )
        elapsed = time.time() - t1

        pred = model.predict(X_oos)
        # Evaluate against true_demand on OOS
        y_oos = oos["true_demand"].astype(float).values
        valid_oos = ~np.isnan(y_oos)
        mae = np.mean(np.abs(y_oos[valid_oos] - pred[valid_oos]))
        coverage = np.mean(y_oos[valid_oos] <= pred[valid_oos])

        models[q] = model
        print(f"  done in {elapsed:.0f}s | MAE={mae:.4f} | coverage={coverage:.1%}",
              flush=True)

    # ── Save ──
    Path("models").mkdir(exist_ok=True)
    joblib.dump({
        "features": features,
        "cat_cols": cat_cols_used,
        "quantile_models": models,
        "quantiles_tested": quantiles,
        "target": "true_demand",
        "weighting": "log1p × 2x_if_within_30d_of_cny",
    }, "models/v2_quantile_models.joblib")

    print(f"\nSaved: models/v2_quantile_models.joblib", flush=True)
    print(f"Total time: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
