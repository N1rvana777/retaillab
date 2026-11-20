"""Train LightGBM quantile regression model on real data.

Usage:
    python -m scripts.train_model
    python -m scripts.train_model --backend xgboost
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features to exclude from training (non-numeric or identifiers)
_EXCLUDE_COLS = {
    # Identifiers & metadata
    "date", "store_id", "sku_id", "barcode", "store_city", "store_city_cn",
    # String columns
    "sku_name", "spec", "unit_of_measure", "barcode_product", "status",
    "barcode_base", "holiday_name",
    # Target + 当天金额（预测时不可知，存在数据穿越）
    "quantity_sold",
    "total_amount",
    # Original values kept for auditing
    "qty_original", "discount_original",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns for training."""
    feature_cols = []
    for col in df.columns:
        if col in _EXCLUDE_COLS:
            continue
        if df[col].dtype in ("object", "string"):
            continue
        feature_cols.append(col)
    return feature_cols


def train_and_evaluate(
    train_path: Path = Path("data/features/features_train.parquet"),
    oos_path: Path = Path("data/features/features_oos.parquet"),
    backend: str = "lightgbm",
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> None:
    """Train quantile models and evaluate on OOS data."""

    # ── 1. Load data ──
    logger.info("Loading training data...")
    t0 = time.time()
    train_df = pd.read_parquet(train_path)
    logger.info("Loaded IS: %d rows in %.1fs", len(train_df), time.time() - t0)

    t0 = time.time()
    oos_df = pd.read_parquet(oos_path)
    logger.info("Loaded OOS: %d rows in %.1fs", len(oos_df), time.time() - t0)

    # ── 2. Select features ──
    feature_cols = get_feature_columns(train_df)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    # Convert boolean columns to int for LightGBM
    for col in feature_cols:
        if train_df[col].dtype == "bool":
            train_df[col] = train_df[col].astype(int)
            oos_df[col] = oos_df[col].astype(int)

    X_train = train_df[feature_cols].values
    y_train = train_df["quantity_sold"].values
    X_oos = oos_df[feature_cols].values
    y_oos = oos_df["quantity_sold"].values

    logger.info("X_train: %s, y_train: %s", X_train.shape, y_train.shape)
    logger.info("X_oos: %s, y_oos: %s", X_oos.shape, y_oos.shape)

    # ── 3. Train quantile models ──
    from quant_engine.forecasting.gradient_boosted import (
        GradientBoostedQuantileForecaster,
    )

    forecaster = GradientBoostedQuantileForecaster(
        backend=backend,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )

    logger.info("Training %s models (n_est=%d, lr=%.3f, depth=%d)...",
                backend, n_estimators, learning_rate, max_depth)
    t0 = time.time()
    forecaster.train(X_train, y_train)
    train_time = time.time() - t0
    logger.info("Training complete in %.1f minutes", train_time / 60)

    # ── 4. Evaluate on OOS ──
    logger.info("Evaluating on OOS data...")
    quantiles = forecaster.trained_quantiles

    # Batch predict for each quantile
    results: dict[float, np.ndarray] = {}
    for q in quantiles:
        model = forecaster._models[q]
        results[q] = model.predict(X_oos)

    # Metrics
    print("\n" + "=" * 60)
    print("OOS Evaluation Results")
    print("=" * 60)

    y_pred_median = results[0.5]
    mae = np.mean(np.abs(y_oos - y_pred_median))
    rmse = np.sqrt(np.mean((y_oos - y_pred_median) ** 2))
    print(f"\nMedian model (q=0.5):")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")

    # Pinball loss for each quantile
    print(f"\nPinball loss by quantile:")
    for q in quantiles:
        y_pred = results[q]
        errors = y_oos - y_pred
        pinball = np.where(errors >= 0, q * errors, (q - 1) * errors)
        mean_pinball = np.mean(pinball)
        print(f"  q={q:.2f}: {mean_pinball:.4f}")

    # Coverage: % of actuals below q prediction
    print(f"\nCoverage (actual < predicted):")
    for q in quantiles:
        y_pred = results[q]
        coverage = np.mean(y_oos <= y_pred)
        print(f"  q={q:.2f}: {coverage:.1%} (target: {q:.0%})")

    # Feature importance (from median model)
    print(f"\nTop 20 feature importances (q=0.5):")
    model_50 = forecaster._models[0.5]
    if hasattr(model_50, "feature_importances_"):
        importances = model_50.feature_importances_
        indices = np.argsort(importances)[::-1]
        for rank, idx in enumerate(indices[:20]):
            print(f"  {rank+1:2d}. {feature_cols[idx]:30s} {importances[idx]:8.0f}")

    # ── 5. Save model ──
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    if backend == "lightgbm":
        import joblib
        model_path = model_dir / "lgbm_quantile_models.joblib"
        joblib.dump({
            "forecaster_params": {
                "backend": backend,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
            },
            "models": forecaster._models,
            "feature_cols": feature_cols,
            "trained_quantiles": list(quantiles),
            "oos_mae": float(mae),
            "oos_rmse": float(rmse),
        }, model_path)
        print(f"\nModel saved to {model_path}")
    else:
        import joblib
        model_path = model_dir / "xgb_quantile_models.joblib"
        joblib.dump({
            "models": forecaster._models,
            "feature_cols": feature_cols,
        }, model_path)
        print(f"\nModel saved to {model_path}")

    print(f"\nTraining time: {train_time/60:.1f} minutes")
    print("Done!")

    # ── 6. Auto-log to training_log.csv ──
    _append_training_log(
        backend=backend,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_features=len(feature_cols),
        n_train_rows=len(y_train),
        n_oos_rows=len(y_oos),
        train_time_min=round(train_time / 60, 1),
        oos_mae=round(float(mae), 4),
        oos_rmse=round(float(rmse), 4),
        coverage_q10=round(float(np.mean(y_oos <= results[0.1])), 3),
        coverage_q50=round(float(np.mean(y_oos <= results[0.5])), 3),
        coverage_q90=round(float(np.mean(y_oos <= results[0.9])), 3),
        top3_features=", ".join(
            feature_cols[i] for i in np.argsort(model_50.feature_importances_)[::-1][:3]
        ) if hasattr(model_50, "feature_importances_") else "",
    )


def _append_training_log(**metrics: object) -> None:
    """Append one row to backtests/training_log.csv with timestamp."""
    from datetime import datetime

    log_path = Path("backtests/training_log.csv")
    log_path.parent.mkdir(exist_ok=True)

    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **metrics}

    if log_path.exists():
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(log_path, index=False)
    logger.info("Appended training log to %s (run #%d)", log_path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quantile regression model")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    train_and_evaluate(
        backend=args.backend,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
