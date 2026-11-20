"""Backtest: simulate replenishment decisions on OOS data.

Loads the trained LightGBM model, plugs into InventoryOptimizer + Newsvendor,
and generates order recommendations for every SKU-store-day in OOS period.

Usage:
    python -m scripts.backtest
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from quant_engine.domain.entities import SKU, ReplenishmentOrder
from quant_engine.domain.enums import ProductCategory
from quant_engine.domain.value_objects import CostParameters
from quant_engine.forecasting.gradient_boosted import GradientBoostedQuantileForecaster
from quant_engine.optimization.inventory_optimizer import InventoryOptimizer
from quant_engine.forecasting.cold_start import ColdStartForecaster

logger = logging.getLogger(__name__)


def load_trained_model(model_path: Path) -> tuple[GradientBoostedQuantileForecaster, list[str]]:
    """Load trained LightGBM model from joblib."""
    data = joblib.load(model_path)
    params = data["forecaster_params"]

    forecaster = GradientBoostedQuantileForecaster(
        backend=params["backend"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
    )
    # Inject trained models directly
    forecaster._models = data["models"]

    return forecaster, data["feature_cols"]


def run_backtest(
    oos_path: Path = Path("data/features/features_oos.parquet"),
    model_path: Path = Path("models/lgbm_quantile_models.joblib"),
    cu: float = 5.0,
    co: float = 3.0,
    sample_skus: int | None = 500,
) -> pd.DataFrame:
    """Run backtest on OOS data.

    Args:
        oos_path: Path to OOS feature parquet.
        model_path: Path to trained model.
        cu: Underage cost per unit (stockout penalty).
        co: Overage cost per unit (waste/holding cost).
        sample_skus: Limit to N SKUs for speed. None = all SKUs.

    Returns:
        DataFrame with backtest results per SKU-store-day.
    """
    # ── 1. Load model and data ──
    logger.info("Loading model from %s", model_path)
    forecaster, feature_cols = load_trained_model(model_path)

    logger.info("Loading OOS data...")
    oos = pd.read_parquet(oos_path)

    # Convert bools for LightGBM
    for col in feature_cols:
        if col in oos.columns and oos[col].dtype == "bool":
            oos[col] = oos[col].astype(int)

    logger.info("OOS: %d rows, %d SKUs", len(oos), oos["sku_id"].nunique())

    # ── 2. Sample SKUs if needed ──
    if sample_skus and oos["sku_id"].nunique() > sample_skus:
        # Pick SKUs with most data points for meaningful backtest
        sku_counts = oos["sku_id"].value_counts()
        top_skus = sku_counts.head(sample_skus).index
        oos = oos[oos["sku_id"].isin(top_skus)]
        logger.info("Sampled %d SKUs (%d rows)", sample_skus, len(oos))

    # ── 3. Set up cost parameters ──
    # Typical supermarket: selling_price=10, unit_cost=7 → margin=3
    # Cu = margin + goodwill = 3 + 2 = 5, Co = cost - salvage = 7 - 4 = 3
    # Critical ratio = 5/8 = 0.625
    cost_params = CostParameters(
        unit_cost=7.0,
        selling_price=10.0,
        salvage_value=4.0,
        goodwill_penalty=2.0,
    )
    cr = cost_params.critical_ratio
    cu = cost_params.underage_cost
    co = cost_params.overage_cost
    logger.info("Critical ratio: %.3f (Cu=%.1f, Co=%.1f)", cr, cu, co)

    # ── 4. Batch predict ──
    logger.info("Generating predictions...")
    X = oos[feature_cols].values

    # Predict at multiple quantiles
    pred_cr = forecaster._models[0.75].predict(X)  # Closest to cr=0.625
    pred_50 = forecaster._models[0.5].predict(X)
    pred_10 = forecaster._models[0.1].predict(X)
    pred_90 = forecaster._models[0.9].predict(X)

    # For exact critical ratio, interpolate between q=0.5 and q=0.75
    # cr = 0.625, between 0.5 and 0.75
    weight = (cr - 0.5) / (0.75 - 0.5)
    pred_exact_cr = pred_50 + weight * (pred_cr - pred_50)

    # ── 5. Compute results ──
    y_actual = oos["quantity_sold"].values

    # Order quantity = prediction at critical ratio, floored at 0
    order_qty = np.maximum(0, np.ceil(pred_exact_cr))

    # Costs
    underage = np.maximum(y_actual - order_qty, 0) * cu
    overage = np.maximum(order_qty - y_actual, 0) * co
    total_cost = underage + overage

    # Naive baseline: order yesterday's quantity
    naive_order = np.maximum(0, oos["sales_lag_1"].values)
    naive_under = np.maximum(y_actual - naive_order, 0) * cu
    naive_over = np.maximum(naive_order - y_actual, 0) * co
    naive_cost = naive_under + naive_over

    # Build results DataFrame
    results = pd.DataFrame({
        "date": oos["date"].values,
        "store_id": oos["store_id"].values,
        "sku_id": oos["sku_id"].values,
        "actual_demand": y_actual,
        "model_order_qty": order_qty,
        "naive_order_qty": naive_order,
        "pred_median": pred_50,
        "pred_lower": pred_10,
        "pred_upper": pred_90,
        "model_cost": total_cost,
        "naive_cost": naive_cost,
        "model_stockout": (y_actual > order_qty).astype(int),
        "naive_stockout": (y_actual > naive_order).astype(int),
        "model_overstock": np.maximum(order_qty - y_actual, 0),
        "naive_overstock": np.maximum(naive_order - y_actual, 0),
    })

    # ── 6. Print summary ──
    valid = results.dropna(subset=["naive_cost"])

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS (OOS: {oos['date'].min()} ~ {oos['date'].max()})")
    print(f"Cu={cu}, Co={co}, Critical Ratio={cr:.3f}")
    print(f"SKUs: {results['sku_id'].nunique()}, Days: {results['date'].nunique()}")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Model':>12} {'Naive':>12} {'Improvement':>12}")
    print("-" * 66)

    m_cost = valid["model_cost"].mean()
    n_cost = valid["naive_cost"].mean()
    print(f"{'Avg cost/SKU-day':<30} {m_cost:>12.2f} {n_cost:>12.2f} {(1-m_cost/n_cost)*100:>11.1f}%")

    m_stockout = valid["model_stockout"].mean()
    n_stockout = valid["naive_stockout"].mean()
    print(f"{'Stockout rate':<30} {m_stockout:>11.1%} {n_stockout:>11.1%} {(n_stockout-m_stockout)*100:>11.1f}pp")

    m_over = valid["model_overstock"].mean()
    n_over = valid["naive_overstock"].mean()
    print(f"{'Avg overstock/SKU-day':<30} {m_over:>12.2f} {n_over:>12.2f} {(1-m_over/n_over)*100:>11.1f}%")

    total_m = valid["model_cost"].sum()
    total_n = valid["naive_cost"].sum()
    print(f"\n{'Total cost (all SKU-days)':<30} {total_m:>12,.0f} {total_n:>12,.0f} {(1-total_m/total_n)*100:>11.1f}%")
    print(f"{'Total saved':<30} {total_n - total_m:>12,.0f}")

    # Per-store breakdown
    print(f"\n{'--- Per-store breakdown ---':}")
    for store in sorted(results["store_id"].unique()):
        s = valid[valid["store_id"] == store]
        sm = s["model_cost"].mean()
        sn = s["naive_cost"].mean()
        print(f"  {store:<25} model={sm:.2f}  naive={sn:.2f}  save={(1-sm/sn)*100:.1f}%")

    # Save raw results
    output_path = Path("data/features/backtest_results.parquet")
    results.to_parquet(output_path, index=False)
    logger.info("Saved backtest results to %s", output_path)

    # ── 7. Auto-log to backtest_log.csv ──
    _append_backtest_log(
        model_mae=float(np.mean(np.abs(y_actual - pred_50))),
        model_rmse=float(np.sqrt(np.mean((y_actual - pred_50) ** 2))),
        model_stockout_rate=float(m_stockout),
        naive_stockout_rate=float(n_stockout),
        model_avg_cost=float(m_cost),
        naive_avg_cost=float(n_cost),
        total_cost_saved=float(total_n - total_m),
        n_skus=int(results["sku_id"].nunique()),
        n_days=int(results["date"].nunique()),
        n_rows=len(results),
        cu=cu, co=co, cr=cr,
        pinball_50=float(np.mean(np.where(
            y_actual >= pred_50,
            0.5 * (y_actual - pred_50),
            0.5 * (pred_50 - y_actual),
        ))),
    )

    return results


def _append_backtest_log(**metrics: float | int) -> None:
    """Append one row to backtests/backtest_log.csv with timestamp."""
    from datetime import datetime

    log_path = Path("backtests/backtest_log.csv")
    log_path.parent.mkdir(exist_ok=True)

    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **metrics}

    if log_path.exists():
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(log_path, index=False)
    logger.info("Appended backtest log to %s (run #%d)", log_path, len(df))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_backtest()


if __name__ == "__main__":
    main()
