"""Train multiple quantile models and compare annual profit."""
from __future__ import annotations

import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


def main() -> None:
    d2 = joblib.load("models/step2_weighted.joblib")
    features = d2["features"]
    print(f"Features: {len(features)}", flush=True)

    t0 = time.time()
    train = pd.read_parquet("data/features/features_train.parquet")
    oos = pd.read_parquet("data/features/features_oos.parquet")
    print(f"Data loaded: {time.time()-t0:.1f}s, train={len(train)}, oos={len(oos)}", flush=True)

    for col in features:
        if col in train.columns and train[col].dtype == "bool":
            train[col] = train[col].astype(int)
        if col in oos.columns and oos[col].dtype == "bool":
            oos[col] = oos[col].astype(int)

    X_train = train[features].values
    y_train = train["quantity_sold"].values
    X_oos = oos[features].values
    y_oos = oos["quantity_sold"].values

    nan_mask = np.isnan(y_train)
    if nan_mask.any():
        valid = ~nan_mask
        X_train = X_train[valid]
        y_train = y_train[valid]

    weights = np.log1p(np.maximum(y_train, 0))

    quantiles = [0.5, 0.55, 0.6, 0.625, 0.75]
    models = {}

    for q in quantiles:
        t1 = time.time()
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=300, max_depth=6, learning_rate=0.05,
            verbose=-1, n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        elapsed = time.time() - t1
        pred = model.predict(X_oos)
        mae = np.mean(np.abs(y_oos - pred))
        coverage = np.mean(y_oos <= pred)
        models[q] = model
        print(f"q={q:.3f}: {elapsed:.0f}s, MAE={mae:.3f}, coverage={coverage:.1%}", flush=True)

    # Save models IMMEDIATELY after training
    joblib.dump({
        "features": features,
        "quantile_models": models,
        "quantiles_tested": quantiles,
    }, "models/step3_results.joblib")
    print("Models saved!", flush=True)

    # Now compute profit
    print("\nComputing annual profit comparison...", flush=True)
    _compute_profit(models, features, oos, X_oos, y_oos)
    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)


def _compute_profit(models, features, oos, X_oos, y_oos):
    def fmt_bc(v):
        if pd.isna(v): return ""
        if isinstance(v, float): return str(int(v))
        return str(v).strip()

    def strip_bc(bc):
        m = re.match(r"^(\d{8,})[a-zA-Z]+$", bc)
        return m.group(1) if m else bc

    def parse_spec(spec):
        if pd.isna(spec): return 1
        spec = str(spec).strip()
        if "*" not in spec: return 1
        m = re.search(r"(\d+(?:\*\d+)+)", spec)
        if not m: return 1
        result = 1
        for p in m.group(1).split("*"):
            try:
                v = int(p)
                if v > 0: result *= v
            except ValueError:
                pass
        return max(1, result)

    pur = pd.read_excel("data/raw/2025年进货统计(1).xls", header=1)
    pur["barcode_base"] = pur["基本条码"].apply(fmt_bc).apply(strip_bc)
    pur["spec_multiplier"] = pur["规格"].apply(parse_spec)
    pur["unit_cost"] = pd.to_numeric(pur["进货均价"], errors="coerce") / pur["spec_multiplier"]
    pur["purchase_pieces"] = pd.to_numeric(pur["进货数量"], errors="coerce") * pur["spec_multiplier"]

    pur_agg = pur.groupby("barcode_base").agg(
        purchase_pieces=("purchase_pieces", "sum"), unit_cost=("unit_cost", "mean"),
    ).reset_index()

    FRESH_KW = ["菜","肉","鱼","虾","蟹","蛋","鲜","水果","瓜","桃","梨","苹果","葡萄","香蕉","橙","柚","芒果","草莓","樱桃","蓝莓","西瓜","白菜","萝卜","土豆","番茄","豆芽","蘑菇","杏鲍菇","猪","牛","羊","鸡","鸭","排骨","五花","里脊","三文鱼","生蚝","熟食","凉菜","卤"]
    SHORT_KW = ["面包","酸奶","鲜奶","牛奶","豆腐","蛋糕","糕点","酸牛奶","乳酸","鲜牛奶"]
    SPOILAGE = {"fresh": 0.50, "short_shelf": 0.30, "long_shelf": 0.00}

    def classify(name):
        if pd.isna(name): return "long_shelf"
        name = str(name)
        for kw in FRESH_KW:
            if kw in name: return "fresh"
        for kw in SHORT_KW:
            if kw in name: return "short_shelf"
        return "long_shelf"

    raw = pd.read_parquet("data/raw/raw_pos.parquet", columns=["sku_id", "sku_name"])
    name_map = raw.drop_duplicates("sku_id").set_index("sku_id")["sku_name"]

    print(f"{'q':>7} | {'ratio':>8} | {'actual':>12} | {'model':>12} | {'diff_wan':>10} | {'pct':>8}", flush=True)
    print("-" * 72, flush=True)

    for q in sorted(models.keys()):
        pred = models[q].predict(oos[features].values)
        daily_qty = np.maximum(0, np.round(pred))
        oos_tmp = oos[["store_id", "sku_id", "barcode", "quantity_sold", "total_amount"]].copy()
        oos_tmp["pred"] = daily_qty
        annual = oos_tmp.groupby(["store_id", "sku_id", "barcode"]).agg(
            model_demand=("pred", "sum"), actual_sold=("quantity_sold", "sum"),
            actual_revenue=("total_amount", "sum"),
        ).reset_index()
        annual["model_qty"] = np.ceil(annual["model_demand"] * 1.05)
        annual = annual.merge(pur_agg, left_on="barcode", right_on="barcode_base", how="inner")
        annual["sku_name"] = annual["sku_id"].map(name_map)
        annual["category"] = annual["sku_name"].apply(classify)
        annual["unit_price"] = np.where(annual["actual_sold"] > 0, annual["actual_revenue"] / annual["actual_sold"], 0)
        store_total = annual.groupby("sku_id")["actual_sold"].transform("sum")
        annual["store_share"] = np.where(store_total > 0, annual["actual_sold"] / store_total, 1 / 3)
        annual["actual_pur"] = annual["purchase_pieces"] * annual["store_share"]
        v = annual[(annual["unit_cost"] > 0) & (annual["unit_price"] > annual["unit_cost"]) & (annual["actual_sold"] > 0)].copy()
        sp = v["category"].map(SPOILAGE)
        ap = v["actual_revenue"].sum() - (v["actual_pur"] * v["unit_cost"]).sum() - (np.maximum(v["actual_pur"] - v["actual_sold"], 0) * v["unit_cost"] * sp).sum()
        mr = (np.minimum(v["model_qty"], v["actual_sold"]) * v["unit_price"]).sum()
        mc = (v["model_qty"] * v["unit_cost"]).sum()
        mw = (np.maximum(v["model_qty"] - v["actual_sold"], 0) * v["unit_cost"] * sp).sum()
        mp = mr - mc - mw
        ratio = v["model_qty"].sum() / v["actual_sold"].sum()
        diff = (mp - ap) / 1e4
        pct = (mp / ap - 1) * 100
        print(f"q={q:.3f} | {ratio:>7.1%} | {ap/1e8:>11.4f}B | {mp/1e8:>11.4f}B | {diff:>+9.0f}W | {pct:>+7.1f}%", flush=True)


if __name__ == "__main__":
    main()
