"""Batch loader — one-command pipeline execution from data/raw/.

Loads real Kemai POS .xls files and Guanjiapo master data,
then runs the full cleaning → feature engineering pipeline.

Usage:
    python -m data_pipeline.ingest.batch_loader
    python -m data_pipeline.ingest.batch_loader --raw-dir data/raw --output-dir data/features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from data_pipeline.ingest.kemai_reader import (
    load_all_stores,
    load_product_master,
    load_purchase_summary,
)
from data_pipeline.orchestrator import PipelineOrchestrator, PipelineReport

logger = logging.getLogger(__name__)


# 批量加载器：一键完成"原始数据 → 清洗 → 特征工程 → 输出 Parquet"全流程
def load_and_run_pipeline(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/features"),
    force_reload: bool = False,
) -> tuple[pd.DataFrame, PipelineReport]:
    """Load all raw data and run the full pipeline.

    Args:
        raw_dir: Directory containing store subfolders and master data.
        output_dir: Where to write the feature Parquet files.

    Returns:
        Tuple of (feature DataFrame, pipeline report).
    """
    # 原始数据目录 + 输出目录 → (特征DataFrame, 管道报告)
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        msg = f"Raw data directory not found: {raw_dir}"
        raise FileNotFoundError(msg)

    # ── 1. 加载POS销售数据 ──
    # 优先级：raw_pos.parquet 缓存（秒级） > .xls 文件（分钟级） > 遗留 CSV
    raw_parquet = raw_dir / "raw_pos.parquet"

    if raw_parquet.exists() and not force_reload:
        # 已有原始缓存，直接读取（跳过 10 分钟的 .xls 解析）
        logger.info("Loading cached raw POS from %s", raw_parquet)
        raw_df = pd.read_parquet(raw_parquet)
    else:
        # 首次运行或强制重载：从 .xls 文件读取
        raw_df = load_all_stores(raw_dir)
        if raw_df.empty:
            pos_path = raw_dir / "pos_transactions.csv"
            if pos_path.exists():
                from data_pipeline.ingest.pos_reader import read_pos_data
                logger.info("Falling back to legacy %s", pos_path)
                raw_df = read_pos_data(pos_path)
            else:
                msg = "No POS data found (no store folders and no pos_transactions.csv)"
                raise RuntimeError(msg)

        # 保存原始数据缓存，下次不用重读 .xls
        if not raw_df.empty and not raw_parquet.exists():
            # object 列统一转 str 避免 pyarrow 混合类型报错
            df_to_save = raw_df.copy()
            for col in df_to_save.select_dtypes(include=["object"]).columns:
                df_to_save[col] = df_to_save[col].astype(str).replace("nan", None)
            df_to_save.to_parquet(raw_parquet, index=False)
            logger.info("Cached raw POS data to %s (%d rows)", raw_parquet, len(raw_df))

    logger.info("Total POS records: %d", len(raw_df))

    # ── 2. 加载商品主数据 ──
    # 优先读管家婆"商品信息.xls"，降级读遗留的 product_master.csv
    product_master = None
    pm_path = raw_dir / "商品信息.xls"
    if not pm_path.exists():
        pm_path = raw_dir / "商品信息.xlsx"
    if pm_path.exists():
        logger.info("Loading product master from %s", pm_path.name)
        product_master = load_product_master(pm_path)
    else:
        legacy_pm = raw_dir / "product_master.csv"
        if legacy_pm.exists():
            from data_pipeline.ingest.product_master_reader import read_product_master
            category_mapping = raw_dir / "category_mapping.csv"
            cat_map = category_mapping if category_mapping.exists() else None
            product_master = read_product_master(legacy_pm, cat_map)

    # ── 3. 加载进货统计获取单位成本 ──
    # 进货统计提供每个SKU的进货均价，这是报童模型 Cu/(Cu+Co) 中 Co（超储成本）的基础
    purchase_files = sorted(raw_dir.glob("*进货统计*.xls"))
    if purchase_files:
        logger.info("Loading purchase summaries: %d files", len(purchase_files))
        purchase_df = load_purchase_summary(purchase_files)

        # 将成本和赠品标记合并到商品主数据中，后续特征工程需要用
        if product_master is not None and not purchase_df.empty:
            product_master = _merge_product_and_cost(product_master, purchase_df)
    else:
        logger.warning("No purchase summary files found")

    # ── 4. POS数据预处理 ──
    # 在进入管道前做折扣率计算和退货标记，保证管道输入格式一致
    raw_df = _preprocess_pos(raw_df)

    # ── 5. 天气数据（可选） ──
    weather_df = None
    weather_path = raw_dir / "weather.csv"
    if weather_path.exists():
        logger.info("Loading weather data from %s", weather_path)
        from data_pipeline.ingest.weather_client import (
            generate_weather_features,
            load_weather_csv,
        )
        weather_df = load_weather_csv(str(weather_path))
        weather_df = generate_weather_features(weather_df)

    # ── 6. 节假日特征（内置日历，不需要外部数据） ──
    holiday_df = None
    if "timestamp" in raw_df.columns:
        date_range = pd.to_datetime(raw_df["timestamp"]).dropna()
        if not date_range.empty:
            from data_pipeline.ingest.holiday_calendar import get_holiday_features
            # 覆盖整个数据日期范围，确保每天都有节假日特征
            full_range = pd.date_range(date_range.min(), date_range.max(), freq="D")
            holiday_df = get_holiday_features(full_range)
            logger.info("Generated holiday features for %d days", len(holiday_df))

    # ── 7. 运行管道 ──
    logger.info("Starting pipeline...")
    orchestrator = PipelineOrchestrator(output_dir=Path(output_dir))
    result_df, report = orchestrator.run(
        raw_df=raw_df,
        weather_df=weather_df,
        holiday_df=holiday_df,
        product_master=product_master,
    )

    # ── 8. 按时间切分为训练集/验证集/留出集 ──
    # 时序数据必须按时间切分，绝不能随机切分（防止数据穿越）
    if "date" in result_df.columns:
        _save_temporal_splits(result_df, output_dir)

    logger.info(report.summary())
    return result_df, report


def _preprocess_pos(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess POS data before pipeline.

    - Convert discount_amount → discount_pct
    - Mark returns (negative quantity)
    """
    # 将绝对折扣金额转为折扣百分比，因为下游模型需要相对值而非绝对值
    if "discount_amount" in df.columns and "total_amount" in df.columns:
        # 用 NA 替换0避免除零，最终 fillna(0) 保证无空值
        safe_total = df["total_amount"].replace(0, pd.NA)
        df["discount_pct"] = (df["discount_amount"].abs() / safe_total.abs() * 100).fillna(0)
        df = df.drop(columns=["discount_amount"])
    elif "discount_pct" not in df.columns:
        df["discount_pct"] = 0.0

    # 标记退货记录（销售数量为负），下游清洗步骤会据此过滤或特殊处理
    if "quantity_sold" in df.columns:
        df["is_return"] = df["quantity_sold"] < 0

    return df


def _merge_product_and_cost(
    product_master: pd.DataFrame,
    purchase_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge product master with purchase cost data via barcode_base.

    Args:
        product_master: From load_product_master() with barcode_base column.
        purchase_df: From load_purchase_summary() with unit_cost, is_promotional_gift.

    Returns:
        Enriched product master with cost info.
    """
    # 通过 barcode_base（去字母后缀的条码）关联商品主数据和进货成本
    # 不能用完整 barcode，因为管家婆条码可能有字母后缀而科脉没有
    if "barcode_base" not in product_master.columns:
        logger.warning("Product master missing barcode_base, skipping cost merge")
        return product_master
    if "barcode_base" not in purchase_df.columns:
        logger.warning("Purchase data missing barcode_base, skipping cost merge")
        return product_master

    cost_cols = ["barcode_base", "unit_cost", "is_promotional_gift"]
    cost_data = purchase_df[[c for c in cost_cols if c in purchase_df.columns]]
    # 同一 barcode_base 可能有多条（不同后缀），取第一条避免一对多膨胀
    cost_data = cost_data.drop_duplicates(subset=["barcode_base"], keep="first")

    # left join：保留所有商品，没有匹配到成本的商品 unit_cost 为 NaN
    merged = product_master.merge(
        cost_data, on="barcode_base", how="left", suffixes=("", "_purchase"),
    )

    matched = merged["unit_cost"].notna().sum()
    logger.info(
        "Merged cost data: %d/%d products matched (%.1f%%)",
        matched, len(merged), matched / max(len(merged), 1) * 100,
    )
    return merged


def _save_temporal_splits(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Split features into IS / OOS / Holdout by date and save separately.

    - IS (Training): 2024.01 - 2025.06
    - OOS (Validation): 2025.07 - 2025.12
    - Holdout (Final test): 2026.01+
    """
    # 严格按时间顺序切分，IS（样本内）训练、OOS（样本外）验证、Holdout 最终测试
    # 这是时序预测的铁律：未来数据绝不能出现在训练集中
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(df["date"])

    # 2024 训练 / 2025 测试（用于与实际年度进货统计对比）
    is_mask = dates < "2025-01-01"
    oos_mask = (dates >= "2025-01-01") & (dates < "2026-01-01")
    holdout_mask = dates >= "2026-01-01"

    splits = {
        "features_train": is_mask,
        "features_oos": oos_mask,
        "features_holdout": holdout_mask,
    }

    for name, mask in splits.items():
        subset = df[mask]
        if not subset.empty:
            path = output_dir / f"{name}.parquet"
            # pyarrow 不接受混合类型的 object 列，统一转为纯字符串
            for col in subset.select_dtypes(include=["object"]).columns:
                subset[col] = subset[col].astype(str).replace("nan", None)
            subset.to_parquet(path, index=False)
            logger.info(
                "Saved %s: %d rows (%s ~ %s)",
                name, len(subset),
                pd.to_datetime(subset["date"]).min().date(),
                pd.to_datetime(subset["date"]).max().date(),
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Load raw Kemai POS data and run full pipeline",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Feature output directory (default: data/features)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force re-read .xls files even if raw_pos.parquet cache exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        df, report = load_and_run_pipeline(
            args.raw_dir, args.output_dir, force_reload=args.force_reload,
        )
        print()
        print(report.summary())
        if report.errors:
            print("Pipeline completed with errors!")
            sys.exit(1)
        print("Pipeline completed successfully!")
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
