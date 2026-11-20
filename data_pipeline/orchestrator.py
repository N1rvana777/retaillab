"""Pipeline orchestrator — coordinates the full ETL flow.

Sequence:
1. Ingest raw POS data
2. Deduplicate records
3. Detect system outages
4. Clean discount anomalies
5. Fix quantity errors
6. Impute outage gaps
7. Aggregate to daily level
8. Generate features (temporal, lag, external, product)
9. Validate for leakage
10. Store to Parquet

Each step produces auditable counts for monitoring dashboards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_pipeline.cleaning.discount_anomaly import detect_discount_anomalies
from data_pipeline.cleaning.imputer import impute_outage_periods
from data_pipeline.cleaning.misrecord_handler import (
    detect_decimal_errors,
    detect_quantity_outliers,
    remove_duplicate_scans,
)
from data_pipeline.cleaning.outage_detector import detect_outages
from data_pipeline.feature_engineering.anomaly_features import add_anomaly_features
from data_pipeline.feature_engineering.lag_features import (
    add_lag_features,
    add_rolling_features,
    add_trend_features,
    add_yearly_lag,
)
from data_pipeline.feature_engineering.leakage_guard import LeakageGuard
from data_pipeline.feature_engineering.temporal_features import add_temporal_features
from data_pipeline.storage.parquet_store import save_features

logger = logging.getLogger(__name__)


# 流水线运行报告：记录每个清洗步骤的计数，用于数据质量监控仪表板
@dataclass
class PipelineReport:
    """Summary of a pipeline run for auditing and monitoring."""

    input_records: int = 0
    after_dedup: int = 0
    outage_flagged: int = 0
    discount_anomalies: int = 0
    qty_outliers: int = 0
    decimal_errors: int = 0
    imputed_records: int = 0
    output_records: int = 0
    features_generated: int = 0
    leakage_checks_passed: bool = False
    errors: list[str] = field(default_factory=list)

    # 输入：无参数 → 输出：人类可读的流水线运行摘要字符串
    def summary(self) -> str:
        """Human-readable pipeline summary."""
        return (
            f"Pipeline Report:\n"
            f"  Input:              {self.input_records:,}\n"
            f"  After dedup:        {self.after_dedup:,} "
            f"(removed {self.input_records - self.after_dedup:,})\n"
            f"  Outage flagged:     {self.outage_flagged:,}\n"
            f"  Discount anomalies: {self.discount_anomalies:,}\n"
            f"  Qty outliers:       {self.qty_outliers:,}\n"
            f"  Decimal errors:     {self.decimal_errors:,}\n"
            f"  Imputed records:    {self.imputed_records:,}\n"
            f"  Output records:     {self.output_records:,}\n"
            f"  Features:           {self.features_generated:,}\n"
            f"  Leakage check:      {'PASSED' if self.leakage_checks_passed else 'FAILED'}\n"
        )


# 流水线编排器：将 POS 原始交易数据经过 10 步处理变为可供模型训练的特征表
class PipelineOrchestrator:
    """Orchestrates the complete data pipeline from raw POS to ML-ready features."""

    def __init__(
        self,
        output_dir: Path = Path("data/features"),
    ) -> None:
        self._output_dir = output_dir
        self._leakage_guard = LeakageGuard()

    # 输入：原始 POS DataFrame + 可选的促销/天气/节假日/商品主数据
    # 输出：(特征 DataFrame, 流水线报告) 元组
    def run(
        self,
        raw_df: pd.DataFrame,
        promo_df: pd.DataFrame | None = None,
        weather_df: pd.DataFrame | None = None,
        holiday_df: pd.DataFrame | None = None,
        product_master: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, PipelineReport]:
        """Execute the full pipeline.

        Args:
            raw_df: Raw POS transaction data.
            promo_df: Optional promotion calendar.
            weather_df: Optional weather data.
            holiday_df: Optional holiday features.
            product_master: Optional product master data.

        Returns:
            Tuple of (feature DataFrame, pipeline report).
        """
        report = PipelineReport(input_records=len(raw_df))
        # 防御性拷贝，避免修改调用方的原始数据
        df = raw_df.copy()

        try:
            # 第1步：去重 — 收银员重复扫码会产生虚假交易记录
            df = remove_duplicate_scans(df)
            report.after_dedup = len(df)

            # 第2步：缺货检测 — 区分"真正零销量"和"系统宕机假零"
            df = detect_outages(df)
            report.outage_flagged = int(df.get("is_outage", pd.Series(dtype=bool)).sum())

            # 第3步：折扣异常检测 — 收银员误操作或系统故障导致的异常折扣
            df = detect_discount_anomalies(df, promo_df=promo_df)
            report.discount_anomalies = int(
                df.get("discount_anomaly", pd.Series(dtype=bool)).sum()
            )

            # 第4步：数量异常检测 — 识别超出正常范围的极端销量
            df = detect_quantity_outliers(df)
            report.qty_outliers = int(df.get("qty_outlier", pd.Series(dtype=bool)).sum())

            # 第5步：小数点错误检测 — POS 录入时常把 5.0 打成 50 或 0.5
            df = detect_decimal_errors(df)
            report.decimal_errors = int(
                df.get("decimal_error", pd.Series(dtype=bool)).sum()
            )

            # 第6步：填补宕机期数据 — 用历史同星期中位数替代假零值，避免模型低估需求
            df = impute_outage_periods(df)
            report.imputed_records = int(
                df.get("is_imputed", pd.Series(dtype=bool)).sum()
            )

            # 第7步：聚合到"日+门店+SKU"粒度 — 交易明细太细，模型需要日粒度
            df = self._aggregate_daily(df)

            # 第8步：特征工程 — 分四类生成特征
            # 时间特征（星期几、月份等）→ 捕捉周期性需求规律
            df = add_temporal_features(df, date_col="date")
            # 异常特征 → 让模型感知历史异常事件对后续需求的影响
            df = add_anomaly_features(df, date_col="date")
            # 滞后特征 → 过去 N 天的销量，最核心的时序信号
            df = add_lag_features(df)
            # 滚动统计特征 → 平滑近期趋势，降低单日噪声
            df = add_rolling_features(df)
            # 趋势因子 → 从 crypto 量化策略迁移，捕捉销量动量和波动位置
            df = add_trend_features(df)
            # 年同比 → 去年同月同日的销量，捕捉年度周期性（龙抬头、换季等）
            df = add_yearly_lag(df)

            # 外部特征（仅在有数据时才合并，避免空 join 引入全 NaN 列）
            if weather_df is not None and not weather_df.empty:
                from data_pipeline.feature_engineering.external_features import (
                    merge_weather_features,
                )
                df = merge_weather_features(df, weather_df)

            if holiday_df is not None and not holiday_df.empty:
                from data_pipeline.feature_engineering.external_features import (
                    merge_holiday_features,
                )
                df = merge_holiday_features(df, holiday_df)

            if product_master is not None and not product_master.empty:
                from data_pipeline.feature_engineering.product_features import (
                    add_product_features,
                )
                df = add_product_features(df, product_master)

            report.features_generated = len(df.columns)

            # 第9步：穿越验证 — 确保滞后/滚动特征没有使用未来数据
            # 只检查包含 lag_ 或 rolling_ 的列，这些是最易出穿越问题的特征
            lag_cols = [c for c in df.columns if "lag_" in c or "rolling_" in c]
            if lag_cols and "quantity_sold" in df.columns:
                self._leakage_guard.validate_lag_features(
                    df, date_col="date", feature_cols=lag_cols, target_col="quantity_sold"
                )
            report.leakage_checks_passed = True

            # 第10步：持久化 — 保存为 Parquet 格式供模型训练使用
            report.output_records = len(df)
            save_features(df, self._output_dir)

        except Exception as e:
            # 任何步骤失败都记录到报告中，但仍然向上抛出异常
            # 这样调用方既能看到报告，又能感知到失败
            report.errors.append(str(e))
            logger.exception("Pipeline failed: %s", e)
            raise

        logger.info(report.summary())
        return df, report

    # 输入：交易级 DataFrame → 输出：按"日+门店+SKU"聚合后的 DataFrame
    @staticmethod
    def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction-level data to daily per-SKU-per-store."""
        # 从时间戳提取日期，用于日粒度聚合
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        # 动态确定分组列，兼容只有部分列的情况
        group_cols = []
        for col in ["date", "store_id", "sku_id"]:
            if col in df.columns:
                group_cols.append(col)

        if not group_cols:
            return df

        # 不同列用不同聚合方式：销量求和、单价取均值、标志位取"是否存在"
        agg_dict: dict[str, str | list[str]] = {}
        if "quantity_sold" in df.columns:
            agg_dict["quantity_sold"] = "sum"
        if "total_amount" in df.columns:
            agg_dict["total_amount"] = "sum"
        if "unit_price" in df.columns:
            agg_dict["unit_price"] = "mean"
        if "discount_pct" in df.columns:
            agg_dict["discount_pct"] = "mean"
        # 布尔标志列用 any：一天中只要有一笔是异常，整天标记为异常
        if "is_outage" in df.columns:
            agg_dict["is_outage"] = "any"
        if "qty_outlier" in df.columns:
            agg_dict["qty_outlier"] = "any"
        if "is_imputed" in df.columns:
            agg_dict["is_imputed"] = "any"
        if "is_return" in df.columns:
            agg_dict["is_return"] = "any"
        # 属性列取第一个值（同一组内值相同）
        if "store_city" in df.columns:
            agg_dict["store_city"] = "first"
        if "barcode" in df.columns:
            agg_dict["barcode"] = "first"

        if not agg_dict:
            return df

        daily = df.groupby(group_cols, as_index=False).agg(agg_dict)

        logger.info(
            "Aggregated %d transactions to %d daily records", len(df), len(daily)
        )
        return daily
