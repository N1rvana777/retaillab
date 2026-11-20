"""Data quality pre-checks before feeding into the pipeline.

These are sanity checks complementary to (not replacing) the pipeline's
own cleaning modules. The goal is to catch obvious problems early —
before wasting time on a full pipeline run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: str  # "ERROR", "WARNING", "INFO"
    check: str
    message: str


@dataclass
class ValidationReport:
    """Result of validating a received dataset."""

    file_name: str
    row_count: int = 0
    column_count: int = 0
    date_range: tuple[str, str] = ("", "")
    unique_stores: int = 0
    unique_skus: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "ERROR" for i in self.issues)

    def summary(self) -> str:
        """Human-readable validation summary."""
        lines = [
            f"验证报告: {self.file_name}",
            f"  行数: {self.row_count:,}",
            f"  列数: {self.column_count}",
        ]
        if self.date_range[0]:
            lines.append(f"  日期范围: {self.date_range[0]} → {self.date_range[1]}")
        if self.unique_stores:
            lines.append(f"  门店数: {self.unique_stores}")
        if self.unique_skus:
            lines.append(f"  SKU数: {self.unique_skus:,}")

        for issue in self.issues:
            prefix = {"ERROR": "[错误]", "WARNING": "[警告]", "INFO": "[信息]"}.get(
                issue.severity, "[?]"
            )
            lines.append(f"  {prefix} {issue.check}: {issue.message}")

        return "\n".join(lines)


# Required columns for sales data (after alias mapping)
_REQUIRED_SALES_COLS = {"store_id", "sku_id", "timestamp", "quantity_sold", "unit_price"}
_REQUIRED_PRODUCT_COLS = {"sku_id"}


def validate_sales_data(
    df: pd.DataFrame,
    file_name: str = "pos_transactions.csv",
    manifest: dict | None = None,
) -> ValidationReport:
    """Validate a sales DataFrame before pipeline ingestion.

    Args:
        df: Sales data with normalized column names.
        file_name: Original file name for the report.
        manifest: Optional manifest dict with expected row counts.

    Returns:
        ValidationReport with findings.
    """
    report = ValidationReport(
        file_name=file_name,
        row_count=len(df),
        column_count=len(df.columns),
    )

    # Check required columns
    missing = _REQUIRED_SALES_COLS - set(df.columns)
    if missing:
        report.issues.append(ValidationIssue(
            severity="ERROR",
            check="required_columns",
            message=f"缺少必要列: {missing}",
        ))
        return report  # Can't do further checks without required columns

    # Date range
    if "timestamp" in df.columns:
        dates = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if len(dates) > 0:
            min_date = dates.min().strftime("%Y-%m-%d")
            max_date = dates.max().strftime("%Y-%m-%d")
            report.date_range = (min_date, max_date)

            # Future dates
            today = datetime.now().strftime("%Y-%m-%d")
            future_count = (dates > pd.Timestamp.now()).sum()
            if future_count > 0:
                report.issues.append(ValidationIssue(
                    severity="ERROR",
                    check="future_dates",
                    message=f"发现 {future_count:,} 条未来日期的记录",
                ))

    # Store and SKU counts
    if "store_id" in df.columns:
        report.unique_stores = df["store_id"].nunique()
    if "sku_id" in df.columns:
        report.unique_skus = df["sku_id"].nunique()

    # Negative prices
    if "unit_price" in df.columns:
        neg_price = (df["unit_price"] < 0).sum()
        ratio = neg_price / len(df) if len(df) > 0 else 0
        if ratio > 0.05:
            report.issues.append(ValidationIssue(
                severity="ERROR",
                check="negative_prices",
                message=f"{neg_price:,} 条记录单价为负 ({ratio:.1%})",
            ))
        elif neg_price > 0:
            report.issues.append(ValidationIssue(
                severity="WARNING",
                check="negative_prices",
                message=f"{neg_price:,} 条记录单价为负 ({ratio:.1%})",
            ))

    # Negative quantities (some are valid returns, but high % is suspicious)
    if "quantity_sold" in df.columns:
        neg_qty = (df["quantity_sold"] < 0).sum()
        ratio = neg_qty / len(df) if len(df) > 0 else 0
        if ratio > 0.2:
            report.issues.append(ValidationIssue(
                severity="WARNING",
                check="negative_quantities",
                message=f"{neg_qty:,} 条负数量记录 ({ratio:.1%})，退货比例偏高",
            ))

    # Manifest row count check
    if manifest and file_name in manifest.get("files", {}):
        expected_rows = manifest["files"][file_name].get("rows", 0)
        if expected_rows and abs(expected_rows - len(df)) / expected_rows > 0.01:
            report.issues.append(ValidationIssue(
                severity="WARNING",
                check="row_count_mismatch",
                message=f"行数与 manifest 不符: 期望 {expected_rows:,}, 实际 {len(df):,}",
            ))

    # Empty stores
    if "store_id" in df.columns:
        store_counts = df["store_id"].value_counts()
        empty_stores = (store_counts < 10).sum()
        if empty_stores > 0:
            report.issues.append(ValidationIssue(
                severity="WARNING",
                check="sparse_stores",
                message=f"{empty_stores} 个门店的交易记录少于 10 条",
            ))

    # Summary info
    report.issues.append(ValidationIssue(
        severity="INFO",
        check="summary",
        message=(
            f"共 {report.unique_stores} 门店, {report.unique_skus:,} SKU, "
            f"日期 {report.date_range[0]} ~ {report.date_range[1]}"
        ),
    ))

    return report


def validate_product_data(
    df: pd.DataFrame,
    file_name: str = "product_master.csv",
) -> ValidationReport:
    """Validate a product master DataFrame."""
    report = ValidationReport(
        file_name=file_name,
        row_count=len(df),
        column_count=len(df.columns),
    )

    missing = _REQUIRED_PRODUCT_COLS - set(df.columns)
    if missing:
        report.issues.append(ValidationIssue(
            severity="ERROR",
            check="required_columns",
            message=f"缺少必要列: {missing}",
        ))
        return report

    if "sku_id" in df.columns:
        report.unique_skus = df["sku_id"].nunique()
        dup_count = len(df) - report.unique_skus
        if dup_count > 0:
            report.issues.append(ValidationIssue(
                severity="WARNING",
                check="duplicate_skus",
                message=f"商品主数据中有 {dup_count:,} 条重复 SKU",
            ))

    # Cost sanity
    if "unit_cost" in df.columns and "selling_price" in df.columns:
        valid = df.dropna(subset=["unit_cost", "selling_price"])
        negative_margin = (valid["selling_price"] < valid["unit_cost"]).sum()
        if negative_margin > 0:
            report.issues.append(ValidationIssue(
                severity="WARNING",
                check="negative_margin",
                message=f"{negative_margin:,} 个 SKU 售价低于进价（可能是促销品或录入错误）",
            ))

    report.issues.append(ValidationIssue(
        severity="INFO",
        check="summary",
        message=f"共 {report.unique_skus:,} 个 SKU, {report.column_count} 个字段",
    ))

    return report
