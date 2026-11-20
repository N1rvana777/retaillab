"""End-to-end integration test: simulated Excel → receive → validate → pipeline.

Tests the full data flow from raw supermarket exports (simulated as CSV files
with Chinese column names) through the receive_and_validate tool, data staging,
and the batch_loader pipeline execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures: simulated supermarket data
# ---------------------------------------------------------------------------

@pytest.fixture
def simulated_sales_csv(tmp_path: Path) -> Path:
    """Create a simulated Guanjiapo sales export CSV with Chinese columns."""
    received_dir = tmp_path / "received"
    received_dir.mkdir()

    rng = np.random.default_rng(42)
    base_date = datetime(2025, 1, 1)
    records = []

    sku_ids = ["SKU-001", "SKU-002", "SKU-003"]
    store_ids = ["STORE-A"]

    for day_offset in range(60):
        dt = base_date + timedelta(days=day_offset)
        for sku in sku_ids:
            for store in store_ids:
                qty = int(rng.poisson(lam=15))
                price = rng.choice([5.0, 8.0, 12.0])
                records.append({
                    "单据日期": dt.strftime("%Y-%m-%d"),
                    "门店编号": store,
                    "商品编码": sku,
                    "数量": qty,
                    "单价": price,
                    "金额": qty * price,
                    "折扣率": 0.0,
                })

    # Inject an outlier
    records.append({
        "单据日期": "2025-01-20",
        "门店编号": "STORE-A",
        "商品编码": "SKU-001",
        "数量": 500,
        "单价": 5.0,
        "金额": 2500.0,
        "折扣率": 0.0,
    })

    df = pd.DataFrame(records)
    csv_path = received_dir / "销售明细_202501.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return received_dir


@pytest.fixture
def simulated_product_csv(tmp_path: Path) -> Path:
    """Create a simulated product master CSV with Chinese columns.

    Uses the same tmp_path/received directory as the sales fixture.
    """
    received_dir = tmp_path / "received"
    received_dir.mkdir(exist_ok=True)

    products = [
        {"商品编码": "SKU-001", "商品名称": "全麦面包", "分类": "面包",
         "进价": 3.0, "售价": 5.0, "保质期": 3, "单位": "个", "供应商": "好面坊"},
        {"商品编码": "SKU-002", "商品名称": "蒙牛纯牛奶1L", "分类": "乳制品",
         "进价": 5.0, "售价": 8.0, "保质期": 45, "单位": "盒", "供应商": "蒙牛"},
        {"商品编码": "SKU-003", "商品名称": "海天酱油500ml", "分类": "调味",
         "进价": 7.0, "售价": 12.0, "保质期": 365, "单位": "瓶", "供应商": "海天"},
    ]
    df = pd.DataFrame(products)
    csv_path = received_dir / "商品信息_202501.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return received_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFileClassification:
    """Test that _classify_file correctly identifies sales vs product data."""

    def test_classify_sales_file(self, simulated_sales_csv: Path) -> None:
        from tools.data_receiver.receive_and_validate import _classify_file

        sales_file = list(simulated_sales_csv.glob("*.csv"))[0]
        assert _classify_file(sales_file) == "sales"

    def test_classify_product_file(self, simulated_product_csv: Path) -> None:
        from tools.data_receiver.receive_and_validate import _classify_file

        product_file = list(simulated_product_csv.glob("商品*"))[0]
        assert _classify_file(product_file) == "product"


class TestInputDetection:
    """Test _detect_input_mode with different file types."""

    def test_detect_raw_csv(self, simulated_sales_csv: Path) -> None:
        from tools.data_receiver.receive_and_validate import _detect_input_mode

        assert _detect_input_mode(simulated_sales_csv) == "raw_files"

    def test_detect_empty_dir(self, tmp_path: Path) -> None:
        from tools.data_receiver.receive_and_validate import _detect_input_mode

        empty = tmp_path / "empty"
        empty.mkdir()
        assert _detect_input_mode(empty) == "unknown"


class TestNormalization:
    """Test _normalize_raw_files turns Chinese-column CSVs into pipeline format."""

    def test_normalize_produces_pos_transactions(
        self, simulated_sales_csv: Path, simulated_product_csv: Path, tmp_path: Path,
    ) -> None:
        from tools.data_receiver.receive_and_validate import _normalize_raw_files

        # Both fixtures use the same received dir
        received_dir = simulated_sales_csv
        work_dir = tmp_path / "work"

        result_dir = _normalize_raw_files(received_dir, work_dir)

        # pos_transactions.csv should exist with canonical columns
        pos_path = result_dir / "pos_transactions.csv"
        assert pos_path.exists()

        df = pd.read_csv(pos_path, encoding="utf-8-sig")
        assert len(df) > 0
        # Check canonical column names after alias mapping
        assert "timestamp" in df.columns or "date" in df.columns or "单据日期" in df.columns
        # The reader should map Chinese columns to canonical names
        for col in ("store_id", "sku_id", "quantity_sold"):
            assert col in df.columns, f"Missing canonical column: {col}"

    def test_normalize_produces_product_master(
        self, simulated_sales_csv: Path, simulated_product_csv: Path, tmp_path: Path,
    ) -> None:
        from tools.data_receiver.receive_and_validate import _normalize_raw_files

        received_dir = simulated_sales_csv
        work_dir = tmp_path / "work"

        result_dir = _normalize_raw_files(received_dir, work_dir)

        product_path = result_dir / "product_master.csv"
        assert product_path.exists()

        df = pd.read_csv(product_path, encoding="utf-8-sig")
        assert len(df) == 3
        assert "sku_id" in df.columns


class TestValidation:
    """Test validator on normalized data."""

    def test_validate_sales_passes(self, simulated_sales_csv: Path, tmp_path: Path) -> None:
        from tools.data_receiver.receive_and_validate import _normalize_raw_files
        from tools.data_receiver.validator import validate_sales_data

        work_dir = tmp_path / "work"
        result_dir = _normalize_raw_files(simulated_sales_csv, work_dir)

        pos_path = result_dir / "pos_transactions.csv"
        df = pd.read_csv(pos_path, encoding="utf-8-sig")

        report = validate_sales_data(df)

        # Should not have blocking errors
        assert not report.has_errors
        assert report.row_count > 0

    def test_validate_product_passes(
        self, simulated_product_csv: Path, tmp_path: Path,
    ) -> None:
        from tools.data_receiver.receive_and_validate import _normalize_raw_files
        from tools.data_receiver.validator import validate_product_data

        work_dir = tmp_path / "work"
        result_dir = _normalize_raw_files(simulated_product_csv, work_dir)

        product_path = result_dir / "product_master.csv"
        if product_path.exists():
            df = pd.read_csv(product_path, encoding="utf-8-sig")
            report = validate_product_data(df)
            assert not report.has_errors


class TestStaging:
    """Test staging validated data to data/raw/."""

    def test_stage_copies_files(
        self, simulated_sales_csv: Path, simulated_product_csv: Path, tmp_path: Path,
    ) -> None:
        from tools.data_receiver.receive_and_validate import _normalize_raw_files
        from tools.data_receiver.stager import stage_data

        received_dir = simulated_sales_csv
        work_dir = tmp_path / "work"
        raw_dir = tmp_path / "data" / "raw"

        result_dir = _normalize_raw_files(received_dir, work_dir)
        staged = stage_data(result_dir, raw_dir)

        assert "pos_transactions.csv" in staged
        assert (raw_dir / "pos_transactions.csv").exists()

    def test_stage_generates_category_mapping_draft(
        self, simulated_sales_csv: Path, simulated_product_csv: Path, tmp_path: Path,
    ) -> None:
        from tools.data_receiver.receive_and_validate import _normalize_raw_files
        from tools.data_receiver.stager import stage_data

        received_dir = simulated_sales_csv
        work_dir = tmp_path / "work"
        raw_dir = tmp_path / "data" / "raw"

        result_dir = _normalize_raw_files(received_dir, work_dir)
        stage_data(result_dir, raw_dir)

        draft = raw_dir / "category_mapping_DRAFT.csv"
        if (result_dir / "product_master.csv").exists():
            assert draft.exists()
            mapping_df = pd.read_csv(draft, encoding="utf-8-sig")
            assert "guanjiapo_category" in mapping_df.columns
            assert "retaillab_category" in mapping_df.columns


class TestEndToEndPipeline:
    """Full end-to-end: CSV → normalize → stage → batch_loader pipeline."""

    def test_full_pipeline_produces_features(
        self, simulated_sales_csv: Path, simulated_product_csv: Path, tmp_path: Path,
    ) -> None:
        """The complete flow from raw CSVs to feature Parquet output."""
        from tools.data_receiver.receive_and_validate import _normalize_raw_files
        from tools.data_receiver.stager import stage_data

        received_dir = simulated_sales_csv
        work_dir = tmp_path / "work"
        raw_dir = tmp_path / "data" / "raw"
        output_dir = tmp_path / "data" / "features"

        # Phase 1: Normalize raw files
        result_dir = _normalize_raw_files(received_dir, work_dir)
        assert (result_dir / "pos_transactions.csv").exists()

        # Phase 2: Stage
        staged = stage_data(result_dir, raw_dir)
        assert "pos_transactions.csv" in staged

        # Phase 3: Run pipeline via batch_loader
        from data_pipeline.ingest.batch_loader import load_and_run_pipeline

        feature_df, report = load_and_run_pipeline(
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

        # Verify pipeline completed
        assert report.input_records > 0
        assert report.output_records > 0
        assert report.leakage_checks_passed
        assert len(report.errors) == 0

        # Verify features were generated
        assert "quantity_sold" in feature_df.columns
        assert feature_df.shape[0] > 0

        # Verify temporal features exist
        temporal_cols = [c for c in feature_df.columns if c.startswith("dow_") or c == "month"]
        assert len(temporal_cols) > 0, "Missing temporal features"

        # Verify lag features exist
        lag_cols = [c for c in feature_df.columns if "lag_" in c or "rolling_" in c]
        assert len(lag_cols) > 0, "Missing lag/rolling features"

        # Verify anomaly features exist
        anomaly_cols = [c for c in feature_df.columns if "anomaly" in c.lower()]
        assert len(anomaly_cols) > 0, "Missing anomaly features"

        # Verify Parquet output was written
        parquet_path = output_dir / "features.parquet"
        assert parquet_path.exists(), f"Expected Parquet at {parquet_path}"

        # Verify Parquet is readable
        loaded = pd.read_parquet(parquet_path)
        assert loaded.shape == feature_df.shape

    def test_pipeline_report_counts_anomalies(
        self, simulated_sales_csv: Path, tmp_path: Path,
    ) -> None:
        """Pipeline report should detect the injected outlier."""
        from tools.data_receiver.receive_and_validate import _normalize_raw_files
        from tools.data_receiver.stager import stage_data
        from data_pipeline.ingest.batch_loader import load_and_run_pipeline

        received_dir = simulated_sales_csv
        work_dir = tmp_path / "work"
        raw_dir = tmp_path / "data" / "raw"
        output_dir = tmp_path / "data" / "features"

        result_dir = _normalize_raw_files(received_dir, work_dir)
        stage_data(result_dir, raw_dir)

        _, report = load_and_run_pipeline(raw_dir=raw_dir, output_dir=output_dir)

        # The injected 500-unit outlier should be detected
        assert report.qty_outliers >= 1, (
            f"Expected at least 1 quantity outlier, got {report.qty_outliers}"
        )
        # Deduplication should have removed some records
        assert report.after_dedup <= report.input_records

    def test_pipeline_with_product_master(
        self, simulated_sales_csv: Path, simulated_product_csv: Path, tmp_path: Path,
    ) -> None:
        """Pipeline should merge product features when product master is staged."""
        from tools.data_receiver.receive_and_validate import _normalize_raw_files
        from tools.data_receiver.stager import stage_data
        from data_pipeline.ingest.batch_loader import load_and_run_pipeline

        received_dir = simulated_sales_csv
        work_dir = tmp_path / "work"
        raw_dir = tmp_path / "data" / "raw"
        output_dir = tmp_path / "data" / "features"

        result_dir = _normalize_raw_files(received_dir, work_dir)
        stage_data(result_dir, raw_dir)

        feature_df, report = load_and_run_pipeline(
            raw_dir=raw_dir,
            output_dir=output_dir,
        )

        assert len(report.errors) == 0
        assert feature_df.shape[0] > 0

        # If product master was present, product features should be merged
        if (raw_dir / "product_master.csv").exists():
            product_cols = [c for c in feature_df.columns if "shelf_life" in c or "unit_cost" in c]
            assert len(product_cols) > 0, "Product features should be merged"
