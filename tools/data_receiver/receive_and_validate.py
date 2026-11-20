"""Data receiver — reassemble, validate, and stage incoming data.

Usage:
    python receive_and_validate.py [received_dir] [--raw-dir data/raw]

This is the developer-side entry point. Point it at a directory containing
files received from the supermarket (via WeChat, etc.) and it will:

1. Detect input format (raw Excel/CSV or packaged .part archives)
2. For raw files: normalize column names and convert to pipeline format
3. For packaged files: reassemble and verify checksums
4. Run validation checks
5. Stage data to data/raw/ for pipeline consumption
6. Generate a category mapping draft for review

Supported input formats:
- Raw Excel files (.xlsx, .xls) directly exported from Guanjiapo
- Raw CSV files (.csv) with Chinese or English column names
- Packaged .part archives from run_export.bat (legacy)
- .tar.gz archives
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _detect_input_mode(received_dir: Path) -> str:
    """Detect what kind of files we received.

    Returns:
        One of: "parts", "archive", "raw_files"
    """
    has_parts = list(received_dir.glob("*.part"))
    has_tar = list(received_dir.glob("*.tar.gz"))
    has_excel = list(received_dir.glob("*.xlsx")) + list(received_dir.glob("*.xls"))
    has_csv = list(received_dir.glob("*.csv"))

    if has_parts:
        return "parts"
    if has_tar:
        return "archive"
    if has_excel or has_csv:
        return "raw_files"

    return "unknown"


def _normalize_raw_files(received_dir: Path, work_dir: Path) -> Path:
    """Normalize raw Excel/CSV files into pipeline-ready CSV format.

    Reads raw files, applies column name mapping (Chinese → canonical),
    classifies them as sales or product data, and writes standardized CSVs.

    Args:
        received_dir: Directory with raw Excel/CSV files from supermarket.
        work_dir: Working directory for normalized output.

    Returns:
        Path to directory containing normalized CSV files.
    """
    import pandas as pd

    from data_pipeline.ingest.pos_reader import read_pos_data as read_pos_transactions
    from data_pipeline.ingest.product_master_reader import read_product_master

    work_dir.mkdir(parents=True, exist_ok=True)

    excel_files = list(received_dir.glob("*.xlsx")) + list(received_dir.glob("*.xls"))
    csv_files = list(received_dir.glob("*.csv"))
    all_files = excel_files + csv_files

    sales_frames: list[pd.DataFrame] = []
    product_frame: pd.DataFrame | None = None

    for fpath in all_files:
        print(f"  分析文件: {fpath.name} ...", end=" ")

        file_type = _classify_file(fpath)

        if file_type == "sales":
            print("→ 识别为销售流水")
            df = read_pos_transactions(fpath)
            sales_frames.append(df)
        elif file_type == "product":
            print("→ 识别为商品主数据")
            df = read_product_master(fpath)
            if product_frame is None:
                product_frame = df
            else:
                product_frame = pd.concat([product_frame, df], ignore_index=True)
        else:
            print("→ 无法识别类型，跳过")

    # Write normalized CSVs
    if sales_frames:
        combined_sales = pd.concat(sales_frames, ignore_index=True)
        out_path = work_dir / "pos_transactions.csv"
        combined_sales.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  销售数据: {len(combined_sales):,} 行 → {out_path.name}")

    if product_frame is not None:
        out_path = work_dir / "product_master.csv"
        product_frame.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  商品数据: {len(product_frame):,} 行 → {out_path.name}")

    if not sales_frames and product_frame is None:
        print("\n  未能识别任何有效数据文件！")
        print("  请确认文件是从管家婆导出的销售明细或商品信息。")
        sys.exit(1)

    return work_dir


def _classify_file(fpath: Path) -> str:
    """Classify a file as 'sales', 'product', or 'unknown'.

    Uses column name heuristics to determine file type.
    """
    import pandas as pd

    try:
        if fpath.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(fpath, nrows=5)
        else:
            # Try common Chinese encodings
            for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312", "gb18030"):
                try:
                    df = pd.read_csv(fpath, nrows=5, encoding=enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            else:
                return "unknown"
    except Exception:
        return "unknown"

    cols_lower = {str(c).lower().strip() for c in df.columns}

    # Sales indicators: timestamp/date + quantity
    sales_keywords = {
        "数量", "销量", "销售数量", "实销数量", "quantity", "qty",
        "quantity_sold", "sales_qty",
    }
    date_keywords = {
        "日期", "销售日期", "单据日期", "交易日期", "date", "timestamp",
        "sale_date", "txn_date", "datetime",
    }

    has_qty = bool(sales_keywords & cols_lower)
    has_date = bool(date_keywords & cols_lower)

    # Product indicators: cost/price without quantity
    product_keywords = {
        "进价", "进货价", "成本价", "采购价", "最新进价", "配送价",
        "cost", "unit_cost", "purchase_price",
    }
    has_cost = bool(product_keywords & cols_lower)

    if has_qty and has_date:
        return "sales"
    if has_cost and not has_qty:
        return "product"

    # Fallback: if it has typical product-only columns
    product_only = {"保质期", "shelf_life", "供应商", "supplier"}
    if product_only & cols_lower and not has_qty:
        return "product"

    # If it has quantity but no date, assume sales anyway
    if has_qty:
        return "sales"

    return "unknown"


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="接收并验证管家婆导出数据",
    )
    parser.add_argument(
        "received_dir",
        type=Path,
        help="收到的文件所在目录（包含 Excel、CSV、.part 或 .tar.gz 文件）",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw",
        help="数据暂存目录 (默认: data/raw/)",
    )
    parser.add_argument(
        "--skip-staging",
        action="store_true",
        help="仅验证，不复制到 data/raw/",
    )
    args = parser.parse_args()

    received_dir = args.received_dir.resolve()
    raw_dir = args.raw_dir.resolve()

    if not received_dir.exists():
        print(f"目录不存在: {received_dir}")
        sys.exit(1)

    # Lazy imports to keep startup fast
    import pandas as pd

    from stager import stage_data
    from validator import validate_product_data, validate_sales_data

    # Step 1: Detect and process input
    print("=" * 50)
    print("  RetailLab 数据接收工具")
    print("=" * 50)
    print()

    mode = _detect_input_mode(received_dir)

    if mode == "parts" or mode == "archive":
        from reassembler import IntegrityError, reassemble_and_extract

        print("Step 1: 重组分片并解压...")
        try:
            work_dir = received_dir / "_work"
            extracted_dir, checksums = reassemble_and_extract(received_dir, work_dir)
            print(f"  解压到: {extracted_dir}")
            print(f"  文件数: {len(checksums)}")
        except IntegrityError as e:
            print(f"\n校验失败！数据可能在传输中损坏:\n{e}")
            print("请让对方重新发送文件。")
            sys.exit(1)

    elif mode == "raw_files":
        print("Step 1: 检测到原始 Excel/CSV 文件，开始识别和转换...")
        work_dir = received_dir / "_normalized"
        extracted_dir = _normalize_raw_files(received_dir, work_dir)

    else:
        print(f"在 {received_dir} 中未找到可处理的文件")
        print("支持: .xlsx, .xls, .csv, .part, .tar.gz 文件")
        sys.exit(1)

    # Load manifest if present
    manifest = None
    manifest_path = extracted_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"\n  Manifest: 来源={manifest.get('source', '?')}, "
              f"导出时间={manifest.get('export_timestamp', '?')}")

    # Step 2: Validate
    print("\nStep 2: 数据质量验证...")
    all_reports = []

    # Sales data
    sales_path = extracted_dir / "pos_transactions.csv"
    if sales_path.exists():
        sales_df = pd.read_csv(sales_path, encoding="utf-8-sig", low_memory=False)
        report = validate_sales_data(sales_df, "pos_transactions.csv", manifest)
        all_reports.append(report)
        print()
        print(report.summary())

    # Product data
    product_path = extracted_dir / "product_master.csv"
    if product_path.exists():
        product_df = pd.read_csv(product_path, encoding="utf-8-sig", low_memory=False)
        report = validate_product_data(product_df, "product_master.csv")
        all_reports.append(report)
        print()
        print(report.summary())

    if not all_reports:
        print("\n未找到有效的数据文件。请检查文件内容。")
        sys.exit(1)

    # Check for blocking errors
    has_errors = any(r.has_errors for r in all_reports)
    if has_errors:
        print("\n发现阻塞性错误！请检查上方报告并修正数据后重试。")
        if not args.skip_staging:
            print("数据未暂存到 data/raw/。")
        sys.exit(1)

    # Step 3: Stage
    if args.skip_staging:
        print("\n已跳过暂存（--skip-staging）。")
    else:
        print(f"\nStep 3: 暂存到 {raw_dir}...")
        staged = stage_data(extracted_dir, raw_dir)
        print(f"  已暂存 {len(staged)} 个文件:")
        for name in staged:
            print(f"    - {name}")

        # Check for category mapping draft
        draft = raw_dir / "category_mapping_DRAFT.csv"
        if draft.exists():
            print(f"\n  请审核品类映射: {draft}")
            print("  审核完成后另存为 category_mapping.csv")

    print("\n完成！")
    if not args.skip_staging:
        print(f"下一步: 运行 pipeline → python -m data_pipeline.ingest.batch_loader")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已取消。")
    except Exception as e:
        print(f"\n出错: {e}")
        sys.exit(1)
