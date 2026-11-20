"""POS (Point of Sale) data reader — supports CSV, Parquet, and Excel.

Handles the messy reality of POS exports: inconsistent column names,
mixed date formats, encoding issues, and missing fields.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 通用POS系统列名别名映射表
# 不同POS系统（科脉、管家婆、海信等）导出的列名各不相同，
# 这里穷举所有已知别名，统一映射为管道内部的规范列名
_COLUMN_ALIASES: dict[str, list[str]] = {
    "transaction_id": ["txn_id", "trans_id", "id", "receipt_no", "单号", "流水号", "小票号"],
    "store_id": ["store", "shop_id", "branch_id", "门店编号", "门店", "门店名称", "机构"],
    "sku_id": ["sku", "product_id", "item_id", "barcode", "商品编码", "条码", "plu码"],
    "sku_name": ["product_name", "item_name", "name", "商品名称", "品名"],
    "timestamp": ["date", "datetime", "sale_date", "txn_date", "日期", "销售日期", "单据日期", "交易日期"],
    "quantity_sold": ["qty", "quantity", "sales_qty", "数量", "销量", "实销数量", "销售数量"],
    "unit_price": ["price", "selling_price", "retail_price", "单价", "售价", "零售价"],
    "total_amount": ["amount", "sales_amount", "revenue", "金额", "销售额", "销售金额", "实收金额"],
    "discount_pct": ["discount", "discount_rate", "折扣", "折扣率", "优惠"],
    "category": ["product_category", "cat", "品类", "分类", "类别", "大类"],
}


# 通用POS读取器：文件路径 → 列名归一化后的 DataFrame
def read_pos_data(
    file_path: Path,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read POS data from file and normalize column names.

    Handles CSV, Parquet, and Excel formats. Automatically detects
    common column name variations and maps them to canonical names.

    Args:
        file_path: Path to the data file.
        encoding: Character encoding for CSV files.

    Returns:
        DataFrame with normalized column names.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"POS data file not found: {path}"
        raise FileNotFoundError(msg)

    # 根据文件扩展名选择读取方式
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = _read_csv_safe(path, encoding)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        msg = f"Unsupported file format: {suffix}"
        raise ValueError(msg)

    logger.info("Read %d rows from %s", len(df), path.name)

    # 将各种POS系统的列名统一为规范名称
    df = _normalize_columns(df)

    # 日期列统一为 datetime 类型，无法解析的设为 NaT
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 数值列强制转型，防止混入的文本值导致下游计算出错
    for col in ("quantity_sold", "unit_price", "total_amount", "discount_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # SKU编码只去空格，不去字母后缀——字母后缀在某些系统中代表不同规格/口味
    if "sku_id" in df.columns:
        df["sku_id"] = df["sku_id"].astype(str).str.strip()

    return df


def _read_csv_safe(path: Path, encoding: str) -> pd.DataFrame:
    """Read CSV with fallback encodings for Chinese POS systems."""
    # 中国POS系统导出的CSV编码不统一，按常见程度依次尝试
    # GBK/GB2312 是 Windows 中文系统最常用的编码
    encodings = [encoding, "utf-8-sig", "gbk", "gb2312", "gb18030", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    msg = f"Could not decode {path} with any known encoding"
    raise ValueError(msg)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various column name conventions to canonical names."""
    # 先统一转小写，消除大小写差异
    col_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_lower)

    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        # 每个规范名只匹配第一个命中的别名，避免多列映射到同一目标
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename_map[alias] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("Normalized columns: %s", rename_map)

    return df
