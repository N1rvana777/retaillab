"""Product master data reader — reads Guanjiapo product exports.

Handles Chinese column names, applies category mapping from
Guanjiapo's classification to RetailLab's ProductCategory enum.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 商品主数据列名别名映射
# 管家婆和其他ERP系统的列名各不相同，这里穷举所有已知变体
_PRODUCT_ALIASES: dict[str, list[str]] = {
    "sku_id": ["商品编码", "商品编号", "条码", "条形码", "货号", "plu码", "product_id", "barcode"],
    "sku_name": ["商品名称", "品名", "商品全名", "product_name", "item_name"],
    "category": ["分类", "品类", "类别", "商品分类", "大类", "小类", "product_category"],
    "unit_cost": ["进价", "进货价", "成本价", "采购价", "最新进价", "配送价", "cost", "purchase_price"],
    "selling_price": ["售价", "零售价", "标准售价", "retail_price", "price"],
    "shelf_life_days": ["保质期", "保质期(天)", "shelf_life"],
    "unit_of_measure": ["单位", "基本单位", "unit", "uom"],
    "supplier": ["供应商", "供应商名称", "supplier_name"],
    "substitution_group": ["替代组", "可替代组", "substitution"],
}


# 商品主数据读取器：文件路径 → 归一化后的商品主数据 DataFrame
def read_product_master(
    file_path: Path,
    category_mapping_path: Path | None = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read product master data and normalize for pipeline consumption.

    Args:
        file_path: Path to product master CSV/Excel file.
        category_mapping_path: Optional path to category_mapping.csv
            that maps Guanjiapo categories to RetailLab ProductCategory.
        encoding: Character encoding for CSV files.

    Returns:
        DataFrame with normalized columns, ready for PipelineOrchestrator.
    """
    path = Path(file_path)
    if not path.exists():
        msg = f"Product master file not found: {path}"
        raise FileNotFoundError(msg)

    # 根据文件格式选择读取方式
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = _read_csv_safe(path, encoding)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        msg = f"Unsupported file format: {suffix}"
        raise ValueError(msg)

    logger.info("Read %d products from %s", len(df), path.name)

    # 将各种中英文列名统一为规范名称
    df = _normalize_columns(df)

    # 如果提供了品类映射表，将管家婆的中文分类转为 RetailLab 的标准品类
    if category_mapping_path and Path(category_mapping_path).exists():
        df = _apply_category_mapping(df, Path(category_mapping_path))

    # 价格和保质期必须是数值类型
    for col in ("unit_cost", "selling_price", "shelf_life_days"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # SKU编码只去空格，保留字母后缀（可能代表不同规格）
    if "sku_id" in df.columns:
        df["sku_id"] = df["sku_id"].astype(str).str.strip()

    # 按 sku_id 去重，保留第一条——重复通常是数据导出问题
    if "sku_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["sku_id"], keep="first")
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d duplicate SKUs", dropped)

    return df


def _read_csv_safe(path: Path, encoding: str) -> pd.DataFrame:
    """Read CSV with fallback encodings for Chinese systems."""
    # 依次尝试常见中文编码，兼容各种 Windows 导出环境
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
    # 只去空格不转小写，因为中文列名没有大小写问题，而英文列名可能混合大小写
    col_lower = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_lower)

    rename_map: dict[str, str] = {}
    for canonical, aliases in _PRODUCT_ALIASES.items():
        # 每个规范名只匹配第一个命中的别名
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename_map[alias] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info("Normalized product columns: %s", rename_map)

    return df


# 合并重复SKU：同一编码多行数据合并为一行
def _merge_duplicate_skus(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows with the same sku_id after code normalization.

    Strategy:
    - Numeric columns (unit_cost, selling_price): weighted average or mean
    - Text columns (sku_name, category, supplier, unit_of_measure): keep first non-null
    - shelf_life_days: keep max (conservative — use the longer shelf life)

    Args:
        df: DataFrame with normalized sku_id column.

    Returns:
        DataFrame with one row per unique sku_id.
    """
    dup_mask = df.duplicated(subset=["sku_id"], keep=False)
    n_dups = dup_mask.sum()
    if n_dups == 0:
        return df

    n_unique_dups = df.loc[dup_mask, "sku_id"].nunique()
    logger.warning(
        "Found %d rows with duplicate sku_id (%d unique SKUs) — merging variants",
        n_dups, n_unique_dups,
    )

    # 检查重复SKU之间关键字段是否一致，不一致的发出警告
    # 这能帮助发现管家婆中"同编码不同商品"的数据质量问题
    check_cols = ["unit_cost", "selling_price", "supplier", "sku_name"]
    for col in check_cols:
        if col not in df.columns:
            continue
        inconsistent = df.groupby("sku_id")[col].nunique()
        bad = inconsistent[inconsistent > 1]
        if len(bad):
            logger.warning(
                "SKU variants have inconsistent '%s' for %d SKUs: %s",
                col, len(bad), bad.index.tolist()[:10],
            )

    # 聚合策略：大部分字段取 first（变体间应一致），保质期取最小值（保守策略）
    agg_dict: dict[str, str] = {}
    for col in df.columns:
        if col == "sku_id":
            continue
        # 保质期取最短值：宁可多补货也不要让商品过期
        if col == "shelf_life_days":
            agg_dict[col] = "min"
        else:
            agg_dict[col] = "first"

    merged = df.groupby("sku_id", as_index=False).agg(agg_dict)
    logger.info(
        "Merged %d product rows → %d unique SKUs", len(df), len(merged),
    )
    return merged


def _apply_category_mapping(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    """Map Guanjiapo categories to RetailLab ProductCategory values."""
    # 管家婆的品类体系（如"饮料"、"日化"）与 RetailLab 的标准品类枚举不同，
    # 通过外部映射表转换，未匹配的默认归为"staple"（日用品）
    mapping_df = pd.read_csv(mapping_path, encoding="utf-8-sig")

    if "guanjiapo_category" not in mapping_df.columns or "retaillab_category" not in mapping_df.columns:
        logger.warning("Category mapping file missing required columns, skipping")
        return df

    cat_map = dict(zip(
        mapping_df["guanjiapo_category"],
        mapping_df["retaillab_category"],
    ))

    if "category" in df.columns:
        original_cats = df["category"].nunique()
        # 未匹配的品类默认为 staple，确保不会有空值传入下游
        df["category"] = df["category"].map(cat_map).fillna("staple")
        logger.info(
            "Mapped %d Guanjiapo categories to RetailLab categories",
            original_cats,
        )

    return df
