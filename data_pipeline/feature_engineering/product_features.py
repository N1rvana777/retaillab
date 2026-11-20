"""Product-level features — attributes that describe the item itself.

These features capture intrinsic SKU properties that influence demand
and spoilage behavior: category, shelf life, price tier, substitution group.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 输入：销售 DataFrame + 商品主数据 → 输出：合并了商品属性特征的 DataFrame
def add_product_features(
    sales_df: pd.DataFrame,
    product_master: pd.DataFrame,
    sku_col: str = "sku_id",
) -> pd.DataFrame:
    """Merge product master data into daily sales.

    Join strategy:
    - If product_master has 'barcode_base' and sales has 'barcode',
      join via barcode (POS 国际条码 ↔ PM 基本条码去字母).
    - Otherwise fall back to sku_col join.

    Args:
        sales_df: Daily sales DataFrame.
        product_master: Product attribute DataFrame.
        sku_col: SKU identifier column (fallback).

    Returns:
        Merged DataFrame with product features.
    """
    if product_master.empty:
        logger.warning("Empty product master, skipping merge")
        return sales_df

    # 判断 join 策略：优先用条码匹配（管家婆的编码可能与 POS 的 sku_id 不一致）
    use_barcode = (
        "barcode" in sales_df.columns
        and "barcode_base" in product_master.columns
    )

    if use_barcode:
        # 条码匹配：POS 的 barcode 对应商品主数据中去掉字母后缀的 barcode_base
        # 去重防止一对多 join 导致行数膨胀
        pm = product_master.drop_duplicates(subset=["barcode_base"])
        result = sales_df.merge(
            pm, left_on="barcode", right_on="barcode_base",
            how="left", suffixes=("", "_product"),
        )
        matched = result["barcode_base"].notna().sum()
        logger.info(
            "Product join via barcode: %d/%d matched (%.1f%%)",
            matched, len(result), matched / max(len(result), 1) * 100,
        )
    else:
        # 退而用 sku_id 直接匹配
        pm = product_master.drop_duplicates(subset=[sku_col])
        result = sales_df.merge(pm, on=sku_col, how="left", suffixes=("", "_product"))

    # 衍生特征：毛利率——高毛利商品缺货损失更大，影响补货优先级
    if "selling_price" in result.columns and "unit_cost" in result.columns:
        result["margin_pct"] = (
            (result["selling_price"] - result["unit_cost"]) / result["selling_price"] * 100
        ).fillna(0)

    # 短保质期标志——保质期 <7 天的商品需要更精准的补货（过量进货会报废）
    if "shelf_life_days" in result.columns:
        result["is_short_shelf_life"] = (result["shelf_life_days"] < 7).astype(int)

    # 品类独热编码——不同品类的需求模式差异大（如乳品 vs 日化）
    if "category" in result.columns:
        category_dummies = pd.get_dummies(result["category"], prefix="cat")
        result = pd.concat([result, category_dummies], axis=1)

    if use_barcode:
        logger.info("Merged product features via barcode (%d products in master)", len(pm))
    elif sku_col in product_master.columns:
        logger.info("Merged product features for %d SKUs", product_master[sku_col].nunique())
    else:
        logger.info("Merged product features (%d products in master)", len(product_master))
    return result


# 输入：含价格列的 DataFrame → 输出：添加了价格分层特征的 DataFrame
def add_price_tier_feature(
    df: pd.DataFrame,
    price_col: str = "selling_price",
    n_tiers: int = 5,
) -> pd.DataFrame:
    """Assign products to price tiers (quintiles).

    Price tier captures purchasing psychology — budget vs premium items
    have different demand elasticities and stockout tolerance.

    Args:
        df: DataFrame with a price column.
        price_col: Name of the price column.
        n_tiers: Number of price tiers.

    Returns:
        DataFrame with 'price_tier' column (1 = cheapest, n_tiers = most expensive).
    """
    df = df.copy()
    if price_col in df.columns:
        # 用分位数切分而非等距切分，确保每层样本量大致均匀
        df["price_tier"] = pd.qcut(
            df[price_col], q=n_tiers, labels=range(1, n_tiers + 1), duplicates="drop"
        ).astype(int)
    return df
