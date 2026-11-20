"""Stage validated data into data/raw/ for pipeline consumption.

Also generates a draft category mapping file that maps Guanjiapo's
Chinese product categories to RetailLab's ProductCategory enum.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Heuristic mapping: Guanjiapo Chinese category keywords → RetailLab category
_CATEGORY_HEURISTICS: dict[str, list[str]] = {
    "perishable": [
        "生鲜", "水果", "蔬菜", "面包", "烘焙", "乳制品", "牛奶", "酸奶",
        "鲜肉", "冷鲜", "熟食", "豆制品", "鸡蛋", "水产", "冷藏", "鲜花",
    ],
    "staple": [
        "米", "面", "油", "调味", "酱油", "醋", "盐", "糖",
        "罐头", "日用", "洗护", "纸品", "清洁", "粮油", "干货",
        "方便面", "饮料", "酒", "零食", "饼干", "糖果", "休闲食品",
    ],
    "seasonal": [
        "月饼", "粽子", "年货", "汤圆", "元宵", "腊味",
        "火锅", "冰品", "雪糕", "冰淇淋", "防晒",
    ],
    "promotional": [
        "促销", "特价", "清仓", "临期",
    ],
}


def stage_data(
    extracted_dir: Path,
    raw_dir: Path,
) -> list[str]:
    """Copy validated data to data/raw/ with standardized names.

    Args:
        extracted_dir: Directory with validated CSV files.
        raw_dir: Target data/raw/ directory.

    Returns:
        List of staged file names.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []

    # Stage sales data
    for name in ("pos_transactions.csv",):
        src = extracted_dir / name
        if src.exists():
            dst = raw_dir / name
            _copy_csv(src, dst)
            staged.append(name)
            logger.info("Staged: %s → %s", src, dst)

    # Stage product master
    product_src = extracted_dir / "product_master.csv"
    if product_src.exists():
        dst = raw_dir / "product_master.csv"
        _copy_csv(product_src, dst)
        staged.append("product_master.csv")

        # Generate category mapping draft
        _generate_category_mapping(product_src, raw_dir)

    # Stage promo calendar if present
    promo_src = extracted_dir / "promo_calendar.csv"
    if promo_src.exists():
        dst = raw_dir / "promo_calendar.csv"
        _copy_csv(promo_src, dst)
        staged.append("promo_calendar.csv")

    return staged


def _generate_category_mapping(
    product_file: Path,
    raw_dir: Path,
) -> Path:
    """Generate a draft category mapping from Guanjiapo categories.

    Creates category_mapping_DRAFT.csv that the developer should review
    and save as category_mapping.csv.

    Args:
        product_file: Path to product_master.csv.
        raw_dir: Where to write the draft mapping.

    Returns:
        Path to the generated draft file.
    """
    df = pd.read_csv(product_file, encoding="utf-8-sig")

    # Find the category column
    cat_col = None
    for col in ("category", "分类", "品类", "类别", "商品分类"):
        if col in df.columns:
            cat_col = col
            break

    if cat_col is None:
        logger.warning("No category column found in product master")
        return raw_dir / "category_mapping_DRAFT.csv"

    unique_cats = df[cat_col].dropna().unique()

    rows = []
    for cat in sorted(unique_cats):
        suggested = _guess_category(str(cat))
        rows.append({
            "guanjiapo_category": cat,
            "retaillab_category": suggested,
            "confidence": "high" if suggested != "staple" else "low",
            "notes": "",
        })

    mapping_df = pd.DataFrame(rows)
    draft_path = raw_dir / "category_mapping_DRAFT.csv"
    mapping_df.to_csv(draft_path, index=False, encoding="utf-8-sig")

    logger.info(
        "Generated category mapping draft: %d categories → %s",
        len(rows),
        draft_path,
    )
    return draft_path


def _guess_category(cat_name: str) -> str:
    """Guess RetailLab ProductCategory from Chinese category name."""
    for retaillab_cat, keywords in _CATEGORY_HEURISTICS.items():
        for kw in keywords:
            if kw in cat_name:
                return retaillab_cat
    return "staple"  # Default fallback


def _copy_csv(src: Path, dst: Path) -> None:
    """Copy a file."""
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            chunk = fsrc.read(8192)
            if not chunk:
                break
            fdst.write(chunk)
