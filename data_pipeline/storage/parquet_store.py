"""Parquet-based feature store — efficient columnar storage.

Parquet is the ideal format for ML feature tables:
- Columnar: Only reads columns needed for training (fast feature selection)
- Compressed: ~5x smaller than CSV
- Schema-preserving: No type guessing on reload
- Partition-friendly: Can partition by date for incremental updates
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# 输入：特征 DataFrame + 输出目录 → 输出：保存后的文件路径
def save_features(
    df: pd.DataFrame,
    output_dir: Path,
    name: str = "features",
    partition_col: str | None = None,
) -> Path:
    """Save a feature DataFrame to Parquet.

    Args:
        df: Feature DataFrame to save.
        output_dir: Directory to save to.
        name: File/directory name prefix.
        partition_col: Optional column to partition by (e.g., 'date').

    Returns:
        Path to the saved file or directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将混合类型的 object 列统一转为字符串——pyarrow 不支持混合类型列
    # "nan" 字符串替换为 pd.NA 以保持语义正确
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).replace("nan", pd.NA)

    if partition_col and partition_col in df.columns:
        # 按分区列存储为目录结构（Hive 风格），支持增量读取某个日期范围
        output_path = output_dir / name
        df.to_parquet(output_path, partition_cols=[partition_col], index=False)
    else:
        output_path = output_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False)

    logger.info(
        "Saved %d rows × %d columns to %s",
        len(df), len(df.columns), output_path,
    )
    return output_path


# 输入：Parquet 文件路径 + 可选列名列表 → 输出：特征 DataFrame
def load_features(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load features from Parquet.

    Args:
        path: Path to .parquet file or partitioned directory.
        columns: Optional subset of columns to load (for efficiency).

    Returns:
        Feature DataFrame.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Feature store not found: {path}"
        raise FileNotFoundError(msg)

    # 指定 columns 参数时 Parquet 只读取需要的列（列式存储的核心优势）
    df = pd.read_parquet(path, columns=columns)

    logger.info(
        "Loaded %d rows × %d columns from %s",
        len(df), len(df.columns), path,
    )
    return df
