"""Data pipeline configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


# 数据流水线全局配置：所有阈值集中管理，通过环境变量 DATA_ 前缀覆盖
# 避免在各模块中硬编码魔法数字，方便不同环境（开发/测试/生产）调参
class PipelineConfig(BaseSettings):
    """Configuration for the data pipeline.

    Values are read from environment variables with the DATA_ prefix.
    """

    # 数据目录路径
    data_dir: Path = Path("data")
    feature_store_dir: Path = Path("data/features")
    raw_data_dir: Path = Path("data/raw")

    # 宕机检测：小时交易量低于中位数的 5% 视为宕机
    outage_min_hourly_txns_ratio: float = 0.05
    """Flag as outage if hourly txns < median * this ratio."""

    # 折扣异常检测
    discount_iqr_multiplier: float = 3.0   # IQR 倍数阈值
    discount_max_pct: float = 80.0          # 绝对上限——超过 80% 直接标记

    # 误录检测
    duplicate_window_seconds: int = 5       # 重复扫码判定窗口（秒）
    quantity_outlier_std_multiplier: float = 5.0  # 数量异常的标准差倍数

    # 填补参数
    imputation_lookback_weeks: int = 4      # 回看周数——用近 4 周同星期中位数填补

    # 特征工程参数
    lag_days: list[int] = [1, 2, 3, 7, 14, 28]  # 滞后天数：覆盖短中长期
    rolling_windows: list[int] = [7, 14, 28]      # 滚动窗口：周/双周/月

    model_config = {"env_prefix": "DATA_"}
