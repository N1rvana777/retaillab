"""Quant Engine configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


# 引擎全局配置：所有参数都可以通过 ENGINE_ 前缀的环境变量覆盖
# 例如 ENGINE_MIN_HISTORY_DAYS=14 会将冷启动阈值改为 14 天
class EngineConfig(BaseSettings):
    """Configuration for the quant engine."""

    # 模型文件存储目录
    model_dir: Path = Path("models")
    # 决策日志存储目录
    log_dir: Path = Path("logs/decisions")
    # 冷启动阈值：低于此天数的 SKU 走品类级预测
    min_history_days: int = 28
    # 额外安全库存缓冲：用于战略性商品（如引流品），默认不加
    safety_stock_buffer: float = 0.0
    # 降级引擎的移动平均窗口天数
    fallback_window_days: int = 7
    # 降级引擎的安全系数（1.2 = 比均值多订 20%）
    fallback_safety_factor: float = 1.2
    # 漂移检测基线窗口：90 天覆盖周/月周期
    drift_baseline_days: int = 90
    # 漂移检测对比窗口：7 天平衡灵敏度和稳定性
    drift_recent_days: int = 7
    # 2σ 告警阈值
    drift_warning_sigma: float = 2.0
    # 3σ 严重告警阈值，触发重新训练
    drift_critical_sigma: float = 3.0

    model_config = {"env_prefix": "ENGINE_"}
