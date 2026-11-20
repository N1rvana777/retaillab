"""Abstract base for demand forecasting models.

All forecasting implementations (XGBoost, LightGBM, cold-start, ensemble)
must implement the ForecastModel protocol. This ensures the InventoryOptimizer
can swap models without changing its own code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


# 预测模型协议：所有预测模型必须实现此接口
# InventoryOptimizer 只依赖这个协议，不依赖任何具体实现
# 这样可以随时替换底层模型（XGBoost → LightGBM → 自定义模型）而不改优化器代码
@runtime_checkable
class ForecastModel(Protocol):
    """Protocol for any demand forecasting model.

    Both gradient-boosted quantile models and cold-start heuristics
    implement this interface. The InventoryOptimizer depends only on
    this protocol, never on concrete implementations.
    """

    # 输入：特征矩阵 + 分位数水平 → 输出：该分位数下的需求预测值
    # 这是最核心的方法：报童模型需要在临界比率分位数上获取预测
    def predict_quantile(self, features: np.ndarray, quantile: float) -> float:
        """Predict demand at a given quantile level.

        Args:
            features: Feature matrix (n_samples, n_features) or vector (n_features,).
            quantile: Target quantile in (0, 1). E.g., 0.5 for median.

        Returns:
            Predicted demand value at the specified quantile.
        """
        ...

    # 输入：特征矩阵 → 输出：需求均值预测
    # 通常用中位数（0.5分位数）近似均值
    def predict_mean(self, features: np.ndarray) -> float:
        """Predict expected (mean) demand.

        Default implementation uses the 0.5 quantile (median) as a proxy.

        Args:
            features: Feature matrix or vector.

        Returns:
            Predicted mean demand.
        """
        ...
