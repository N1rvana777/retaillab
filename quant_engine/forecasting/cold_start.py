"""Cold-start forecasting for new SKUs with insufficient history.

When a new product launches or is introduced to a store, there isn't enough
sales data to train a reliable SKU-level model. This module provides
category-level aggregate forecasting as a Bayesian prior.

Strategy:
1. Train a separate model on aggregate category data (e.g., all bakery items).
2. Use this model to produce baseline forecasts for new SKUs.
3. The InventoryOptimizer applies a conservative multiplier (0.8) on top.
4. As data accumulates, the system automatically transitions to SKU-level models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# 冷启动预测器：为没有足够历史数据的新品提供品类级的需求基线
# 实现了 ForecastModel 协议，可以无缝替换到 InventoryOptimizer 中
class ColdStartForecaster:
    """Category-level quantile forecaster for new products.

    This model is trained on aggregate sales data for an entire category
    (e.g., "all bread products across all stores"). It provides a reasonable
    baseline that is better than guessing zero or using a fixed number.

    Implements the ForecastModel protocol.
    """

    def __init__(
        self,
        category_models: dict[str, Any] | None = None,
        default_daily_demand: float = 10.0,
    ) -> None:
        """Initialize the cold-start forecaster.

        Args:
            category_models: Pre-trained models keyed by category name.
                Each model should have a .predict() method.
            default_daily_demand: Fallback if no category model exists.
        """
        # 品类级预训练模型字典，键是品类名（如 "perishable"），值是训练好的模型
        self._category_models = category_models or {}
        # 终极兜底值：当品类模型也不存在时使用的默认日均需求
        self._default_daily_demand = default_daily_demand

    # 输入：特征 + 分位数 + 品类名 → 输出：品类级的需求分位数预测
    def predict_quantile(
        self,
        features: np.ndarray,
        quantile: float,
        category: str | None = None,
    ) -> float:
        """Predict demand at a given quantile using category-level data.

        If a category model exists, use it. Otherwise, fall back to a
        simple heuristic based on the default daily demand.

        Args:
            features: Feature vector (may be partially populated for new SKUs).
            quantile: Target quantile in (0, 1).
            category: Product category for model lookup.

        Returns:
            Predicted demand at the specified quantile.
        """
        # 优先使用品类级预训练模型
        if category and category in self._category_models:
            model = self._category_models[category]
            feat_2d = features.reshape(1, -1) if features.ndim == 1 else features
            try:
                return float(model.predict(feat_2d)[0])
            except Exception:
                # 品类模型预测失败时不崩溃，降级到启发式方法
                logger.warning(
                    "Category model prediction failed for '%s', using heuristic",
                    category,
                )

        # 启发式兜底：用默认日均需求 + 正态分布假设来估算不同分位数
        from scipy import stats

        # 将分位数转换为标准正态的 z 值
        z = stats.norm.ppf(quantile)
        # 假设变异系数(CV) = 0.3，即标准差 = 均值 * 0.3
        # 0.3 是零售行业中等波动品类的经验值
        cv = 0.3
        estimated_std = self._default_daily_demand * cv
        # 分位数预测 = 均值 + z * 标准差（正态分布性质）
        return max(0.0, self._default_daily_demand + z * estimated_std)

    # 输入：特征 + 品类名 → 输出：品类级的均值需求预测
    def predict_mean(self, features: np.ndarray, category: str | None = None) -> float:
        """Predict mean demand using category-level data.

        Args:
            features: Feature vector.
            category: Product category.

        Returns:
            Predicted mean demand.
        """
        return self.predict_quantile(features, quantile=0.5, category=category)
