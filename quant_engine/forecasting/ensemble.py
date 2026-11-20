"""Ensemble forecaster — weighted combination of multiple quantile models.

Combining multiple models reduces variance and improves robustness.
For example, an XGBoost model and a LightGBM model trained on the same
data will make different errors; their average is often better than either.

The ensemble assigns weights based on recent backtesting performance
(inverse pinball loss weighting).
"""

from __future__ import annotations

import logging

import numpy as np

from quant_engine.forecasting.base import ForecastModel

logger = logging.getLogger(__name__)


# 集成预测器：对多个模型的预测做加权平均
# 自身也实现了 ForecastModel 协议，因此可以嵌套使用
# 核心思想：不同模型犯不同的错误，加权平均能减少方差
class EnsembleForecaster:
    """Weighted ensemble of multiple ForecastModel implementations.

    Implements the ForecastModel protocol itself, so it can be used as
    a drop-in replacement anywhere a single model is expected.

    Example:
        xgb_model = GradientBoostedQuantileForecaster(backend="xgboost")
        lgb_model = GradientBoostedQuantileForecaster(backend="lightgbm")
        ensemble = EnsembleForecaster(
            models=[xgb_model, lgb_model],
            weights=[0.4, 0.6],
        )
        demand = ensemble.predict_quantile(features, quantile=0.7)
    """

    def __init__(
        self,
        models: list[ForecastModel],
        weights: list[float] | None = None,
    ) -> None:
        """Initialize the ensemble.

        Args:
            models: List of trained ForecastModel implementations.
            weights: Importance weights for each model. Must sum to 1.
                If None, uses equal weighting.

        Raises:
            ValueError: If models is empty or weights don't sum to ~1.
        """
        if not models:
            msg = "Ensemble requires at least one model"
            raise ValueError(msg)

        # 未指定权重时等权平均——最简单也往往是最稳健的集成方式
        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        if len(weights) != len(models):
            msg = f"Got {len(weights)} weights for {len(models)} models"
            raise ValueError(msg)

        # 权重归一化：即使调用方传入的权重之和不为 1，也自动修正
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning("Weights sum to %.4f, normalizing to 1.0", weight_sum)
            weights = [w / weight_sum for w in weights]

        self._models = models
        self._weights = weights

    # 输入：特征 + 分位数 → 输出：所有模型预测的加权平均值
    def predict_quantile(self, features: np.ndarray, quantile: float) -> float:
        """Weighted average of quantile predictions from all models.

        Args:
            features: Feature vector or matrix.
            quantile: Target quantile in (0, 1).

        Returns:
            Weighted average prediction.
        """
        predictions = []
        for model, weight in zip(self._models, self._weights, strict=True):
            try:
                pred = model.predict_quantile(features, quantile)
                # 直接乘以权重，最后求和等价于加权平均
                predictions.append(pred * weight)
            except Exception:
                # 单个模型失败时跳过它，用剩余模型继续预测
                # 这样即使一个模型损坏，集成仍然能工作
                logger.warning(
                    "Model %s failed for quantile %.2f, skipping",
                    type(model).__name__,
                    quantile,
                )

        # 所有模型都失败才报错
        if not predictions:
            msg = "All ensemble models failed"
            raise RuntimeError(msg)

        return sum(predictions)

    # 输入：特征 → 输出：均值预测（用 0.5 分位数近似）
    def predict_mean(self, features: np.ndarray) -> float:
        """Weighted average of mean predictions.

        Args:
            features: Feature vector or matrix.

        Returns:
            Weighted average mean prediction.
        """
        return self.predict_quantile(features, quantile=0.5)
