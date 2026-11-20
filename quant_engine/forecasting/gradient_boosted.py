"""XGBoost / LightGBM quantile regression forecaster.

This is the primary forecasting engine. Instead of predicting a single point
estimate, it trains separate models at multiple quantile levels, enabling
the InventoryOptimizer to read off demand at any quantile — particularly
the Newsvendor critical ratio quantile.

Design choice: We pre-train models at standard quantile levels and interpolate
for non-standard levels. For production, we also train a dedicated model at
the exact critical ratio when it's known in advance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# 标准分位数：覆盖从保守(0.1)到激进(0.9)的订货策略
# 非标准分位数通过相邻模型线性插值获得
STANDARD_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


# 梯度提升分位数预测器：每个分位数训练一个独立模型
# 核心思想：不是预测一个点估计，而是预测需求的整个分布
class GradientBoostedQuantileForecaster:
    """Wrapper around XGBoost/LightGBM for quantile demand forecasting.

    Trains one model per quantile level. For arbitrary quantiles not in the
    pre-trained set, linearly interpolates between the two nearest models.

    Example:
        forecaster = GradientBoostedQuantileForecaster(backend="lightgbm")
        forecaster.train(X_train, y_train)
        demand_p80 = forecaster.predict_quantile(X_test, quantile=0.8)
    """

    def __init__(
        self,
        backend: Literal["xgboost", "lightgbm"] = "lightgbm",
        model_dir: Path | None = None,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
    ) -> None:
        """Initialize the forecaster.

        Args:
            backend: Which gradient boosting library to use.
            model_dir: Directory to save/load model artifacts.
            n_estimators: Number of boosting rounds.
            learning_rate: Step size shrinkage.
            max_depth: Maximum tree depth (XGBoost) or num_leaves proxy (LightGBM).
            subsample: Row sampling ratio.
            colsample_bytree: Column sampling ratio.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
        """
        self._backend = backend
        self._model_dir = model_dir
        self._n_estimators = n_estimators
        # 学习率故意设得低(0.05)，配合500棵树，防止过拟合
        self._learning_rate = learning_rate
        self._max_depth = max_depth
        # 行/列采样率 0.8：每棵树只看 80% 的数据和特征，增加模型多样性
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        # L1/L2 正则化：防止单棵树过于复杂
        self._reg_alpha = reg_alpha
        self._reg_lambda = reg_lambda
        # 键是分位数水平，值是训练好的模型实例
        self._models: dict[float, Any] = {}

    @property
    def is_trained(self) -> bool:
        """Whether any quantile models have been trained."""
        return len(self._models) > 0

    @property
    def trained_quantiles(self) -> tuple[float, ...]:
        """Sorted tuple of quantile levels with trained models."""
        return tuple(sorted(self._models.keys()))

    # 输入：训练数据 + 分位数列表 → 副作用：为每个分位数训练一个独立模型
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        quantiles: tuple[float, ...] | None = None,
    ) -> None:
        """Train one model per quantile level.

        Args:
            x_train: Feature matrix (n_samples, n_features).
            y_train: Target demand values (n_samples,).
            quantiles: Quantile levels to train. Defaults to STANDARD_QUANTILES.
        """
        quantiles = quantiles or STANDARD_QUANTILES

        # 逐个分位数训练独立模型：每个模型最小化对应分位数的弹球损失
        for q in quantiles:
            logger.info("Training %s model at quantile %.3f", self._backend, q)
            model = self._build_model(q)
            model.fit(x_train, y_train)
            self._models[q] = model

        logger.info(
            "Training complete: %d quantile models (%s)",
            len(self._models),
            ", ".join(f"{q:.2f}" for q in sorted(self._models)),
        )

    # 输入：特征 + 目标分位数 → 输出：该分位数下的需求预测值
    def predict_quantile(self, features: np.ndarray, quantile: float) -> float:
        """Predict demand at a given quantile.

        If a model exists at exactly this quantile, use it directly.
        Otherwise, linearly interpolate between the two nearest trained quantiles.

        Args:
            features: Feature vector (n_features,) or matrix (1, n_features).
            quantile: Target quantile in (0, 1).

        Returns:
            Predicted demand at the specified quantile.

        Raises:
            RuntimeError: If no models have been trained.
        """
        if not self._models:
            msg = "No trained models. Call train() first."
            raise RuntimeError(msg)

        feat_2d = features.reshape(1, -1) if features.ndim == 1 else features

        # 精确匹配：生产环境中最常见的情况（已在临界比率处预训练）
        if quantile in self._models:
            pred = self._models[quantile].predict(feat_2d)
            return float(pred[0])

        # === 插值回退：请求的分位数不在预训练集合中 ===
        sorted_qs = sorted(self._models.keys())

        # 请求的分位数低于/高于所有已训练的分位数时，使用边界模型
        if quantile <= sorted_qs[0]:
            pred = self._models[sorted_qs[0]].predict(feat_2d)
            return float(pred[0])
        if quantile >= sorted_qs[-1]:
            pred = self._models[sorted_qs[-1]].predict(feat_2d)
            return float(pred[0])

        # 找到目标分位数的左右邻居
        lower_q = max(q for q in sorted_qs if q <= quantile)
        upper_q = min(q for q in sorted_qs if q >= quantile)

        if lower_q == upper_q:
            pred = self._models[lower_q].predict(feat_2d)
            return float(pred[0])

        lower_pred = float(self._models[lower_q].predict(feat_2d)[0])
        upper_pred = float(self._models[upper_q].predict(feat_2d)[0])

        # 线性插值：假设分位数函数在相邻点之间近似线性
        # 例如 q=0.6 在 q=0.5 和 q=0.75 之间，weight = (0.6-0.5)/(0.75-0.5) = 0.4
        weight = (quantile - lower_q) / (upper_q - lower_q)
        return lower_pred + weight * (upper_pred - lower_pred)

    def predict_mean(self, features: np.ndarray) -> float:
        """Predict median demand (proxy for mean).

        Args:
            features: Feature vector or matrix.

        Returns:
            Median demand prediction.
        """
        # 用中位数近似均值：对于偏态的需求分布，中位数更稳健
        return self.predict_quantile(features, quantile=0.5)

    # 输入：分位数水平 → 输出：配置好的未训练模型实例
    def _build_model(self, quantile: float) -> Any:
        """Construct a gradient boosting model for a specific quantile.

        Args:
            quantile: The target quantile level.

        Returns:
            An unfitted model instance.
        """
        if self._backend == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,  # LightGBM 用 alpha 参数指定分位数水平
                n_estimators=self._n_estimators,
                learning_rate=self._learning_rate,
                # LightGBM 用 num_leaves 控制复杂度，转换公式：2^depth - 1
                num_leaves=2**self._max_depth - 1,
                min_child_samples=20,  # 叶节点最少样本数，防止过拟合
                subsample=self._subsample,
                colsample_bytree=self._colsample_bytree,
                reg_alpha=self._reg_alpha,
                reg_lambda=self._reg_lambda,
                verbose=-1,  # 静默模式，不输出训练日志
            )
        else:
            import xgboost as xgb

            return xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,  # XGBoost 用 quantile_alpha 指定分位数水平
                n_estimators=self._n_estimators,
                learning_rate=self._learning_rate,
                max_depth=self._max_depth,
                subsample=self._subsample,
                colsample_bytree=self._colsample_bytree,
                reg_alpha=self._reg_alpha,
                reg_lambda=self._reg_lambda,
                verbosity=0,
            )
