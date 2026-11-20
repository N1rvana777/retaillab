"""Model training pipeline with enforced time-series discipline.

CRITICAL: This module FORBIDS ordinary cross-validation (KFold, ShuffleSplit)
for time-series data. Only TimeSeriesSplit is allowed, because random splits
create look-ahead bias that makes backtests meaningless.

Usage:
    trainer = TimeSeriesTrainer(n_splits=5)
    results = trainer.train_and_evaluate(model, X, y, dates)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import TimeSeriesSplit

from quant_engine.forecasting.base import ForecastModel
from quant_engine.training.evaluator import BacktestEvaluator, ErrorAnalysisReport

logger = logging.getLogger(__name__)


# 禁止使用非时序交叉验证的自定义异常
# 这是一条硬性规则：随机切分会导致用未来数据训练，使回测结果虚高
class ForbiddenCVError(Exception):
    """Raised when non-temporal cross-validation is attempted on time-series data."""


# 输入：CV 类 → 副作用：如果不是 TimeSeriesSplit 就抛异常
# 这是一道防护墙，防止开发者不小心用了 KFold 等随机切分
def _enforce_time_series_split(cv_class: type) -> None:
    """Guard: reject any CV splitter that isn't TimeSeriesSplit.

    Args:
        cv_class: The cross-validation class being used.

    Raises:
        ForbiddenCVError: If the CV class is not TimeSeriesSplit.
    """
    allowed = {TimeSeriesSplit}
    if cv_class not in allowed:
        msg = (
            f"FORBIDDEN: {cv_class.__name__} is not allowed for time-series data. "
            f"Use sklearn.model_selection.TimeSeriesSplit instead. "
            f"Random splits create look-ahead bias and invalidate all results."
        )
        raise ForbiddenCVError(msg)


# 训练结果：包含每折的评估报告和汇总指标
@dataclass
class TrainResult:
    """Result of a training run across all CV folds."""

    fold_reports: list[ErrorAnalysisReport] = field(default_factory=list)
    best_fold_index: int = 0
    mean_pinball_loss: float = 0.0
    mean_coverage: float = 0.0


# 时序训练器：强制使用 TimeSeriesSplit，保证训练集严格早于测试集
# 这是防止数据穿越的最后一道防线
class TimeSeriesTrainer:
    """Trains and evaluates forecasting models with strict temporal discipline.

    Key guarantees:
    1. Only TimeSeriesSplit is used — never random CV
    2. Each fold's training data strictly precedes its test data
    3. Full error analysis (Top 5% worst) is generated for each fold
    """

    def __init__(
        self,
        n_splits: int = 5,
        cu: float = 5.0,
        co: float = 3.0,
        target_quantile: float = 0.5,
    ) -> None:
        """Initialize the trainer.

        Args:
            n_splits: Number of temporal CV folds.
            cu: Underage cost for evaluation.
            co: Overage cost for evaluation.
            target_quantile: The quantile being optimized.
        """
        # 构造时就验证一次，确保代码逻辑正确
        _enforce_time_series_split(TimeSeriesSplit)  # Self-check

        self._n_splits = n_splits
        # TimeSeriesSplit 保证第 k 折的训练集 = 前 k 份数据，测试集 = 第 k+1 份
        self._cv = TimeSeriesSplit(n_splits=n_splits)
        self._evaluator = BacktestEvaluator(
            cu=cu, co=co, quantile=target_quantile
        )
        self._target_quantile = target_quantile

    # 输入：模型 + 特征矩阵 + 目标值 → 输出：包含每折详细评估的训练结果
    def train_and_evaluate(
        self,
        model: ForecastModel,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        dates: NDArray | None = None,
        metadata: list[dict] | None = None,
    ) -> TrainResult:
        """Train the model using TimeSeriesSplit and evaluate each fold.

        Args:
            model: A ForecastModel to train (must have a .train() method
                if it's a GradientBoostedQuantileForecaster).
            x: Feature matrix (n_samples, n_features).
            y: Target demand values (n_samples,).
            dates: Optional date array for logging.
            metadata: Optional per-sample metadata for error analysis.

        Returns:
            TrainResult with per-fold reports and aggregate metrics.
        """
        result = TrainResult()
        fold_losses: list[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(self._cv.split(x)):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            logger.info(
                "Fold %d/%d: train=%d samples, test=%d samples",
                fold_idx + 1, self._n_splits, len(train_idx), len(test_idx),
            )

            # 二次验证时间顺序：即使 TimeSeriesSplit 保证了，仍然显式检查
            # 因为如果输入数据本身没有按时间排序，TimeSeriesSplit 也会产生穿越
            if dates is not None:
                train_max = dates[train_idx].max()
                test_min = dates[test_idx].min()
                if train_max >= test_min:
                    msg = f"Fold {fold_idx}: train bleeds into test period!"
                    raise ForbiddenCVError(msg)

            # 训练：通过 duck typing 调用 model.train()
            if hasattr(model, "train"):
                model.train(x_train, y_train)  # type: ignore[attr-defined]

            # 逐样本预测（非向量化），因为 predict_quantile 协议返回标量
            y_pred = np.array([
                model.predict_quantile(x_test[i], self._target_quantile)
                for i in range(len(x_test))
            ])

            # 评估：生成包含 Top 5% 最差预测的详细报告
            fold_meta = (
                [metadata[i] for i in test_idx]
                if metadata is not None
                else None
            )
            report = self._evaluator.evaluate(y_test, y_pred, metadata=fold_meta)
            result.fold_reports.append(report)
            fold_losses.append(report.mean_pinball_loss)

            logger.info(
                "Fold %d: pinball=%.4f, coverage=%.3f, MAE=%.2f",
                fold_idx + 1,
                report.mean_pinball_loss,
                report.coverage,
                report.mean_absolute_error,
            )

        # 汇总所有折的指标
        result.mean_pinball_loss = float(np.mean(fold_losses))
        result.mean_coverage = float(np.mean([
            r.coverage for r in result.fold_reports
        ]))
        # 选择弹球损失最低的折作为最佳折（代表模型在最近数据上的表现）
        result.best_fold_index = int(np.argmin(fold_losses))

        logger.info(
            "Training complete: mean_pinball=%.4f, mean_coverage=%.3f, best_fold=%d",
            result.mean_pinball_loss,
            result.mean_coverage,
            result.best_fold_index,
        )

        return result
