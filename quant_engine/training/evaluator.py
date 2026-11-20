"""Model evaluation with enhanced error analysis.

Goes beyond aggregate metrics (MAPE, RMSE) to surface the decisions that
actually cost the most money. Average errors are misleading — a model with
5% MAPE can still cause a single $10,000 stockout if it badly misses a
holiday demand spike.

Key outputs:
1. Standard metrics: Pinball loss, coverage rate, MAPE
2. Top 5% absolute error samples — the worst predictions
3. Top 5% costliest decisions — the most expensive mistakes (in Newsvendor $)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quant_engine.optimization.loss_functions import (
    asymmetric_inventory_cost,
    coverage_rate,
    pinball_loss,
)

logger = logging.getLogger(__name__)


# 单条预测误差记录，用于定位最差的预测样本
@dataclass
class ErrorRecord:
    """A single prediction error for analysis."""

    index: int
    """Row index in the evaluation dataset."""

    actual: float
    predicted: float
    absolute_error: float
    newsvendor_cost: float

    sku_id: str = ""
    store_id: str = ""
    date: str = ""


# 完整的误差分析报告
# 不仅包含平均指标，还包含"最差的那些预测"——因为平均误差可能掩盖极端损失
@dataclass
class ErrorAnalysisReport:
    """Comprehensive error analysis from a backtest run.

    This report surfaces the predictions and decisions that matter most,
    not just aggregate averages.
    """

    # 聚合指标
    mean_pinball_loss: float
    coverage: float
    mean_absolute_error: float
    median_absolute_error: float
    total_newsvendor_cost: float

    # 极端值分析：暴露最差的 5% 样本
    top_error_samples: list[ErrorRecord] = field(default_factory=list)
    """Top 5% samples by absolute prediction error."""

    top_cost_samples: list[ErrorRecord] = field(default_factory=list)
    """Top 5% samples by realized Newsvendor cost (the real money impact)."""

    # 误差分布统计
    error_percentiles: dict[str, float] = field(default_factory=dict)
    """Error distribution: p50, p90, p95, p99."""

    quantile_level: float = 0.5
    n_samples: int = 0


# 回测评估器：用真实需求评估预测和订货决策的好坏
# 核心思想：不只看预测准不准，更看订货决策亏不亏钱
class BacktestEvaluator:
    """Evaluates forecast + ordering decisions against actual demand.

    Usage:
        evaluator = BacktestEvaluator(cu=5.0, co=3.0, quantile=0.65)
        report = evaluator.evaluate(y_true, y_pred, order_qtys)
    """

    def __init__(
        self,
        cu: float,
        co: float,
        quantile: float,
        top_pct: float = 0.05,
    ) -> None:
        """Initialize the evaluator.

        Args:
            cu: Underage cost per unit (for Newsvendor cost calculation).
            co: Overage cost per unit.
            quantile: The quantile level being evaluated.
            top_pct: Fraction of worst samples to surface (default 5%).
        """
        self._cu = cu
        self._co = co
        self._quantile = quantile
        # 只展示最差的 5%，因为长尾中的极端案例才是真正的风险来源
        self._top_pct = top_pct

    # 输入：真实需求 + 预测值 + 订货量 → 输出：包含极端值分析的完整报告
    def evaluate(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        order_qtys: NDArray[np.float64] | None = None,
        metadata: list[dict] | None = None,
    ) -> ErrorAnalysisReport:
        """Run full evaluation and generate error analysis report.

        Args:
            y_true: Actual demand values.
            y_pred: Predicted demand values (at the target quantile).
            order_qtys: Actual order quantities. If None, uses y_pred as proxy.
            metadata: Optional list of dicts with 'sku_id', 'store_id', 'date'
                for each sample, enabling identification of worst performers.

        Returns:
            Comprehensive ErrorAnalysisReport.
        """
        # 如果没有实际订货量，用预测值代替（假设"按预测订货"）
        if order_qtys is None:
            order_qtys = y_pred

        n = len(y_true)
        abs_errors = np.abs(y_true - y_pred)
        # 计算每个样本的实际报童成本：这是评估决策质量的关键指标
        nv_costs = asymmetric_inventory_cost(y_true, order_qtys, self._cu, self._co)

        # 聚合指标
        mean_pl = pinball_loss(y_true, y_pred, self._quantile)
        cov = coverage_rate(y_true, y_pred)
        mae = float(np.mean(abs_errors))
        mdae = float(np.median(abs_errors))
        total_cost = float(np.sum(nv_costs))

        # 误差分位数分布：p90/p95/p99 比均值更能反映尾部风险
        error_pcts = {
            "p50": float(np.percentile(abs_errors, 50)),
            "p90": float(np.percentile(abs_errors, 90)),
            "p95": float(np.percentile(abs_errors, 95)),
            "p99": float(np.percentile(abs_errors, 99)),
        }

        # 取最差的 Top K 个样本（至少 1 个）
        k = max(1, int(n * self._top_pct))
        meta = metadata or [{}] * n

        # 按绝对误差排序：找出模型预测最不准的样本
        top_error_indices = np.argsort(abs_errors)[-k:][::-1]
        top_errors = [
            self._make_record(i, y_true, y_pred, abs_errors, nv_costs, meta)
            for i in top_error_indices
        ]

        # 按报童成本排序：找出造成最大经济损失的决策
        # 这可能和误差最大的不是同一批——高误差但低成本的预测可能无害
        top_cost_indices = np.argsort(nv_costs)[-k:][::-1]
        top_costs = [
            self._make_record(i, y_true, y_pred, abs_errors, nv_costs, meta)
            for i in top_cost_indices
        ]

        logger.info(
            "Backtest: n=%d, pinball=%.3f, coverage=%.3f, MAE=%.2f, total_cost=$%.2f",
            n, mean_pl, cov, mae, total_cost,
        )
        logger.info(
            "Worst 5%%: max_error=%.2f, max_cost=$%.2f",
            abs_errors[top_error_indices[0]] if len(top_error_indices) > 0 else 0,
            nv_costs[top_cost_indices[0]] if len(top_cost_indices) > 0 else 0,
        )

        return ErrorAnalysisReport(
            mean_pinball_loss=mean_pl,
            coverage=cov,
            mean_absolute_error=mae,
            median_absolute_error=mdae,
            total_newsvendor_cost=total_cost,
            top_error_samples=top_errors,
            top_cost_samples=top_costs,
            error_percentiles=error_pcts,
            quantile_level=self._quantile,
            n_samples=n,
        )

    # 输入：样本索引 + 各种数组 → 输出：该样本的 ErrorRecord
    @staticmethod
    def _make_record(
        idx: int,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        abs_errors: NDArray[np.float64],
        nv_costs: NDArray[np.float64],
        metadata: list[dict],
    ) -> ErrorRecord:
        """Create an ErrorRecord for a specific sample index."""
        meta = metadata[idx] if idx < len(metadata) else {}
        return ErrorRecord(
            index=int(idx),
            actual=float(y_true[idx]),
            predicted=float(y_pred[idx]),
            absolute_error=float(abs_errors[idx]),
            newsvendor_cost=float(nv_costs[idx]),
            sku_id=meta.get("sku_id", ""),
            store_id=meta.get("store_id", ""),
            date=meta.get("date", ""),
        )
