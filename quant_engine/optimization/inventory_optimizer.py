"""Core InventoryOptimizer — orchestrates forecasting, Newsvendor optimization, and constraints.

This is the central class of the quant_engine. It ties together:
1. Demand forecasting (via ForecastModel protocol)
2. Newsvendor cost-optimal quantity calculation
3. Operational constraints (MOQ, shelf, truck)
4. Cold-start handling for new products
5. Telemetry logging for every decision

Design decisions:
- Prefers DIRECT quantile regression (train XGBoost at the target quantile)
  over the two-step approach (forecast distribution → invert CDF).
- Falls back to parametric Newsvendor when residuals are well-behaved.
- Uses cold-start heuristics for SKUs with < min_history_days data.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from quant_engine.domain.entities import ReplenishmentOrder, SKU
from quant_engine.domain.enums import OptimizationMethod
from quant_engine.domain.value_objects import CostParameters
from quant_engine.optimization.constraints import OrderConstraints, apply_all_constraints
from quant_engine.optimization.newsvendor import (
    expected_newsvendor_cost,
    robust_std_from_quantiles,
)

if TYPE_CHECKING:
    from datetime import date

    from quant_engine.forecasting.base import ForecastModel

logger = logging.getLogger(__name__)

# 冷启动保守系数：新品订货量打 8 折，避免对未经验证的商品过量进货
_COLD_START_CONSERVATIVE_FACTOR = 0.8

# 标准分位数水平，用于生成置信区间和估算标准差
_STANDARD_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


# 库存优化器：将预测模型、报童模型、业务约束三者串联起来
# 是整个 quant_engine 的核心编排类
class InventoryOptimizer:
    """Computes optimal order quantities using Newsvendor + ML forecasting.

    Usage:
        optimizer = InventoryOptimizer(forecast_model, cold_start_model)
        order = optimizer.optimize(sku, features, cost_params, delivery_date, ...)
    """

    def __init__(
        self,
        forecast_model: ForecastModel,
        cold_start_model: ForecastModel,
        min_history_days: int = 28,
        safety_stock_buffer: float = 0.0,
    ) -> None:
        """Initialize the optimizer.

        Args:
            forecast_model: Primary model for SKUs with sufficient history.
            cold_start_model: Category-level fallback for new SKUs.
            min_history_days: Minimum days of history before using primary model.
            safety_stock_buffer: Additive buffer for strategic items (default 0).
        """
        self._forecast_model = forecast_model
        self._cold_start_model = cold_start_model
        # 28天是经验值：至少需要4周数据才能捕捉到周内模式（周末效应等）
        self._min_history_days = min_history_days
        self._safety_stock_buffer = safety_stock_buffer

    # 输入：SKU + 特征 + 成本参数 + 历史天数 → 输出：补货订单
    def optimize(
        self,
        sku: SKU,
        features: np.ndarray,
        cost_params: CostParameters,
        delivery_date: date,
        history_days: int,
        constraints: OrderConstraints | None = None,
    ) -> ReplenishmentOrder:
        """Compute optimal order quantity for a single SKU-store combination.

        Strategy selection:
        1. history_days < min_history_days → cold start path
        2. Otherwise → direct quantile regression at critical ratio

        Args:
            sku: The SKU to optimize.
            features: Feature vector for this SKU-store-date.
            cost_params: Asymmetric cost structure.
            delivery_date: When the order will arrive.
            history_days: How many days of sales history exist for this SKU.
            constraints: Optional operational constraints.

        Returns:
            ReplenishmentOrder with the recommended quantity and full audit trail.
        """
        # 临界比率 = Cu/(Cu+Co)，决定了应该在需求分布的哪个分位数上订货
        cr = cost_params.critical_ratio
        effective_constraints = constraints or OrderConstraints()

        # 根据历史数据量选择策略：数据不足走冷启动，数据充足走主路径
        if history_days < self._min_history_days:
            return self._cold_start_path(
                sku, features, cost_params, delivery_date, effective_constraints
            )

        return self._primary_path(
            sku, features, cost_params, cr, delivery_date, effective_constraints
        )

    # 输入：SKU列表 + 特征矩阵 → 输出：批量补货订单列表
    def optimize_batch(
        self,
        skus: list[SKU],
        features_matrix: np.ndarray,
        cost_params_list: list[CostParameters],
        delivery_date: date,
        history_days_list: list[int],
        constraints: OrderConstraints | None = None,
    ) -> list[ReplenishmentOrder]:
        """Batch optimization for all SKUs in a store.

        Args:
            skus: List of SKUs to optimize.
            features_matrix: Feature matrix, one row per SKU.
            cost_params_list: Cost parameters for each SKU.
            delivery_date: Delivery date for all orders.
            history_days_list: History days for each SKU.
            constraints: Shared constraints (or per-SKU if extended).

        Returns:
            List of ReplenishmentOrders.
        """
        # 逐个 SKU 调用 optimize，未来可以改为向量化批量预测以提升性能
        return [
            self.optimize(sku, features, cp, delivery_date, hd, constraints)
            for sku, features, cp, hd in zip(
                skus, features_matrix, cost_params_list, history_days_list, strict=True
            )
        ]

    def _primary_path(
        self,
        sku: SKU,
        features: np.ndarray,
        cost_params: CostParameters,
        cr: float,
        delivery_date: date,
        constraints: OrderConstraints,
    ) -> ReplenishmentOrder:
        """Main optimization path: direct quantile regression at critical ratio.

        The XGBoost model predicts demand directly at the optimal quantile
        determined by the Newsvendor critical ratio. This avoids the
        error-prone two-step process of (1) estimating a full distribution
        then (2) inverting its CDF.
        """
        # 确保特征是二维的，因为模型的 predict 接口要求 (n_samples, n_features) 形状
        feat_2d = features.reshape(1, -1) if features.ndim == 1 else features

        # 核心：直接在临界比率对应的分位数上预测需求
        # 这就是最优订货量，不需要再做 F^{-1}(cr) 的转换
        forecast_at_cr = self._forecast_model.predict_quantile(feat_2d, quantile=cr)

        # 同时预测中位数和 10%/90% 分位数，用于置信区间和标准差估算
        forecast_median = self._forecast_model.predict_quantile(feat_2d, quantile=0.5)
        forecast_lower = self._forecast_model.predict_quantile(feat_2d, quantile=0.1)
        forecast_upper = self._forecast_model.predict_quantile(feat_2d, quantile=0.9)

        # 用 10% 和 90% 分位数估算标准差，比样本标准差更稳健
        forecast_std = robust_std_from_quantiles(forecast_lower, forecast_upper)

        # 原始最优量 + 可选的安全库存缓冲（用于战略性商品）
        raw_qty = max(0.0, forecast_at_cr + self._safety_stock_buffer)

        # 将数学最优解转化为实际可执行的订单（考虑 MOQ、货架容量等约束）
        constrained_qty, constraints_applied = apply_all_constraints(raw_qty, constraints)

        # 计算最终订货量下的期望报童成本，用于决策审计和模型评估
        exp_cost = expected_newsvendor_cost(
            order_qty=constrained_qty,
            demand_mean=forecast_median,
            demand_std=forecast_std,
            cu=cost_params.underage_cost,
            co=cost_params.overage_cost,
        )

        return ReplenishmentOrder(
            sku_id=sku.sku_id,
            store_id=sku.store_id,
            # 向上取整：宁可多订一个也不少订（报童成本已经考虑了过量惩罚）
            recommended_qty=int(math.ceil(constrained_qty)),
            delivery_date=delivery_date,
            critical_ratio=cr,
            forecast_mean=float(forecast_median),
            forecast_std=float(forecast_std),
            confidence_lower=float(forecast_lower),
            confidence_upper=float(forecast_upper),
            expected_cost=float(exp_cost),
            method=OptimizationMethod.QUANTILE_DIRECT,
            fallback_used=False,
            constraints_applied=constraints_applied,
        )

    def _cold_start_path(
        self,
        sku: SKU,
        features: np.ndarray,
        cost_params: CostParameters,
        delivery_date: date,
        constraints: OrderConstraints,
    ) -> ReplenishmentOrder:
        """Handle new products with insufficient sales history.

        Uses category-level priors from the cold_start_model (trained on
        aggregate category data). Applies a conservative multiplier (0.8)
        to avoid over-ordering on unproven items.
        """
        cr = cost_params.critical_ratio
        feat_2d = features.reshape(1, -1) if features.ndim == 1 else features

        # 用品类级模型预测（例如"所有面包类"的聚合模型），因为单品数据不足
        forecast_at_cr = self._cold_start_model.predict_quantile(feat_2d, quantile=cr)
        # 打 0.8 折：新品的实际销量往往低于品类均值，保守订货避免积压
        conservative_qty = forecast_at_cr * _COLD_START_CONSERVATIVE_FACTOR

        raw_qty = max(0.0, conservative_qty)
        constrained_qty, constraints_applied = apply_all_constraints(raw_qty, constraints)

        logger.info(
            "Cold start for SKU %s (store %s): category prior=%.1f, conservative=%.1f",
            sku.sku_id,
            sku.store_id,
            forecast_at_cr,
            constrained_qty,
        )

        return ReplenishmentOrder(
            sku_id=sku.sku_id,
            store_id=sku.store_id,
            recommended_qty=int(math.ceil(max(0, constrained_qty))),
            delivery_date=delivery_date,
            critical_ratio=cr,
            forecast_mean=float(forecast_at_cr),
            forecast_std=0.0,  # 冷启动阶段无法可靠估算标准差
            confidence_lower=0.0,
            confidence_upper=float(forecast_at_cr * 1.5),  # 粗略上界：品类预测的 1.5 倍
            expected_cost=0.0,  # 无分布信息，无法计算期望成本
            method=OptimizationMethod.COLD_START,
            fallback_used=False,
            constraints_applied=constraints_applied,
            reasoning=(
                f"New SKU with <{self._min_history_days} days history. "
                f"Using category-level prior with {_COLD_START_CONSERVATIVE_FACTOR:.0%} "
                "conservative factor."
            ),
        )
