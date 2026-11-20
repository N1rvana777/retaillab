"""Safety stock fallback — the last line of defense when the ML engine fails.

If the XGBoost engine crashes due to corrupted features, missing model files,
or unexpected data, the system MUST NOT return a 500 error. Instead, it
degrades gracefully to this simple but reliable rule:

    order_qty = moving_average(last_N_days) * safety_factor

This is intentionally simple. When the smart system fails, you want
the dumbest possible thing that still works — not another complex system
that might also fail.

Usage:
    fallback = SafetyStockFallback(window_days=7, safety_factor=1.2)
    qty = fallback.compute(recent_sales=[10, 12, 8, 15, 11, 9, 13])
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from quant_engine.domain.entities import ReplenishmentOrder
from quant_engine.domain.enums import OptimizationMethod

logger = logging.getLogger(__name__)


# 品类默认日均需求：连销售数据都没有时的终极兜底值
@dataclass(frozen=True)
class CategoryDefault:
    """Default daily demand for a product category when no data exists at all."""

    category: str
    default_daily_qty: float


# 品类默认值字典——生产环境应该从配置文件覆盖这些值
_CATEGORY_DEFAULTS: dict[str, float] = {
    "perishable": 15.0,    # 生鲜类日均 15 件
    "staple": 30.0,        # 日用品日均 30 件
    "seasonal": 10.0,      # 季节性商品日均 10 件
    "promotional": 20.0,   # 促销品日均 20 件
}


# 安全库存降级器：ML引擎失败时的最后防线
# 设计原则：越简单越好——复杂的降级方案可能也会出错
class SafetyStockFallback:
    """Computes a simple safety-stock-based order quantity.

    This is the graceful degradation path. It requires only recent sales
    history (no features, no model, no training). If even that isn't
    available, it falls back to category-level defaults.
    """

    def __init__(
        self,
        window_days: int = 7,
        safety_factor: float = 1.2,
        category_defaults: dict[str, float] | None = None,
    ) -> None:
        """Initialize the fallback.

        Args:
            window_days: Number of recent days for moving average.
            safety_factor: Multiplier on top of moving average.
                1.0 = order exactly the average, 1.2 = order 20% extra buffer.
            category_defaults: Default daily demand by category.
                Used when no sales history is available at all.
        """
        # 7天窗口：刚好一周，能反映周内销售模式
        self._window_days = window_days
        # 安全系数 1.2：比均值多订 20%，用过量成本换取服务水平
        self._safety_factor = safety_factor
        self._category_defaults = category_defaults or _CATEGORY_DEFAULTS

    # 输入：近期销售数据 + 品类 → 输出：降级补货量（连续值）
    def compute(
        self,
        recent_sales: list[float] | np.ndarray | None,
        category: str = "staple",
    ) -> float:
        """Compute fallback order quantity.

        Args:
            recent_sales: Last N days of sales data. May be None or empty.
            category: Product category for default lookup.

        Returns:
            Recommended order quantity (continuous). Caller rounds as needed.
        """
        if recent_sales is not None and len(recent_sales) > 0:
            sales = np.array(recent_sales, dtype=np.float64)
            # 只取最近 window_days 天，更早的数据可能已经过时
            window = sales[-self._window_days:]
            avg = float(np.mean(window))
        else:
            # 完全没有销售数据——使用品类默认值
            avg = self._category_defaults.get(
                category, self._category_defaults.get("staple", 10.0)
            )
            logger.warning(
                "No sales history available, using category default: %.1f for '%s'",
                avg,
                category,
            )

        # 移动平均 * 安全系数，且不低于 0
        return max(0.0, avg * self._safety_factor)

    # 输入：SKU/门店信息 + 销售数据 → 输出：完整的补货订单对象
    # 生成的订单和正常订单格式完全一致，API 层无需特殊处理
    def compute_order(
        self,
        sku_id: str,
        store_id: str,
        recent_sales: list[float] | np.ndarray | None,
        category: str = "staple",
        delivery_date: str = "",
        error_message: str = "",
    ) -> ReplenishmentOrder:
        """Compute a full ReplenishmentOrder using the fallback method.

        This produces a complete order object that can be returned through
        the API, indistinguishable from a normal order except for the
        `method` and `fallback_used` fields.

        Args:
            sku_id: SKU identifier.
            store_id: Store identifier.
            recent_sales: Recent daily sales.
            category: Product category.
            delivery_date: Expected delivery date string.
            error_message: Why the primary engine failed.

        Returns:
            ReplenishmentOrder with fallback_used=True.
        """
        from datetime import date as date_type

        qty = self.compute(recent_sales, category)

        # 尝试从销售数据中估算基本统计量，用于填充订单的置信区间字段
        if recent_sales is not None and len(recent_sales) > 0:
            sales = np.array(recent_sales, dtype=np.float64)
            mean = float(np.mean(sales))
            # ddof=1：贝塞尔修正，样本量为1时无法计算标准差
            std = float(np.std(sales, ddof=1)) if len(sales) > 1 else 0.0
        else:
            mean = self._category_defaults.get(category, 10.0)
            std = 0.0

        # 解析交付日期，格式异常时用今天作为兜底
        try:
            d_date = date_type.fromisoformat(delivery_date) if delivery_date else date_type.today()
        except ValueError:
            d_date = date_type.today()

        return ReplenishmentOrder(
            sku_id=sku_id,
            store_id=store_id,
            recommended_qty=int(math.ceil(qty)),
            delivery_date=d_date,
            critical_ratio=0.5,  # 降级路径下不知道真实成本结构，用对称假设
            forecast_mean=mean,
            forecast_std=std,
            # 置信区间用 ±2σ 的粗略估计
            confidence_lower=max(0.0, mean - 2 * std),
            confidence_upper=mean + 2 * std,
            expected_cost=0.0,  # 无成本参数，无法估算期望成本
            method=OptimizationMethod.FALLBACK_SAFETY_STOCK,
            fallback_used=True,
            constraints_applied=[],
            reasoning=(
                f"Primary engine failed: {error_message}. "
                f"Using {self._window_days}-day moving average * {self._safety_factor} safety factor."
            ),
        )
