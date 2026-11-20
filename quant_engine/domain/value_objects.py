"""Immutable value objects for the quant engine domain.

These are frozen dataclasses representing concepts without identity —
two CostParameters with the same values are considered equal.
"""

from __future__ import annotations

from dataclasses import dataclass


# 非对称成本参数：报童模型的核心输入
# frozen=True 保证不可变，同样参数值的两个实例被视为相等
@dataclass(frozen=True)
class CostParameters:
    """Asymmetric cost structure for a single SKU.

    Encodes the Newsvendor Model's core insight: the cost of ordering too much
    (overage) vs. too little (underage) is rarely symmetric. The critical_ratio
    derived from these costs determines the optimal quantile for ordering.

    For perishables (bakery, fruit):
        - salvage_value ≈ 0, shelf_life_days is short
        - overage_cost is HIGH → critical_ratio < 0.5 → order conservatively

    For staples (rice, canned goods):
        - goodwill_penalty is HIGH (customer churn from stockouts)
        - underage_cost is HIGH → critical_ratio > 0.5 → order aggressively
    """

    unit_cost: float
    """Purchase price per unit (c)."""

    selling_price: float
    """Retail price per unit (p)."""

    salvage_value: float = 0.0
    """Recovery per unsold unit (v). Zero for full spoilage."""

    goodwill_penalty: float = 0.0
    """Intangible cost of a lost sale (g). Proxy for customer churn risk."""

    holding_cost_per_day: float = 0.0
    """Storage cost per unit per day."""

    shelf_life_days: int = 365
    """Remaining shelf life at delivery time. Short for perishables."""

    def __post_init__(self) -> None:
        # 售价低于成本意味着每卖一个都在亏钱，很可能是参数配置错误
        if self.selling_price < self.unit_cost:
            msg = (
                f"selling_price ({self.selling_price}) < unit_cost ({self.unit_cost}): "
                "negative margin SKU — verify cost parameters"
            )
            raise ValueError(msg)
        # 残值高于进价在经济学上不合理（除非有补贴）
        if self.salvage_value > self.unit_cost:
            msg = (
                f"salvage_value ({self.salvage_value}) > unit_cost ({self.unit_cost}): "
                "salvage cannot exceed purchase cost"
            )
            raise ValueError(msg)

    @property
    def underage_cost(self) -> float:
        """Cu = lost margin + goodwill penalty.

        The cost incurred for each unit of demand that goes unfulfilled.
        """
        # Cu = 缺货成本 = 损失的利润（售价-进价）+ 商誉损失（客户流失风险）
        return (self.selling_price - self.unit_cost) + self.goodwill_penalty

    @property
    def overage_cost(self) -> float:
        """Co = (purchase cost - salvage) + holding cost over shelf life.

        The cost incurred for each unit ordered in excess of demand.
        """
        # Co = 过剩成本 = 净损耗（进价-残值）+ 仓储成本（单价×天数）
        # 对生鲜来说残值≈0且保质期短，所以 Co 主要是进价本身
        return (self.unit_cost - self.salvage_value) + (
            self.holding_cost_per_day * self.shelf_life_days
        )

    @property
    def critical_ratio(self) -> float:
        """Newsvendor critical ratio: Cu / (Cu + Co).

        This is the optimal quantile of the demand distribution at which to order.
        The order quantity q* = F^{-1}(critical_ratio).

        Returns a value in (0, 1). Values > 0.5 mean "order aggressively"
        (underage is more expensive). Values < 0.5 mean "order conservatively"
        (overage is more expensive).
        """
        cu = self.underage_cost
        co = self.overage_cost
        total = cu + co
        # 临界比率 = Cu/(Cu+Co) = 缺货成本占总成本的比例
        # 直觉：缺货越贵，临界比率越高，应该在更高的分位数上订货（多订）
        # 例如 Cu=8, Co=2 → cr=0.8，意味着应该覆盖 80% 的需求场景
        if total <= 0:
            return 0.5  # 退化情况：成本为零时用对称假设
        return cu / total

    @property
    def margin(self) -> float:
        """Gross margin per unit sold."""
        # 毛利 = 售价 - 进价
        return self.selling_price - self.unit_cost


# 分位数目标：将分位数水平和对应的预测值打包在一起
# 用于在预测模块和优化模块之间传递分位数回归结果
@dataclass(frozen=True)
class QuantileTarget:
    """A specific quantile level with its associated prediction.

    Used to pass quantile regression results between forecasting and optimization.
    """

    level: float
    """Quantile level in (0, 1). E.g., 0.5 for median, 0.9 for 90th percentile."""

    value: float
    """Predicted demand at this quantile level."""

    def __post_init__(self) -> None:
        # 分位数水平必须在 (0,1) 之间，0 和 1 没有意义（对应 -∞ 和 +∞）
        if not 0 < self.level < 1:
            msg = f"Quantile level must be in (0, 1), got {self.level}"
            raise ValueError(msg)
