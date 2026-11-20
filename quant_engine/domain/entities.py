"""Domain entities — objects with identity and lifecycle.

Unlike value objects, entities are identified by their IDs, not their attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from quant_engine.domain.enums import OptimizationMethod, ProductCategory


# SKU 实体：被预测和订货的最小单元，由 sku_id + store_id 唯一标识
@dataclass
class SKU:
    """A Stock Keeping Unit — the atomic item being forecasted and ordered."""

    sku_id: str
    store_id: str
    name: str
    category: ProductCategory
    unit_of_measure: str = "units"
    """E.g., 'units', 'kg', 'liters'."""

    shelf_life_days: int = 365
    # 替代品组：同组商品可以互相替代（如可口可乐和百事可乐）
    # 未来可用于需求转移建模：A 缺货时客户可能转买 B
    substitution_group: str | None = None
    """Group ID for substitute products (e.g., Coke and Pepsi)."""


# 门店实体：一个物理零售网点
@dataclass
class Store:
    """A physical retail location."""

    store_id: str
    name: str
    region: str
    # 营业时间影响日内需求分布（如夜间需求为零）
    operating_hours: tuple[int, int] = (8, 22)
    """(open_hour, close_hour) in 24h format."""


# 需求预测结果：某个 SKU-门店-日期组合的预测值
@dataclass
class DemandForecast:
    """A demand prediction for a specific SKU-store-date combination."""

    sku_id: str
    store_id: str
    target_date: date
    predicted_mean: float
    predicted_std: float
    # 多分位数预测结果，键是分位数水平，值是预测需求量
    # 例如 {0.1: 5.0, 0.5: 12.0, 0.9: 20.0} 表示10%概率需求<5，50%概率需求<12
    quantiles: dict[float, float] = field(default_factory=dict)
    """Mapping of quantile level → predicted value. E.g., {0.1: 5.0, 0.5: 12.0, 0.9: 20.0}."""

    model_version: str = "unknown"


# 补货订单：系统最终输出，告诉门店"这个SKU该订多少"
@dataclass
class ReplenishmentOrder:
    """The final output: how much to order for a single SKU."""

    sku_id: str
    store_id: str
    recommended_qty: int
    delivery_date: date
    # 决策依据的临界比率，便于事后审计"为什么订这么多/少"
    critical_ratio: float
    forecast_mean: float
    forecast_std: float
    confidence_lower: float
    confidence_upper: float
    expected_cost: float
    # 记录决策方法，便于追踪哪些SKU用了降级路径
    method: OptimizationMethod
    fallback_used: bool = False
    # 记录哪些约束生效了，便于分析约束对订货量的影响
    constraints_applied: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    # 人类可读的决策理由，用于运营人员理解"这个数字怎么来的"
    reasoning: str | None = None
    """Human-readable explanation of why this quantity was chosen."""
