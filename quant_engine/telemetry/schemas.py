"""Structured telemetry schemas for decision audit trail.

Every replenishment decision is captured as a DecisionSnapshot,
enabling post-hoc debugging ("Why did we order 200 boxes of milk on Tuesday?")
and systematic performance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# 决策快照：完整记录一次补货决策的所有输入、输出和上下文
# 写入 JSONL 日志（一行一条），是事后调试和合规审查的主要依据
@dataclass
class DecisionSnapshot:
    """Complete audit trail of a single replenishment decision.

    This is written to JSONL logs (one line per decision) and is the
    primary artifact for debugging and compliance review.
    """

    # === 身份标识 ===
    timestamp: datetime
    sku_id: str
    store_id: str
    delivery_date: str

    # === 输入特征快照（决策时刻冻结） ===
    input_features: dict[str, float]
    """Feature vector snapshot. Keys are feature names, values are the actual
    values used for this prediction. Enables exact replay."""

    # === 模型输出 ===
    predicted_quantiles: dict[str, float]
    """Quantile predictions. Keys like '0.1', '0.5', '0.9', 'cr'.
    'cr' is the prediction at the critical ratio quantile."""

    # 报童临界比率 Cu/(Cu+Co)，决定了在需求分布的哪个分位数上订货
    critical_ratio: float
    """The Newsvendor critical ratio Cu/(Cu+Co) that determined the target quantile."""

    # === 优化过程追踪 ===
    # 约束前的原始最优量，用于分析约束对决策的影响
    raw_qty: float
    """Unconstrained optimal quantity before any business rules."""

    # 记录哪些约束生效了，便于分析约束是否过于激进
    constraints_applied: list[str]
    """Which constraints were binding. E.g., ['moq:24', 'shelf_capacity:100']."""

    # 最终推荐量（经过所有约束调整后）
    final_qty: int
    """Final recommended order quantity after all constraints."""

    # === 决策方法和降级标记 ===
    method: str
    """How the quantity was determined: 'quantile_direct', 'cold_start', 'fallback_safety_stock'."""

    # 是否使用了降级路径——如果频繁出现 True，说明主引擎有系统性问题
    fallback_used: bool
    """True if the primary engine failed and we fell back to safety stock."""

    # === 成本估算 ===
    expected_cost: float = 0.0
    """Estimated Newsvendor cost at the recommended quantity."""

    # === 元数据 ===
    model_version: str = "unknown"
    # 引擎耗时：用于监控性能退化，如果突然变慢可能是特征计算出了问题
    engine_latency_ms: float = 0.0
    """Time taken by the optimization engine in milliseconds."""

    # 如果触发了降级，记录原始错误信息便于排查
    error_message: str | None = None
    """If fallback was triggered, the original error message."""

    # 输入：无 → 输出：扁平化的字典，可直接序列化为 JSON
    def to_dict(self) -> dict:
        """Serialize to a flat dictionary for JSONL output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "sku_id": self.sku_id,
            "store_id": self.store_id,
            "delivery_date": self.delivery_date,
            "input_features": self.input_features,
            "predicted_quantiles": self.predicted_quantiles,
            "critical_ratio": self.critical_ratio,
            "raw_qty": self.raw_qty,
            "constraints_applied": self.constraints_applied,
            "final_qty": self.final_qty,
            "method": self.method,
            "fallback_used": self.fallback_used,
            "expected_cost": self.expected_cost,
            "model_version": self.model_version,
            "engine_latency_ms": self.engine_latency_ms,
            "error_message": self.error_message,
        }
