"""Product and business domain enumerations."""

from enum import StrEnum


# 商品品类分类：每种品类隐含不同的成本结构和优化策略
class ProductCategory(StrEnum):
    """Classification of SKUs by inventory management strategy.

    Each category implies a different cost structure and optimization behavior:
    - PERISHABLE: High overage cost (spoilage), conservative ordering
    - STAPLE: High underage cost (lost sales + churn), aggressive ordering
    - SEASONAL: Regime-dependent demand, requires changepoint detection
    - PROMOTIONAL: Temporary items with artificial demand spikes
    """

    PERISHABLE = "perishable"    # 生鲜类：过量成本高（变质），宁可少订
    STAPLE = "staple"            # 日用品：缺货成本高（客户流失），宁可多订
    SEASONAL = "seasonal"        # 季节性商品：需求模式随季节切换
    PROMOTIONAL = "promotional"  # 促销品：人为制造的需求尖峰，需特殊处理


# 概念漂移告警级别
class DriftSeverity(StrEnum):
    """Concept drift alert severity levels."""

    NORMAL = "normal"           # 正常：损失在基线波动范围内
    WARNING = "warning"         # 警告：偏离基线 2σ，可能是漂移早期信号
    CRITICAL = "critical"       # 严重：偏离基线 3σ，几乎确定发生了漂移，必须重新训练


# 订货量的决策方法：记录每笔订单是怎么算出来的
class OptimizationMethod(StrEnum):
    """How the final order quantity was determined."""

    QUANTILE_DIRECT = "quantile_direct"           # 直接分位数回归：XGBoost 在临界比率分位数上预测
    PARAMETRIC_NEWSVENDOR = "parametric_newsvendor"  # 参数化报童：先估分布再求逆 CDF
    COLD_START = "cold_start"                      # 冷启动：品类级先验，用于新品
    FALLBACK_SAFETY_STOCK = "fallback_safety_stock"  # 降级模式：7天移动平均 * 安全系数
