"""Asymmetric loss functions for inventory optimization.

These loss functions encode the fundamental insight that in retail,
the cost of over-ordering (spoilage, markdowns) and under-ordering
(lost sales, customer churn) are NOT symmetric.

The system optimizes for realized Newsvendor cost, NOT RMSE or MAPE.
A biased model that matches the critical ratio is preferred over an
unbiased model that minimizes symmetric error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# 输入：真实值 + 预测值 + 分位数水平 → 输出：平均弹球损失
# 弹球损失是分位数回归的标准评分函数，XGBoost 训练时内部也在最小化这个损失
def pinball_loss(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    quantile: float,
) -> float:
    """Pinball (quantile) loss — the proper scoring rule for quantile forecasts.

    L_q(y, ŷ) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)

    This is the loss function that XGBoost/LightGBM minimize internally
    when trained with objective='quantile'. We use it for evaluation
    and backtesting to verify model calibration.

    A well-calibrated model at quantile q should have ~q fraction of
    actual values below the prediction.

    Args:
        y_true: Actual demand values.
        y_pred: Predicted demand values.
        quantile: Target quantile in (0, 1).

    Returns:
        Mean pinball loss across all samples.
    """
    residual = y_true - y_pred
    # 关键的非对称性：预测偏低（residual > 0）时惩罚权重为 q，
    # 预测偏高（residual < 0）时惩罚权重为 (1-q)
    # 例如 q=0.8 时，少预测的惩罚是多预测的 4 倍，鼓励模型预测偏高
    return float(np.mean(
        np.where(residual >= 0, quantile * residual, (quantile - 1.0) * residual)
    ))


# 输入：实际需求 + 订货量 + 缺货/过剩成本 → 输出：每期的实际报童成本
# 这是事后（ex-post）成本，用于回测评估决策质量
def asymmetric_inventory_cost(
    demand: NDArray[np.float64],
    order_qty: NDArray[np.float64],
    cu: float,
    co: float,
) -> NDArray[np.float64]:
    """Realized Newsvendor cost per period.

    Cost_t = Cu * max(D_t - Q_t, 0) + Co * max(Q_t - D_t, 0)

    This is the *realized* cost (ex-post), not the expected cost (ex-ante).
    Use this for backtesting to measure how much money a decision actually cost.

    Args:
        demand: Actual demand per period.
        order_qty: Ordered quantity per period.
        cu: Underage cost per unit (lost margin + goodwill).
        co: Overage cost per unit (purchase - salvage + holding).

    Returns:
        Array of costs, one per period.
    """
    # 缺货成本：需求超出订货量的部分，每单位损失 Cu（利润损失 + 客户流失）
    underage = np.maximum(demand - order_qty, 0.0) * cu
    # 过剩成本：订货量超出需求的部分，每单位损失 Co（进价 - 残值 + 仓储）
    overage = np.maximum(order_qty - demand, 0.0) * co
    return underage + overage


# 输入：需求 + 订货量 + 成本 + 保质期信息 → 输出：含生鲜折价的总成本
# 针对生鲜类商品：剩余库存会随时间贬值，不能简单用固定过剩成本衡量
def spoilage_adjusted_cost(
    demand: NDArray[np.float64],
    order_qty: NDArray[np.float64],
    cu: float,
    co: float,
    shelf_life_days: int,
    daily_decay_rate: float = 0.1,
) -> NDArray[np.float64]:
    """Extended cost model for perishables with time-dependent quality decay.

    Leftover inventory doesn't just sit on the shelf — it depreciates daily.
    Fresh bread on day 1 might sell at full price, but on day 3 it needs a
    50% markdown, and by day 5 it's waste.

    The decay multiplier models the cumulative markdown pressure:
        markdown_cost = leftover * Co * Σ(decay_rate^d) / shelf_life

    Args:
        demand: Actual demand per period.
        order_qty: Ordered quantity per period.
        cu: Underage cost per unit.
        co: Overage cost per unit (base, before decay adjustment).
        shelf_life_days: Days until product expires.
        daily_decay_rate: Quality loss rate per day (0.1 = 10%/day).

    Returns:
        Array of total costs including markdown pressure.
    """
    # 先计算基础的非对称成本
    base_cost = asymmetric_inventory_cost(demand, order_qty, cu, co)

    # 剩余库存量（卖不掉的部分）
    leftover = np.maximum(order_qty - demand, 0.0)

    # 等比级数求和：每天品质衰减 r，第 d 天的衰减累积为 r^d
    # 例如 decay_rate=0.1, shelf_life=5: Σ = 0.1 + 0.01 + 0.001 + ... ≈ 0.11
    if daily_decay_rate <= 0 or shelf_life_days <= 0:
        return base_cost

    decay_multiplier = sum(
        daily_decay_rate**d for d in range(1, shelf_life_days + 1)
    )
    # 额外的降价损失：剩余库存 * 基础过剩成本 * 衰减系数 / 保质期天数
    # 除以保质期是为了将累积衰减均摊到每天
    markdown_cost = leftover * co * decay_multiplier / shelf_life_days

    return base_cost + markdown_cost


# 输入：真实值 + 分位数预测值 → 输出：覆盖率（实际值低于预测值的比例）
# 校准检查：q=0.8 的模型，覆盖率应接近 80%
def coverage_rate(
    y_true: NDArray[np.float64],
    y_pred_quantile: NDArray[np.float64],
) -> float:
    """Fraction of actual values at or below the predicted quantile.

    For a well-calibrated model at quantile q, this should be ≈ q.
    E.g., a model predicting at the 0.8 quantile should have ~80% of
    actual values below its predictions.

    Args:
        y_true: Actual demand values.
        y_pred_quantile: Predicted values at some quantile level.

    Returns:
        Coverage rate in [0, 1].
    """
    return float(np.mean(y_true <= y_pred_quantile))
