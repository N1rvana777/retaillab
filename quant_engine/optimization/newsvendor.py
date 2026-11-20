"""Newsvendor Model solver — the mathematical foundation of the system.

The Newsvendor (or "newspaper boy") problem determines the optimal order quantity
when demand is uncertain and costs are asymmetric:

    q* = F^{-1}(Cu / (Cu + Co))

where:
    Cu = underage cost (cost per unit of unmet demand)
    Co = overage cost (cost per unit of excess inventory)
    F  = cumulative distribution function of demand

References:
    - Newsvendor Model: https://en.wikipedia.org/wiki/Newsvendor_model
    - Ban & Rudin (2019), "The Big Data Newsvendor", Operations Research
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy import stats


# 需求分布协议：任何实现了 ppf（逆CDF）和 cdf 方法的对象都可以作为需求分布
# 这样可以使用 scipy.stats 的任何分布，也可以用自定义分布
@runtime_checkable
class DemandDistribution(Protocol):
    """Any object implementing ppf and cdf (scipy.stats compatible)."""

    def ppf(self, q: float) -> float:
        """Percent-point function (inverse CDF)."""
        ...

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        ...


# 输入：临界比率 + 需求分布 → 输出：最优订货量（连续值）
# 这是报童模型的核心公式：q* = F^{-1}(Cu/(Cu+Co))
def newsvendor_optimal_quantity(
    critical_ratio: float,
    demand_distribution: DemandDistribution,
) -> float:
    """Classic Newsvendor: q* = F^{-1}(critical_ratio).

    Args:
        critical_ratio: Cu / (Cu + Co), must be in (0, 1).
        demand_distribution: Scipy-compatible distribution with ppf().

    Returns:
        Optimal order quantity (continuous). Caller must round to int/MOQ.

    Raises:
        ValueError: If critical_ratio is outside (0, 1).
    """
    # 临界比率必须严格在 (0,1) 之间，否则 ppf 无意义
    if not 0 < critical_ratio < 1:
        msg = f"critical_ratio must be in (0, 1), got {critical_ratio}"
        raise ValueError(msg)

    # 通过逆CDF求最优订货量：在需求分布的临界比率分位数处订货
    # 含义：恰好使"多订1个的边际成本 = 少订1个的边际成本"
    q_star = float(demand_distribution.ppf(critical_ratio))
    # 订货量不能为负
    return max(0.0, q_star)


# 输入：已有分位数预测 + 目标临界比率 → 输出：调整后的订货量
# 当 XGBoost 预测的分位数和报童最优分位数不一致时，用正态近似做线性调整
def newsvendor_from_quantile_forecast(
    predicted_quantile_value: float,
    predicted_quantile_level: float,
    critical_ratio: float,
    demand_std: float,
) -> float:
    """Adjust a quantile forecast to the target critical ratio.

    When XGBoost predicts demand at quantile `predicted_quantile_level`
    but the optimal quantile is `critical_ratio`, we shift using the
    normal approximation:

        q* = predicted_value + (z_target - z_predicted) * std

    This is a fallback for when direct quantile regression at the exact
    critical ratio is not available. Prefer training the model directly
    at the critical ratio quantile when possible.

    Args:
        predicted_quantile_value: The forecast at predicted_quantile_level.
        predicted_quantile_level: The quantile level of the existing forecast.
        critical_ratio: The target quantile (from Newsvendor cost structure).
        demand_std: Estimated standard deviation of demand.

    Returns:
        Adjusted order quantity, floored at 0.
    """
    # 标准差为零说明需求确定，无需调整
    if demand_std <= 0:
        return max(0.0, predicted_quantile_value)

    # 将两个分位数分别转化为标准正态的 z 值
    z_predicted = stats.norm.ppf(predicted_quantile_level)
    z_target = stats.norm.ppf(critical_ratio)
    # 利用正态分布的线性性质：不同分位数之间的差 = z 值差 * 标准差
    adjustment = (z_target - z_predicted) * demand_std

    return max(0.0, predicted_quantile_value + adjustment)


# 输入：订货量 + 需求分布参数 + 成本参数 → 输出：期望总成本
# 用于评估一个订货决策的好坏（成本越低越好）
def expected_newsvendor_cost(
    order_qty: float,
    demand_mean: float,
    demand_std: float,
    cu: float,
    co: float,
) -> float:
    """Analytical expected Newsvendor cost assuming normally distributed demand.

    E[Cost] = Cu * E[max(D - Q, 0)] + Co * E[max(Q - D, 0)]

    For normal demand:
        E[max(D - Q, 0)] = σ * [φ(z) - z * (1 - Φ(z))]
        E[max(Q - D, 0)] = (Q - μ) * Φ(z) + σ * φ(z)

    where z = (Q - μ) / σ, φ is the standard normal PDF, Φ is the CDF.

    Args:
        order_qty: Planned order quantity Q.
        demand_mean: Mean of demand distribution μ.
        demand_std: Standard deviation of demand σ.
        cu: Underage cost per unit.
        co: Overage cost per unit.

    Returns:
        Expected total cost. Returns 0 if demand_std <= 0.
    """
    if demand_std <= 0:
        # 需求确定（无随机性）时，直接计算确定性的过量/缺量成本
        underage = max(demand_mean - order_qty, 0) * cu
        overage = max(order_qty - demand_mean, 0) * co
        return underage + overage

    # 标准化 z 值：订货量偏离需求均值多少个标准差
    z = (order_qty - demand_mean) / demand_std
    # φ(z) 是标准正态的概率密度函数值
    phi_z = stats.norm.pdf(z)
    # Φ(z) 是标准正态的累积分布函数值
    big_phi_z = stats.norm.cdf(z)

    # E[max(D - Q, 0)] 是期望缺货量：需求超过订货量的期望值
    # 公式含义：σ * [φ(z) - z * (1 - Φ(z))]，这是截断正态分布的期望
    expected_underage = demand_std * (phi_z - z * (1 - big_phi_z))

    # E[max(Q - D, 0)] 是期望过剩量：订货量超过需求的期望值
    expected_overage = (order_qty - demand_mean) * big_phi_z + demand_std * phi_z

    # 总期望成本 = 缺货成本 * 期望缺货量 + 过剩成本 * 期望过剩量
    return cu * expected_underage + co * expected_overage


# 输入：两个分位数预测值 → 输出：估算的标准差
# 比直接用样本标准差更稳健，因为分位数对离群值不敏感
def robust_std_from_quantiles(
    q_low: float,
    q_high: float,
    level_low: float = 0.1,
    level_high: float = 0.9,
) -> float:
    """Estimate standard deviation from two quantile predictions.

    Uses the normal approximation:
        σ ≈ (Q_high - Q_low) / (z_high - z_low)

    This is more robust than using sample std, which is sensitive to outliers.

    Args:
        q_low: Predicted value at the lower quantile.
        q_high: Predicted value at the upper quantile.
        level_low: Lower quantile level (default 0.1).
        level_high: Upper quantile level (default 0.9).

    Returns:
        Estimated standard deviation. Returns 0 if q_high <= q_low.
    """
    # 如果高分位数 <= 低分位数，说明预测结果异常（分位数交叉），返回 0
    if q_high <= q_low:
        return 0.0

    # z 值之差：例如 z(0.9) - z(0.1) ≈ 1.28 - (-1.28) = 2.56
    z_spread = stats.norm.ppf(level_high) - stats.norm.ppf(level_low)
    if z_spread <= 0:
        return 0.0

    # 利用正态分布性质：分位数间距 / z 值间距 = 标准差
    return (q_high - q_low) / z_spread
