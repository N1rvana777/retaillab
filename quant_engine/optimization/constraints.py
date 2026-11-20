"""Operational constraints applied after the Newsvendor optimization.

These transform the mathematically optimal quantity into a practically
feasible order that respects real-world business rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# 单个 SKU 的业务约束，所有字段可选，只有设置了的约束才会生效
@dataclass(frozen=True)
class OrderConstraints:
    """Business constraints for a single SKU order.

    All fields are optional — only active constraints are applied.
    """

    moq: int = 1
    """Minimum Order Quantity. Orders must be multiples of this."""

    shelf_capacity: int | None = None
    """Maximum units that fit on the shelf. None = unlimited."""

    current_inventory: int = 0
    """Current stock on hand. Order qty = recommended - on_hand (if positive)."""

    truck_remaining_capacity: float | None = None
    """Remaining truck capacity in units. None = unlimited."""

    max_order_qty: int | None = None
    """Hard cap on order quantity from supplier. None = unlimited."""


# 输入：原始订货量 + MOQ → 输出：向上取整到 MOQ 整数倍的订货量
# 向上取整是因为供应商不接受低于 MOQ 的订单，且报童模型已考虑过量成本
def apply_moq_constraint(qty: float, moq: int = 1) -> float:
    """Round up to the nearest Minimum Order Quantity.

    Suppliers often require orders in case packs (e.g., MOQ=24 for
    a case of bottled water). We round UP because ordering less than
    MOQ is not allowed, and the Newsvendor already accounts for
    overage cost.

    Args:
        qty: Raw order quantity.
        moq: Minimum order quantity (must be >= 1).

    Returns:
        Quantity rounded up to nearest MOQ multiple.
    """
    if moq <= 1:
        return qty
    return math.ceil(qty / moq) * moq


# 输入：订货量 + 货架容量 → 输出：不超过货架容量的订货量
# 订 500 个但货架只放得下 100 个是浪费
def apply_shelf_capacity(qty: float, shelf_capacity: int | None) -> float:
    """Cap order at available shelf space.

    No point ordering 500 units if the shelf only holds 100.

    Args:
        qty: Order quantity.
        shelf_capacity: Maximum shelf capacity. None = no constraint.

    Returns:
        Capped quantity.
    """
    if shelf_capacity is None:
        return qty
    return min(qty, float(shelf_capacity))


# 输入：订货量 + 卡车剩余容量 → 输出：不超过卡车容量的订货量
def apply_truck_fill(qty: float, truck_remaining: float | None) -> float:
    """Cap order at remaining truck capacity.

    Args:
        qty: Order quantity.
        truck_remaining: Remaining truck space in units. None = no constraint.

    Returns:
        Capped quantity.
    """
    if truck_remaining is None:
        return qty
    return min(qty, truck_remaining)


# 输入：建议补货量 + 当前库存 → 输出：实际需要订购的净量
# 已有库存要扣掉，避免重复订货
def apply_inventory_netting(qty: float, current_inventory: int) -> float:
    """Subtract current on-hand inventory from the order.

    If we recommend ordering 100 but already have 30 in stock,
    only order 70. Never order negative (return logic is separate).

    Args:
        qty: Recommended total inventory level.
        current_inventory: Current stock on hand.

    Returns:
        Net order quantity, floored at 0.
    """
    return max(0.0, qty - current_inventory)


# 输入：原始最优量 + 全部约束 → 输出：(约束后的量, 生效的约束列表)
# 约束应用顺序很重要：先扣库存、再限容量、最后取整 MOQ
def apply_all_constraints(
    raw_qty: float,
    constraints: OrderConstraints,
) -> tuple[float, list[str]]:
    """Apply all operational constraints in priority order.

    Returns both the constrained quantity and a list of which constraints
    were binding (for telemetry logging).

    Args:
        raw_qty: Unconstrained optimal order quantity.
        constraints: Business constraints to apply.

    Returns:
        Tuple of (constrained_qty, list of constraint names that were binding).
    """
    qty = raw_qty
    applied: list[str] = []

    # 1. 扣减当前库存（先做，因为后续所有约束都应该基于"还需要补多少"来判断）
    if constraints.current_inventory > 0:
        netted = apply_inventory_netting(qty, constraints.current_inventory)
        if netted < qty:
            applied.append(f"inventory_netting:{constraints.current_inventory}")
            qty = netted

    # 2. 货架容量限制（超出货架容量的订货没有意义）
    if constraints.shelf_capacity is not None:
        capped = apply_shelf_capacity(qty, constraints.shelf_capacity)
        if capped < qty:
            applied.append(f"shelf_capacity:{constraints.shelf_capacity}")
            qty = capped

    # 3. 卡车容量限制（物流约束）
    if constraints.truck_remaining_capacity is not None:
        capped = apply_truck_fill(qty, constraints.truck_remaining_capacity)
        if capped < qty:
            applied.append(f"truck_fill:{constraints.truck_remaining_capacity}")
            qty = capped

    # 4. 供应商硬性最大订货量
    if constraints.max_order_qty is not None:
        if qty > constraints.max_order_qty:
            applied.append(f"max_order:{constraints.max_order_qty}")
            qty = float(constraints.max_order_qty)

    # 5. MOQ 最后应用，因为它是向上取整，必须在所有"向下压"的约束之后
    # 否则先取整再压上限会浪费取整带来的额外数量
    if constraints.moq > 1 and qty > 0:
        before_moq = qty
        qty = apply_moq_constraint(qty, constraints.moq)
        if qty != before_moq:
            applied.append(f"moq:{constraints.moq}")

    return qty, applied
