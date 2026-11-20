"""Quant Engine Service Facade — the ONLY entry point for external consumers.

This is the gateway through which api_gateway accesses quant_engine.
It wraps all engine calls in try-except blocks, ensuring that failures
in the ML pipeline degrade gracefully to safety stock fallback.

Architecture rule: api_gateway imports ONLY this module from quant_engine.
It never touches optimization, forecasting, or domain internals directly.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

from quant_engine.domain.entities import ReplenishmentOrder, SKU
from quant_engine.domain.enums import OptimizationMethod
from quant_engine.domain.value_objects import CostParameters
from quant_engine.fallback.safety_stock import SafetyStockFallback
from quant_engine.optimization.constraints import OrderConstraints
from quant_engine.optimization.inventory_optimizer import InventoryOptimizer
from quant_engine.telemetry.decision_logger import DecisionLogger
from quant_engine.telemetry.schemas import DecisionSnapshot

logger = logging.getLogger(__name__)


# 量化引擎的唯一对外门面类，所有外部调用必须经过此类
# 核心设计原则：永远不向调用方抛异常，失败时自动降级到安全库存
class QuantEngineService:
    """Facade for the quant engine — handles errors, logging, and fallback.

    This is designed to NEVER raise exceptions to the caller.
    Every public method either returns a valid result or a fallback result.
    """

    def __init__(
        self,
        optimizer: InventoryOptimizer | None = None,
        fallback: SafetyStockFallback | None = None,
        decision_logger: DecisionLogger | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            optimizer: The primary ML-based optimizer. None = fallback only.
            fallback: Safety stock fallback engine.
            decision_logger: Telemetry logger for decision audit trail.
        """
        self._optimizer = optimizer
        # 如果没有传入降级引擎，使用默认配置创建一个，保证降级路径始终可用
        self._fallback = fallback or SafetyStockFallback()
        # 决策日志必须存在，每笔补货决策都要留痕，便于事后审计和调试
        self._decision_logger = decision_logger or DecisionLogger()

    # 输入：SKU信息 + 特征向量 + 成本参数 → 输出：补货订单（保证永远返回有效结果）
    def compute_replenishment(
        self,
        sku: SKU,
        features: np.ndarray,
        cost_params: CostParameters,
        delivery_date: date,
        history_days: int,
        recent_sales: list[float] | None = None,
        constraints: OrderConstraints | None = None,
    ) -> ReplenishmentOrder:
        """Compute optimal order quantity with full error protection.

        This method NEVER raises exceptions. If the primary engine fails,
        it falls back to safety stock. If even that fails, it returns a
        zero-order with an error message.

        Args:
            sku: The SKU to optimize.
            features: Feature vector for prediction.
            cost_params: Asymmetric cost structure.
            delivery_date: When the order will arrive.
            history_days: Days of sales history available.
            recent_sales: Last N days of raw sales (for fallback).
            constraints: Optional operational constraints.

        Returns:
            ReplenishmentOrder — always valid, never None.
        """
        start_time = time.perf_counter()
        error_msg: str | None = None

        # === 第一层：ML 主引擎路径 ===
        # 尝试使用训练好的 XGBoost/LightGBM 分位数回归模型计算最优补货量
        try:
            if self._optimizer is not None:
                order = self._optimizer.optimize(
                    sku=sku,
                    features=features,
                    cost_params=cost_params,
                    delivery_date=delivery_date,
                    history_days=history_days,
                    constraints=constraints,
                )
                self._log_decision(order, features, cost_params, start_time)
                return order
            else:
                # 优化器未初始化，直接进入降级路径（可能是系统启动阶段模型还没加载）
                error_msg = "Optimizer not initialized"
        except Exception as e:
            # 主引擎可能因为特征异常、模型文件损坏等原因失败，此处捕获所有异常
            error_msg = f"Optimizer failed: {type(e).__name__}: {e}"
            logger.error(
                "Primary engine failed for SKU %s (store %s): %s",
                sku.sku_id, sku.store_id, error_msg,
            )

        # === 第二层：安全库存降级路径 ===
        # 主引擎失败时，用近期销量的移动平均 * 安全系数作为补货量
        # 虽然不够精确，但能保证不断货
        try:
            order = self._fallback.compute_order(
                sku_id=sku.sku_id,
                store_id=sku.store_id,
                recent_sales=recent_sales,
                category=sku.category.value,
                delivery_date=delivery_date.isoformat(),
                error_message=error_msg or "unknown",
            )
            self._log_decision(
                order, features, cost_params, start_time, error_msg=error_msg
            )
            return order
        except Exception as e2:
            # 极端情况：连安全库存降级都失败了（比如 recent_sales 数据格式异常）
            logger.critical(
                "BOTH primary and fallback engines failed for SKU %s: %s",
                sku.sku_id, e2,
            )
            # === 第三层：终极兜底 —— 返回零订单而不是崩溃 ===
            # 宁可少订也不能让系统报 500 错误，运营人员可以手动处理
            return ReplenishmentOrder(
                sku_id=sku.sku_id,
                store_id=sku.store_id,
                recommended_qty=0,
                delivery_date=delivery_date,
                critical_ratio=0.5,
                forecast_mean=0.0,
                forecast_std=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                expected_cost=0.0,
                method=OptimizationMethod.FALLBACK_SAFETY_STOCK,
                fallback_used=True,
                reasoning=f"Both engines failed: {error_msg} / {e2}",
            )

    # 输入：补货订单 + 特征 + 成本参数 + 计时起点 → 输出：写入遥测日志（无返回值）
    def _log_decision(
        self,
        order: ReplenishmentOrder,
        features: np.ndarray,
        cost_params: CostParameters,
        start_time: float,
        error_msg: str | None = None,
    ) -> None:
        """Write decision telemetry."""
        # 计算引擎耗时，用于监控性能退化
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 构造决策快照，包含完整的输入→输出审计链路
        snapshot = DecisionSnapshot(
            timestamp=datetime.now(timezone.utc),
            sku_id=order.sku_id,
            store_id=order.store_id,
            delivery_date=order.delivery_date.isoformat(),
            # 只记录前 20 个特征，避免日志过大；flatten 处理多维特征
            input_features={f"f{i}": float(v) for i, v in enumerate(features.flatten()[:20])},
            predicted_quantiles={
                "0.1": order.confidence_lower,
                "0.5": order.forecast_mean,
                "0.9": order.confidence_upper,
                "cr": float(order.forecast_mean),  # 近似值，实际 cr 分位数预测未单独存储
            },
            critical_ratio=order.critical_ratio,
            raw_qty=float(order.recommended_qty),
            constraints_applied=order.constraints_applied,
            final_qty=order.recommended_qty,
            method=order.method.value,
            fallback_used=order.fallback_used,
            expected_cost=order.expected_cost,
            engine_latency_ms=latency_ms,
            error_message=error_msg,
        )

        # 遥测日志写入失败不能影响主流程，静默处理
        try:
            self._decision_logger.log(snapshot)
        except Exception:
            logger.warning("Failed to write decision telemetry", exc_info=True)
