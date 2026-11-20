"""Concept drift monitoring — detects when model performance degrades.

Supermarket demand patterns shift constantly: new competitors open,
seasons change, pandemics hit, consumer preferences evolve. A model
trained 3 months ago may be making terrible predictions today.

This module continuously monitors the Pinball Loss of recent predictions
against a historical baseline. When the recent loss deviates significantly
(>2σ warning, >3σ critical), it triggers alerts for model retraining.

Usage:
    detector = DriftDetector(baseline_window_days=90, recent_window_days=7)
    detector.update(date, pinball_loss_value)
    alert = detector.check()
    if alert.severity == DriftSeverity.CRITICAL:
        trigger_retraining()
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import date

import numpy as np

from quant_engine.domain.enums import DriftSeverity

logger = logging.getLogger(__name__)


# 漂移告警结果：包含严重程度、z 分数和人类可读的消息
@dataclass
class DriftAlert:
    """Result of a drift check."""

    severity: DriftSeverity
    baseline_mean: float
    baseline_std: float
    recent_mean: float
    z_score: float
    message: str
    check_date: date
    category: str | None = None
    """Optional: per-category drift monitoring."""


# 概念漂移检测器：监控模型预测质量随时间的变化
# 原理：用长期历史（基线窗口）的损失统计量和近期（对比窗口）做 z 检验
# 超过 2σ 告警，超过 3σ 要求重新训练
class DriftDetector:
    """Monitors prediction quality over time using Pinball Loss.

    Maintains a rolling window of historical losses (baseline) and
    compares the most recent window against it. Uses a simple z-score
    test: if recent_mean > baseline_mean + k * baseline_std, alert.

    The baseline window should be long enough to capture normal variation
    (e.g., 90 days covers weekday/weekend cycles, monthly patterns).
    The recent window should be short enough to detect problems quickly
    (e.g., 7 days).
    """

    def __init__(
        self,
        baseline_window_days: int = 90,
        recent_window_days: int = 7,
        warning_threshold_sigma: float = 2.0,
        critical_threshold_sigma: float = 3.0,
        category: str | None = None,
    ) -> None:
        """Initialize the drift detector.

        Args:
            baseline_window_days: Days of history for baseline statistics.
            recent_window_days: Days for the recent comparison window.
            warning_threshold_sigma: σ multiplier for WARNING alerts.
            critical_threshold_sigma: σ multiplier for CRITICAL alerts.
            category: Optional category label for per-category monitoring.
        """
        # 基线窗口 90 天：足够覆盖周末/月末/季节等周期性波动
        self._baseline_window = baseline_window_days
        # 对比窗口 7 天：足够快速发现问题，又不会被单日异常误触发
        self._recent_window = recent_window_days
        self._warning_sigma = warning_threshold_sigma
        self._critical_sigma = critical_threshold_sigma
        self._category = category

        # 滚动存储：使用 deque 限制最大长度，自动丢弃过旧的数据
        self._history: deque[tuple[date, float]] = deque(
            maxlen=baseline_window_days + recent_window_days
        )

    # 输入：日期 + 当日弹球损失 → 副作用：追加到历史记录
    def update(self, check_date: date, pinball_loss: float) -> None:
        """Record a daily Pinball Loss measurement.

        Args:
            check_date: The date of the measurement.
            pinball_loss: Mean Pinball Loss for this day.
        """
        self._history.append((check_date, pinball_loss))

    # 输入：日期-损失列表 → 副作用：批量追加
    def update_batch(self, records: list[tuple[date, float]]) -> None:
        """Record multiple daily measurements at once.

        Args:
            records: List of (date, pinball_loss) tuples.
        """
        for check_date, loss in records:
            self.update(check_date, loss)

    # 输入：无 → 输出：漂移告警（包含严重程度、z 分数、消息）
    def check(self) -> DriftAlert:
        """Run the drift check against current history.

        Returns:
            DriftAlert with severity, z-score, and human-readable message.
        """
        losses = [loss for _, loss in self._history]
        n_total = len(losses)

        # 数据不足时无法做有意义的比较，返回正常状态
        if n_total < self._recent_window + 10:
            return DriftAlert(
                severity=DriftSeverity.NORMAL,
                baseline_mean=0.0,
                baseline_std=0.0,
                recent_mean=0.0,
                z_score=0.0,
                message=f"Insufficient data ({n_total} days). Need at least {self._recent_window + 10}.",
                check_date=date.today(),
                category=self._category,
            )

        # 将历史数据切分为基线窗口和近期窗口
        # 近期窗口：最后 N 天
        recent_losses = np.array(losses[-self._recent_window:])
        # 基线窗口：近期之前的所有数据
        baseline_losses = np.array(losses[:-self._recent_window])

        # 基线数据太少（<10天）时也无法可靠估算标准差
        if len(baseline_losses) < 10:
            return DriftAlert(
                severity=DriftSeverity.NORMAL,
                baseline_mean=0.0,
                baseline_std=0.0,
                recent_mean=float(np.mean(recent_losses)),
                z_score=0.0,
                message="Insufficient baseline data.",
                check_date=date.today(),
                category=self._category,
            )

        baseline_mean = float(np.mean(baseline_losses))
        # ddof=1 使用贝塞尔修正的样本标准差
        baseline_std = float(np.std(baseline_losses, ddof=1))
        recent_mean = float(np.mean(recent_losses))

        # z 分数 = (近期均值 - 基线均值) / 基线标准差
        # 衡量近期表现偏离正常水平多少个标准差
        if baseline_std < 1e-10:
            # 标准差几乎为零说明历史损失非常稳定，无法做有意义的比较
            z_score = 0.0
        else:
            z_score = (recent_mean - baseline_mean) / baseline_std

        # 根据 z 分数判断严重程度
        if z_score >= self._critical_sigma:
            # >=3σ：极不可能是正常波动（概率 < 0.3%），必须重新训练
            severity = DriftSeverity.CRITICAL
            message = (
                f"CRITICAL: Pinball Loss spiked to {recent_mean:.4f} "
                f"(baseline {baseline_mean:.4f} +/- {baseline_std:.4f}, "
                f"z={z_score:.2f}). MODEL RETRAINING REQUIRED."
            )
            logger.error(message)
        elif z_score >= self._warning_sigma:
            # >=2σ：可能是概念漂移的早期信号（概率 < 5%），密切关注
            severity = DriftSeverity.WARNING
            message = (
                f"WARNING: Pinball Loss elevated at {recent_mean:.4f} "
                f"(baseline {baseline_mean:.4f} +/- {baseline_std:.4f}, "
                f"z={z_score:.2f}). Monitor closely."
            )
            logger.warning(message)
        else:
            # z < 2σ：在正常波动范围内
            severity = DriftSeverity.NORMAL
            message = (
                f"Normal: Pinball Loss {recent_mean:.4f} "
                f"(baseline {baseline_mean:.4f} +/- {baseline_std:.4f}, "
                f"z={z_score:.2f})."
            )

        return DriftAlert(
            severity=severity,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            recent_mean=recent_mean,
            z_score=z_score,
            message=message,
            check_date=date.today(),
            category=self._category,
        )

    @property
    def has_sufficient_data(self) -> bool:
        """Whether enough history has accumulated for meaningful drift detection."""
        return len(self._history) >= self._recent_window + 10
