"""Tests for the concept drift detector."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_engine.domain.enums import DriftSeverity
from quant_engine.monitoring.drift_detector import DriftDetector


@pytest.fixture
def detector() -> DriftDetector:
    return DriftDetector(
        baseline_window_days=30,
        recent_window_days=7,
        warning_threshold_sigma=2.0,
        critical_threshold_sigma=3.0,
    )


class TestDriftDetection:
    """Test drift severity classification."""

    def test_normal_conditions(self, detector):
        """Stable loss should report NORMAL."""
        base_date = date(2025, 1, 1)
        # 40 days of stable loss around 1.0
        for i in range(40):
            detector.update(base_date + timedelta(days=i), 1.0 + 0.05 * (i % 3))

        alert = detector.check()
        assert alert.severity == DriftSeverity.NORMAL

    def test_warning_on_elevated_loss(self, detector):
        """Loss elevated >2σ should trigger WARNING."""
        base_date = date(2025, 1, 1)
        # 30 days baseline: loss ≈ 1.0 with some natural variance
        import random
        random.seed(42)
        for i in range(30):
            detector.update(base_date + timedelta(days=i), 1.0 + random.uniform(-0.1, 0.1))

        # 7 days elevated: loss ≈ 3.0 (well above 2σ)
        for i in range(7):
            detector.update(base_date + timedelta(days=30 + i), 3.0)

        alert = detector.check()
        assert alert.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)
        assert alert.z_score > 2.0

    def test_critical_on_spike(self, detector):
        """Loss spiked >3σ should trigger CRITICAL."""
        base_date = date(2025, 1, 1)
        # 30 days baseline: stable loss with small variance
        import random
        random.seed(123)
        for i in range(30):
            detector.update(base_date + timedelta(days=i), 1.0 + random.uniform(-0.1, 0.1))

        # 7 days massive spike
        for i in range(7):
            detector.update(base_date + timedelta(days=30 + i), 10.0)

        alert = detector.check()
        assert alert.severity == DriftSeverity.CRITICAL
        assert "RETRAINING" in alert.message

    def test_insufficient_data(self, detector):
        """With too little data, should report NORMAL (not enough evidence)."""
        detector.update(date(2025, 1, 1), 1.0)
        alert = detector.check()
        assert alert.severity == DriftSeverity.NORMAL
        assert "Insufficient" in alert.message

    def test_has_sufficient_data(self, detector):
        base_date = date(2025, 1, 1)
        assert not detector.has_sufficient_data
        for i in range(20):
            detector.update(base_date + timedelta(days=i), 1.0)
        assert detector.has_sufficient_data
