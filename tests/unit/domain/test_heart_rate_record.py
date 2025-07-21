"""
Tests for Heart Rate Record Domain Entity

Following TDD for heart rate data validation and clinical rules.
"""

from datetime import UTC, datetime

import pytest

class TestHeartMetricType:
    """Test suite for HeartMetricType enum."""

    def test_heart_metric_type_from_healthkit_identifier(self):
        """Test conversion from HealthKit identifiers."""
        from big_mood_detector.domain.entities.heart_rate_record import HeartMetricType

        # ARRANGE & ACT & ASSERT
        assert (
            HeartMetricType.from_healthkit_identifier(
                "HKQuantityTypeIdentifierHeartRate"
            )
            == HeartMetricType.HEART_RATE
        )
        assert (
            HeartMetricType.from_healthkit_identifier(
                "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
            )
            == HeartMetricType.HRV_SDNN
        )
        assert (
            HeartMetricType.from_healthkit_identifier(
                "HKQuantityTypeIdentifierRestingHeartRate"
            )
            == HeartMetricType.RESTING_HEART_RATE
        )

    def test_invalid_healthkit_identifier_raises_error(self):
        """Test that invalid identifiers raise ValueError."""
        from big_mood_detector.domain.entities.heart_rate_record import HeartMetricType

        with pytest.raises(ValueError, match="Unknown heart metric type"):
            HeartMetricType.from_healthkit_identifier("InvalidIdentifier")

    def test_is_hrv_metric(self):
        """Test HRV metric detection."""
        from big_mood_detector.domain.entities.heart_rate_record import HeartMetricType

        assert HeartMetricType.HRV_SDNN.is_hrv_metric()
        assert not HeartMetricType.HEART_RATE.is_hrv_metric()
        assert not HeartMetricType.RESTING_HEART_RATE.is_hrv_metric()

class TestMotionContext:
    """Test suite for MotionContext enum."""

    def test_motion_context_values(self):
        """Test motion context enum values."""
        from big_mood_detector.domain.entities.heart_rate_record import MotionContext

        assert MotionContext.SEDENTARY.value == "sedentary"
        assert MotionContext.ACTIVE.value == "active"
        assert MotionContext.UNKNOWN.value == "unknown"

    def test_from_string(self):
        """Test creating motion context from string."""
        from big_mood_detector.domain.entities.heart_rate_record import MotionContext

        assert MotionContext.from_string("sedentary") == MotionContext.SEDENTARY
        assert MotionContext.from_string("active") == MotionContext.ACTIVE
        assert MotionContext.from_string("") == MotionContext.UNKNOWN
        assert MotionContext.from_string(None) == MotionContext.UNKNOWN
        assert MotionContext.from_string("invalid") == MotionContext.UNKNOWN

class TestHeartRateRecord:
    """Test suite for HeartRateRecord entity."""

    def test_create_valid_heart_rate_record(self):
        """Test creating a valid heart rate record."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
            MotionContext,
        )

        # ARRANGE
        timestamp = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)

        # ACT
        record = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=timestamp,
            metric_type=HeartMetricType.HEART_RATE,
            value=72.0,
            unit="count/min",
            motion_context=MotionContext.SEDENTARY,
        )

        # ASSERT
        assert record.source_name == "Apple Watch"
        assert record.timestamp == timestamp
        assert record.metric_type == HeartMetricType.HEART_RATE
        assert record.value == 72.0
        assert record.unit == "count/min"
        assert record.motion_context == MotionContext.SEDENTARY

    def test_heart_rate_record_is_immutable(self):
        """Test that heart rate record cannot be modified after creation."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ARRANGE
        record = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=72.0,
            unit="count/min",
        )

        # ACT & ASSERT
        with pytest.raises(AttributeError):
            record.value = 80.0

    def test_empty_source_name_raises_error(self):
        """Test that empty source name is not allowed."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Source name is required"):
            HeartRateRecord(
                source_name="",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                metric_type=HeartMetricType.HEART_RATE,
                value=72.0,
                unit="count/min",
            )

    def test_negative_heart_rate_raises_error(self):
        """Test that negative heart rate values are not allowed."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Heart metric value cannot be negative"):
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                metric_type=HeartMetricType.HEART_RATE,
                value=-10.0,
                unit="count/min",
            )

    def test_empty_unit_raises_error(self):
        """Test that empty unit is not allowed."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Unit is required"):
            HeartRateRecord(
                source_name="Apple Watch",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                metric_type=HeartMetricType.HEART_RATE,
                value=72.0,
                unit="",
            )

    def test_is_high_heart_rate(self):
        """Test detection of abnormally high heart rate."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
            MotionContext,
        )

        # ARRANGE - High heart rate at rest
        high_hr = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=110.0,  # >100 at rest
            unit="count/min",
            motion_context=MotionContext.SEDENTARY,
        )

        normal_hr = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=75.0,
            unit="count/min",
            motion_context=MotionContext.SEDENTARY,
        )

        # ASSERT
        assert high_hr.is_high_heart_rate
        assert not normal_hr.is_high_heart_rate

    def test_is_low_heart_rate(self):
        """Test detection of abnormally low heart rate."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ARRANGE - Low heart rate
        low_hr = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=45.0,  # <50
            unit="count/min",
        )

        normal_hr = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=65.0,
            unit="count/min",
        )

        # ASSERT
        assert low_hr.is_low_heart_rate
        assert not normal_hr.is_low_heart_rate

    def test_is_low_hrv(self):
        """Test detection of low HRV (autonomic dysfunction)."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ARRANGE - Low HRV
        low_hrv = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HRV_SDNN,
            value=15.0,  # <20ms indicates poor autonomic function
            unit="ms",
        )

        normal_hrv = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HRV_SDNN,
            value=45.0,
            unit="ms",
        )

        # ASSERT
        assert low_hrv.is_low_hrv
        assert not normal_hrv.is_low_hrv

    def test_is_clinically_significant(self):
        """Test detection of clinically significant values."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
            MotionContext,
        )

        # ARRANGE
        high_hr = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=110.0,
            unit="count/min",
            motion_context=MotionContext.SEDENTARY,
        )

        low_hrv = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HRV_SDNN,
            value=15.0,
            unit="ms",
        )

        normal = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=72.0,
            unit="count/min",
        )

        # ASSERT
        assert high_hr.is_clinically_significant
        assert low_hrv.is_clinically_significant
        assert not normal.is_clinically_significant

    def test_high_heart_rate_during_activity_not_significant(self):
        """Test that high HR during activity is not flagged as abnormal."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
            MotionContext,
        )

        # ARRANGE
        active_hr = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=150.0,  # High but active
            unit="count/min",
            motion_context=MotionContext.ACTIVE,
        )

        # ASSERT
        assert not active_hr.is_high_heart_rate  # Not high for active context
        assert not active_hr.is_clinically_significant

    def test_instantaneous_property(self):
        """Test that heart rate records are always instantaneous."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ARRANGE
        record = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=72.0,
            unit="count/min",
        )

        # ASSERT
        assert record.is_instantaneous

    def test_is_same_type(self):
        """Test checking if two records are same type."""
        from big_mood_detector.domain.entities.heart_rate_record import (
            HeartMetricType,
            HeartRateRecord,
        )

        # ARRANGE
        hr1 = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=72.0,
            unit="count/min",
        )

        hr2 = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HEART_RATE,
            value=75.0,
            unit="count/min",
        )

        hrv = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metric_type=HeartMetricType.HRV_SDNN,
            value=45.0,
            unit="ms",
        )

        # ASSERT
        assert hr1.is_same_type(hr2)
        assert not hr1.is_same_type(hrv)
