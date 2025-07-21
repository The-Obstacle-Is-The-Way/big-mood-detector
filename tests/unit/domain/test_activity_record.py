"""
Tests for Activity Record Domain Entity

Following TDD for activity data validation and business rules.
"""

from datetime import UTC, datetime, timedelta

import pytest

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)


class TestActivityType:
    """Test suite for ActivityType enum."""

    def test_activity_type_from_healthkit_identifier(self):
        """Test conversion from HealthKit identifiers."""
        # ARRANGE & ACT & ASSERT
        assert (
            ActivityType.from_healthkit_identifier("HKQuantityTypeIdentifierStepCount")
            == ActivityType.STEP_COUNT
        )
        assert (
            ActivityType.from_healthkit_identifier(
                "HKQuantityTypeIdentifierActiveEnergyBurned"
            )
            == ActivityType.ACTIVE_ENERGY
        )

    def test_invalid_healthkit_identifier_raises_error(self):
        """Test that invalid identifiers raise ValueError."""
        with pytest.raises(ValueError, match="Unknown activity type"):
            ActivityType.from_healthkit_identifier("InvalidIdentifier")


class TestActivityRecord:
    """Test suite for ActivityRecord entity."""

    def test_create_valid_activity_record(self):
        """Test creating a valid activity record."""
        # ARRANGE
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)

        # ACT
        record = ActivityRecord(
            source_name="Apple Watch",
            start_date=start,
            end_date=end,
            activity_type=ActivityType.STEP_COUNT,
            value=10000.0,
            unit="count",
        )

        # ASSERT
        assert record.source_name == "Apple Watch"
        assert record.start_date == start
        assert record.end_date == end
        assert record.activity_type == ActivityType.STEP_COUNT
        assert record.value == 10000.0
        assert record.unit == "count"

    def test_activity_record_is_immutable(self):
        """Test that activity record cannot be modified after creation."""
        # ARRANGE
        record = ActivityRecord(
            source_name="iPhone",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=5000.0,
            unit="count",
        )

        # ACT & ASSERT
        with pytest.raises(AttributeError):
            record.value = 6000.0

    def test_invalid_date_range_raises_error(self):
        """Test that end date must be after or equal to start date."""
        # ARRANGE
        start = datetime(2024, 1, 2, tzinfo=UTC)
        end = datetime(2024, 1, 1, tzinfo=UTC)

        # ACT & ASSERT
        with pytest.raises(ValueError, match="End date must be after or equal"):
            ActivityRecord(
                source_name="Apple Watch",
                start_date=start,
                end_date=end,
                activity_type=ActivityType.STEP_COUNT,
                value=1000.0,
                unit="count",
            )

    def test_empty_source_name_raises_error(self):
        """Test that empty source name is not allowed."""
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Source name is required"):
            ActivityRecord(
                source_name="",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 1, tzinfo=UTC),
                activity_type=ActivityType.STEP_COUNT,
                value=1000.0,
                unit="count",
            )

    def test_negative_value_raises_error(self):
        """Test that negative activity values are not allowed."""
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Activity value cannot be negative"):
            ActivityRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 1, tzinfo=UTC),
                activity_type=ActivityType.STEP_COUNT,
                value=-100.0,
                unit="count",
            )

    def test_empty_unit_raises_error(self):
        """Test that empty unit is not allowed."""
        # ACT & ASSERT
        with pytest.raises(ValueError, match="Unit is required"):
            ActivityRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 1, tzinfo=UTC),
                activity_type=ActivityType.STEP_COUNT,
                value=1000.0,
                unit="",
            )

    def test_duration_hours_calculation(self):
        """Test duration calculation in hours."""
        # ARRANGE
        start = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 17, 0, tzinfo=UTC)
        record = ActivityRecord(
            source_name="Apple Watch",
            start_date=start,
            end_date=end,
            activity_type=ActivityType.STEP_COUNT,
            value=8000.0,
            unit="count",
        )

        # ACT & ASSERT
        assert record.duration_hours == 8.0

    def test_is_instantaneous_property(self):
        """Test detection of instantaneous measurements."""
        # ARRANGE
        same_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        instant_record = ActivityRecord(
            source_name="Apple Watch",
            start_date=same_time,
            end_date=same_time,
            activity_type=ActivityType.STEP_COUNT,
            value=100.0,
            unit="count",
        )

        duration_record = ActivityRecord(
            source_name="Apple Watch",
            start_date=same_time,
            end_date=same_time + timedelta(hours=1),
            activity_type=ActivityType.STEP_COUNT,
            value=100.0,
            unit="count",
        )

        # ASSERT
        assert instant_record.is_instantaneous
        assert not duration_record.is_instantaneous

    def test_intensity_per_hour_calculation(self):
        """Test intensity calculation for rate-based metrics."""
        # ARRANGE
        record = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 4, 0, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=4000.0,
            unit="count",
        )

        # ACT & ASSERT
        assert record.intensity_per_hour == 1000.0  # 4000 steps / 4 hours

    def test_is_high_activity_step_count(self):
        """Test high activity detection for step count."""
        # ARRANGE - High activity (>15000 steps/day rate)
        high_activity = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=8000.0,  # 16000/day rate
            unit="count",
        )

        normal_activity = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=5000.0,  # 10000/day rate
            unit="count",
        )

        # ASSERT
        assert high_activity.is_high_activity
        assert not normal_activity.is_high_activity

    def test_is_low_activity_step_count(self):
        """Test low activity detection for step count."""
        # ARRANGE - Low activity (<2000 steps/day rate)
        low_activity = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=900.0,  # 1800/day rate
            unit="count",
        )

        # ASSERT
        assert low_activity.is_low_activity

    def test_is_high_activity_energy(self):
        """Test high activity detection for energy burn."""
        # ARRANGE - High energy burn (>500 cal/day)
        high_energy = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 6, 0, tzinfo=UTC),
            activity_type=ActivityType.ACTIVE_ENERGY,
            value=150.0,  # 600/day rate
            unit="Cal",
        )

        # ASSERT
        assert high_energy.is_high_activity

    def test_can_aggregate_with_same_type(self):
        """Test aggregation compatibility check."""
        # ARRANGE
        record1 = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=1000.0,
            unit="count",
        )

        record2 = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, 2, 0, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=1000.0,
            unit="count",
        )

        # ASSERT
        assert record1.can_aggregate_with(record2)

    def test_cannot_aggregate_different_types(self):
        """Test that different activity types cannot aggregate."""
        # ARRANGE
        steps = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=1000.0,
            unit="count",
        )

        energy = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, tzinfo=UTC),
            activity_type=ActivityType.ACTIVE_ENERGY,
            value=100.0,
            unit="Cal",
        )

        # ASSERT
        assert not steps.can_aggregate_with(energy)

    def test_cannot_aggregate_different_sources(self):
        """Test that different sources cannot aggregate."""
        # ARRANGE
        watch_record = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=1000.0,
            unit="count",
        )

        phone_record = ActivityRecord(
            source_name="iPhone",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 1, tzinfo=UTC),
            activity_type=ActivityType.STEP_COUNT,
            value=1000.0,
            unit="count",
        )

        # ASSERT
        assert not watch_record.can_aggregate_with(phone_record)
