"""
Tests for Time Period Value Object

Following TDD for time-based calculations.
"""

from datetime import UTC, datetime, timedelta

import pytest

from big_mood_detector.domain.value_objects.time_period import TimePeriod


class TestTimePeriod:
    """Test suite for TimePeriod value object."""

    def test_create_valid_time_period(self):
        """Test creating a valid time period."""
        # ARRANGE
        start = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 17, 0, tzinfo=UTC)

        # ACT
        period = TimePeriod(start=start, end=end)

        # ASSERT
        assert period.start == start
        assert period.end == end

    def test_time_period_is_immutable(self):
        """Test that time period cannot be modified after creation."""
        # ARRANGE
        period = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 17, 0, tzinfo=UTC),
        )

        # ACT & ASSERT
        with pytest.raises(AttributeError):
            period.start = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)

    def test_invalid_time_period_raises_error(self):
        """Test that end must be after start."""
        # ARRANGE
        start = datetime(2024, 1, 1, 17, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Invalid time period"):
            TimePeriod(start=start, end=end)

    def test_equal_start_end_raises_error(self):
        """Test that start and end cannot be equal."""
        # ARRANGE
        same_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

        # ACT & ASSERT
        with pytest.raises(ValueError, match="Invalid time period"):
            TimePeriod(start=same_time, end=same_time)

    def test_duration_property(self):
        """Test duration calculation as timedelta."""
        # ARRANGE
        period = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 17, 30, tzinfo=UTC),
        )

        # ACT & ASSERT
        assert period.duration == timedelta(hours=8, minutes=30)

    def test_duration_hours_property(self):
        """Test duration calculation in hours."""
        # ARRANGE
        period = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 17, 30, tzinfo=UTC),
        )

        # ACT & ASSERT
        assert period.duration_hours == 8.5

    def test_duration_minutes_property(self):
        """Test duration calculation in minutes."""
        # ARRANGE
        period = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 9, 45, tzinfo=UTC),
        )

        # ACT & ASSERT
        assert period.duration_minutes == 45

    def test_overlaps_with_overlapping_periods(self):
        """Test detection of overlapping periods."""
        # ARRANGE
        period1 = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        period2 = TimePeriod(
            start=datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
        )

        # ACT & ASSERT
        assert period1.overlaps_with(period2)
        assert period2.overlaps_with(period1)

    def test_overlaps_with_non_overlapping_periods(self):
        """Test non-overlapping periods."""
        # ARRANGE
        period1 = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        period2 = TimePeriod(
            start=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 17, 0, tzinfo=UTC),
        )

        # ACT & ASSERT
        assert not period1.overlaps_with(period2)
        assert not period2.overlaps_with(period1)

    def test_overlaps_with_adjacent_periods(self):
        """Test adjacent periods don't overlap."""
        # ARRANGE
        period1 = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        period2 = TimePeriod(
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 17, 0, tzinfo=UTC),
        )

        # ACT & ASSERT
        assert not period1.overlaps_with(period2)

    def test_contains_timestamp(self):
        """Test checking if timestamp is within period."""
        # ARRANGE
        period = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 17, 0, tzinfo=UTC),
        )

        # ACT & ASSERT
        assert period.contains(datetime(2024, 1, 1, 12, 0, tzinfo=UTC))
        assert period.contains(
            datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        )  # Start inclusive
        assert period.contains(datetime(2024, 1, 1, 17, 0, tzinfo=UTC))  # End inclusive
        assert not period.contains(datetime(2024, 1, 1, 8, 59, tzinfo=UTC))
        assert not period.contains(datetime(2024, 1, 1, 17, 1, tzinfo=UTC))

    def test_merge_with_overlapping_periods(self):
        """Test merging overlapping periods."""
        # ARRANGE
        period1 = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        period2 = TimePeriod(
            start=datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
        )

        # ACT
        merged = period1.merge_with(period2)

        # ASSERT
        assert merged is not None
        assert merged.start == datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        assert merged.end == datetime(2024, 1, 1, 14, 0, tzinfo=UTC)

    def test_merge_with_adjacent_periods(self):
        """Test merging adjacent periods (within 1 minute)."""
        # ARRANGE
        period1 = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        period2 = TimePeriod(
            start=datetime(2024, 1, 1, 12, 0, 30, tzinfo=UTC),  # 30 seconds gap
            end=datetime(2024, 1, 1, 17, 0, tzinfo=UTC),
        )

        # ACT
        merged = period1.merge_with(period2)

        # ASSERT
        assert merged is not None
        assert merged.start == datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
        assert merged.end == datetime(2024, 1, 1, 17, 0, tzinfo=UTC)

    def test_merge_with_distant_periods(self):
        """Test merging fails for distant periods."""
        # ARRANGE
        period1 = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        period2 = TimePeriod(
            start=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),  # 1 hour gap
            end=datetime(2024, 1, 1, 17, 0, tzinfo=UTC),
        )

        # ACT
        merged = period1.merge_with(period2)

        # ASSERT
        assert merged is None

    def test_string_representation(self):
        """Test human-readable string representation."""
        # ARRANGE
        period = TimePeriod(
            start=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2024, 1, 1, 17, 30, tzinfo=UTC),
        )

        # ACT
        result = str(period)

        # ASSERT
        assert "9:00:00+00:00" in result
        assert "17:30:00+00:00" in result
        assert "8.5h" in result
