"""
Tests for Activity Aggregator Domain Service

Test-driven development for clinical activity aggregation.
Following Uncle Bob's Clean Architecture principles.
"""

from datetime import UTC, date, datetime, time

import pytest

class TestDailyActivitySummary:
    """Test suite for DailyActivitySummary value object."""

    def test_create_daily_activity_summary(self):
        """Test creating a daily activity summary."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        # ARRANGE & ACT
        summary = DailyActivitySummary(
            date=date(2024, 1, 1),
            total_steps=10000,
            total_active_energy=450.0,
            total_distance_km=7.5,
            activity_sessions=5,
            peak_activity_hour=14,
            activity_variance=0.25,
            sedentary_hours=8,
            active_hours=16,
            earliest_activity=time(6, 0),
            latest_activity=time(22, 30),
        )

        # ASSERT
        assert summary.date == date(2024, 1, 1)
        assert summary.total_steps == 10000
        assert summary.total_active_energy == 450.0
        assert summary.activity_variance == 0.25

    def test_daily_summary_is_immutable(self):
        """Test that daily summary cannot be modified."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        # ARRANGE
        summary = DailyActivitySummary(
            date=date(2024, 1, 1),
            total_steps=10000,
        )

        # ACT & ASSERT
        with pytest.raises(AttributeError):
            summary.total_steps = 12000

    def test_clinically_significant_high_activity(self):
        """Test detection of manic-level activity."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        # ARRANGE - Very high step count (>15000)
        summary = DailyActivitySummary(
            date=date(2024, 1, 1),
            total_steps=18000,  # Manic indicator
            total_active_energy=600.0,
            activity_variance=0.15,  # Consistent high activity
        )

        # ASSERT
        assert summary.is_clinically_significant
        assert summary.is_high_activity
        assert not summary.is_low_activity

    def test_clinically_significant_low_activity(self):
        """Test detection of depressive-level activity."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        # ARRANGE - Very low step count (<2000)
        summary = DailyActivitySummary(
            date=date(2024, 1, 1),
            total_steps=800,  # Depression indicator
            total_active_energy=50.0,
            sedentary_hours=20,
        )

        # ASSERT
        assert summary.is_clinically_significant
        assert summary.is_low_activity
        assert not summary.is_high_activity

    def test_clinically_significant_erratic_pattern(self):
        """Test detection of erratic activity patterns."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        # ARRANGE - High variance in activity
        summary = DailyActivitySummary(
            date=date(2024, 1, 1),
            total_steps=8000,
            activity_variance=0.75,  # Very erratic pattern
        )

        # ASSERT
        assert summary.is_clinically_significant

    def test_normal_activity_not_significant(self):
        """Test normal activity is not clinically significant."""
        from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary

        # ARRANGE
        summary = DailyActivitySummary(
            date=date(2024, 1, 1),
            total_steps=8000,  # Normal range
            total_active_energy=350.0,
            activity_variance=0.25,  # Normal variance
            sedentary_hours=8,
        )

        # ASSERT
        assert not summary.is_clinically_significant
        assert not summary.is_high_activity
        assert not summary.is_low_activity

class TestActivityAggregator:
    """Test suite for ActivityAggregator service."""

    @pytest.fixture
    def aggregator(self):
        """Provide ActivityAggregator instance."""
        return ActivityAggregator()

    @pytest.fixture
    def single_day_records(self):
        """Provide activity records for a single day."""
        day = datetime(2024, 1, 1, tzinfo=UTC)
        return [
            # Morning walk
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=7),
                end_date=day.replace(hour=8),
                activity_type=ActivityType.STEP_COUNT,
                value=3000.0,
                unit="count",
            ),
            # Afternoon activity
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=14),
                end_date=day.replace(hour=15),
                activity_type=ActivityType.STEP_COUNT,
                value=2000.0,
                unit="count",
            ),
            # Evening walk
            ActivityRecord(
                source_name="iPhone",
                start_date=day.replace(hour=19),
                end_date=day.replace(hour=20),
                activity_type=ActivityType.STEP_COUNT,
                value=5000.0,
                unit="count",
            ),
            # Energy burn
            ActivityRecord(
                source_name="Apple Watch",
                start_date=day.replace(hour=7),
                end_date=day.replace(hour=20),
                activity_type=ActivityType.ACTIVE_ENERGY,
                value=450.0,
                unit="Cal",
            ),
        ]

    def test_aggregate_empty_records(self, aggregator):
        """Test aggregating empty list returns empty dict."""
        # ACT
        result = aggregator.aggregate_daily([])

        # ASSERT
        assert result == {}

    def test_aggregate_single_day(self, aggregator, single_day_records):
        """Test aggregating a single day's activity."""
        # ACT
        result = aggregator.aggregate_daily(single_day_records)

        # ASSERT
        assert len(result) == 1
        assert date(2024, 1, 1) in result

        summary = result[date(2024, 1, 1)]
        assert summary.total_steps == 10000  # 3000 + 2000 + 5000
        assert summary.total_active_energy == 450.0
        assert summary.activity_sessions == 4
        assert summary.peak_activity_hour == 19  # Hour with most steps

    def test_aggregate_multiple_sources(self, aggregator):
        """Test aggregating from multiple sources."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        # ARRANGE
        records = [
            ActivityRecord(
                "iPhone",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                ActivityType.STEP_COUNT,
                3000.0,
                "count",
            ),
            ActivityRecord(
                "Apple Watch",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                ActivityType.STEP_COUNT,
                3100.0,  # Watch usually more accurate
                "count",
            ),
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        # Should prefer higher value when overlapping
        assert summary.total_steps == 3100.0

    def test_activity_variance_calculation(self, aggregator):
        """Test calculation of activity variance (erratic patterns)."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        # ARRANGE - Highly variable activity
        day = datetime(2024, 1, 1, tzinfo=UTC)
        records = []

        # Simulate erratic pattern: very high then very low activity
        for hour in [8, 10, 12, 14, 16, 18]:
            value = 5000.0 if hour % 4 == 0 else 100.0  # Alternating high/low
            records.append(
                ActivityRecord(
                    "iPhone",
                    day.replace(hour=hour),
                    day.replace(hour=hour + 1),
                    ActivityType.STEP_COUNT,
                    value,
                    "count",
                )
            )

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.activity_variance > 0.5  # High variance

    def test_sedentary_vs_active_hours(self, aggregator):
        """Test calculation of sedentary vs active hours."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        # ARRANGE
        day = datetime(2024, 1, 1, tzinfo=UTC)
        records = [
            # Active hours (7-9 AM)
            ActivityRecord(
                "iPhone",
                day.replace(hour=7),
                day.replace(hour=9),
                ActivityType.STEP_COUNT,
                2000.0,
                "count",
            ),
            # Sedentary period (9 AM - 5 PM)
            ActivityRecord(
                "iPhone",
                day.replace(hour=9),
                day.replace(hour=17),
                ActivityType.STEP_COUNT,
                500.0,  # Very low for 8 hours
                "count",
            ),
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.active_hours < summary.sedentary_hours

    def test_circadian_activity_markers(self, aggregator):
        """Test extraction of circadian rhythm markers."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        # ARRANGE
        day = datetime(2024, 1, 1, tzinfo=UTC)
        records = [
            ActivityRecord(
                "iPhone",
                day.replace(hour=6, minute=30),  # Early morning
                day.replace(hour=7),
                ActivityType.STEP_COUNT,
                500.0,
                "count",
            ),
            ActivityRecord(
                "iPhone",
                day.replace(hour=22, minute=45),  # Late evening
                day.replace(hour=23),
                ActivityType.STEP_COUNT,
                200.0,
                "count",
            ),
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.earliest_activity == time(6, 30)
        assert summary.latest_activity == time(22, 45)

    def test_multiple_days_aggregation(self, aggregator):
        """Test aggregating activity across multiple days."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        # ARRANGE
        records = []
        for day_offset in range(3):
            day = datetime(2024, 1, day_offset + 1, tzinfo=UTC)
            records.append(
                ActivityRecord(
                    "iPhone",
                    day.replace(hour=10),
                    day.replace(hour=11),
                    ActivityType.STEP_COUNT,
                    5000.0,
                    "count",
                )
            )

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        assert len(result) == 3
        for day_num in range(1, 4):
            assert date(2024, 1, day_num) in result
            assert result[date(2024, 1, day_num)].total_steps == 5000.0

    def test_activity_type_aggregation(self, aggregator):
        """Test proper aggregation of different activity types."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )

        # ARRANGE
        day = datetime(2024, 1, 1, tzinfo=UTC)
        records = [
            ActivityRecord(
                "iPhone",
                day.replace(hour=10),
                day.replace(hour=11),
                ActivityType.STEP_COUNT,
                5000.0,
                "count",
            ),
            ActivityRecord(
                "iPhone",
                day.replace(hour=10),
                day.replace(hour=11),
                ActivityType.DISTANCE_WALKING,
                3.5,
                "km",
            ),
            ActivityRecord(
                "Apple Watch",
                day.replace(hour=10),
                day.replace(hour=11),
                ActivityType.ACTIVE_ENERGY,
                200.0,
                "Cal",
            ),
            ActivityRecord(
                "iPhone",
                day.replace(hour=10),
                day.replace(hour=11),
                ActivityType.FLIGHTS_CLIMBED,
                10.0,
                "count",
            ),
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.total_steps == 5000.0
        assert summary.total_distance_km == 3.5
        assert summary.total_active_energy == 200.0
        assert summary.flights_climbed == 10.0
