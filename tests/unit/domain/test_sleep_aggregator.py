"""
Tests for Sleep Aggregator Domain Service

Test-driven development for clinical sleep aggregation.
"""

from datetime import UTC, date, datetime, time, timedelta

import pytest

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.sleep_aggregator import (
    DailySleepSummary,
    SleepAggregator,
)


class TestDailySleepSummary:
    """Test suite for DailySleepSummary value object."""

    def test_create_daily_summary(self):
        """Test creating a daily sleep summary."""
        # ARRANGE & ACT
        summary = DailySleepSummary(
            date=date(2024, 1, 1),
            total_time_in_bed_hours=8.5,
            total_sleep_hours=7.5,
            sleep_efficiency=0.88,
            sleep_sessions=1,
            longest_sleep_hours=7.5,
            sleep_fragmentation_index=0.0,
            earliest_bedtime=time(23, 0),
            latest_wake_time=time(7, 30),
            mid_sleep_time=datetime(2024, 1, 2, 3, 15, tzinfo=UTC),
        )

        # ASSERT
        assert summary.date == date(2024, 1, 1)
        assert summary.total_sleep_hours == 7.5
        assert summary.sleep_efficiency == 0.88
        assert not summary.is_clinically_significant

    def test_clinically_significant_too_little_sleep(self):
        """Test detection of clinically significant sleep deprivation."""
        # ARRANGE
        summary = DailySleepSummary(
            date=date(2024, 1, 1),
            total_time_in_bed_hours=4.0,
            total_sleep_hours=3.5,  # < 4 hours
            sleep_efficiency=0.88,
            sleep_sessions=1,
            longest_sleep_hours=3.5,
            sleep_fragmentation_index=0.0,
        )

        # ASSERT
        assert summary.is_clinically_significant

    def test_clinically_significant_too_much_sleep(self):
        """Test detection of clinically significant hypersomnia."""
        # ARRANGE
        summary = DailySleepSummary(
            date=date(2024, 1, 1),
            total_time_in_bed_hours=11.0,
            total_sleep_hours=10.5,  # > 10 hours
            sleep_efficiency=0.95,
            sleep_sessions=1,
            longest_sleep_hours=10.5,
            sleep_fragmentation_index=0.0,
        )

        # ASSERT
        assert summary.is_clinically_significant

    def test_clinically_significant_poor_efficiency(self):
        """Test detection of poor sleep efficiency."""
        # ARRANGE
        summary = DailySleepSummary(
            date=date(2024, 1, 1),
            total_time_in_bed_hours=8.0,
            total_sleep_hours=5.0,
            sleep_efficiency=0.625,  # < 0.70
            sleep_sessions=1,
            longest_sleep_hours=5.0,
            sleep_fragmentation_index=0.0,
        )

        # ASSERT
        assert summary.is_clinically_significant

    def test_clinically_significant_high_fragmentation(self):
        """Test detection of fragmented sleep."""
        # ARRANGE
        summary = DailySleepSummary(
            date=date(2024, 1, 1),
            total_time_in_bed_hours=8.0,
            total_sleep_hours=7.0,
            sleep_efficiency=0.88,
            sleep_sessions=4,
            longest_sleep_hours=3.0,
            sleep_fragmentation_index=0.35,  # > 0.3
        )

        # ASSERT
        assert summary.is_clinically_significant


class TestSleepAggregator:
    """Test suite for SleepAggregator service."""

    @pytest.fixture
    def aggregator(self):
        """Provide SleepAggregator instance."""
        return SleepAggregator()

    @pytest.fixture
    def single_night_records(self):
        """Provide sleep records for a single night."""
        night = datetime(2024, 1, 1, 23, 0, tzinfo=UTC)
        return [
            SleepRecord(
                source_name="Apple Watch",
                start_date=night,
                end_date=night + timedelta(hours=0.5),
                state=SleepState.IN_BED,
            ),
            SleepRecord(
                source_name="Apple Watch",
                start_date=night + timedelta(hours=0.5),
                end_date=night + timedelta(hours=8.5),
                state=SleepState.ASLEEP,
            ),
        ]

    def test_aggregate_empty_records(self, aggregator):
        """Test aggregating empty list returns empty dict."""
        # ACT
        result = aggregator.aggregate_daily([])

        # ASSERT
        assert result == {}

    def test_aggregate_single_night(self, aggregator, single_night_records):
        """Test aggregating a single night's sleep."""
        # ACT
        result = aggregator.aggregate_daily(single_night_records)

        # ASSERT
        assert len(result) == 1
        assert date(2024, 1, 1) in result

        summary = result[date(2024, 1, 1)]
        assert summary.total_time_in_bed_hours == 8.5
        assert summary.total_sleep_hours == 8.0
        assert summary.sleep_efficiency == pytest.approx(0.941, rel=0.01)
        assert summary.sleep_sessions == 1
        assert summary.longest_sleep_hours == 8.0

    def test_fragmented_sleep_calculation(self, aggregator):
        """Test fragmentation index calculation."""
        # ARRANGE - Multiple sleep sessions with gaps
        night = datetime(2024, 1, 1, 23, 0, tzinfo=UTC)
        records = [
            SleepRecord(
                "Watch",
                night,
                night + timedelta(hours=3),
                SleepState.ASLEEP,
            ),
            SleepRecord(
                "Watch",
                night + timedelta(hours=3.5),  # 0.5 hour gap
                night + timedelta(hours=6),
                SleepState.ASLEEP,
            ),
            SleepRecord(
                "Watch",
                night + timedelta(hours=6.5),  # 0.5 hour gap
                night + timedelta(hours=8),
                SleepState.ASLEEP,
            ),
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.sleep_fragmentation_index == pytest.approx(0.125, rel=0.01)
        # Total gap time = 1 hour, total period = 8 hours, index = 1/8 = 0.125

    def test_sleep_date_assignment_evening(self, aggregator):
        """Test sleep starting in evening assigns to that date."""
        # ARRANGE - Sleep starting at 10 PM
        records = [
            SleepRecord(
                "Watch",
                datetime(2024, 1, 1, 22, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 6, 0, tzinfo=UTC),
                SleepState.ASLEEP,
            )
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        assert date(2024, 1, 1) in result
        assert date(2024, 1, 2) not in result

    def test_sleep_date_assignment_early_morning(self, aggregator):
        """Test sleep starting early morning assigns to previous date."""
        # ARRANGE - Sleep starting at 2 AM
        records = [
            SleepRecord(
                "Watch",
                datetime(2024, 1, 2, 2, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
                SleepState.ASLEEP,
            )
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        assert date(2024, 1, 1) in result
        assert date(2024, 1, 2) not in result

    def test_circadian_markers(self, aggregator):
        """Test calculation of circadian rhythm markers."""
        # ARRANGE
        night = datetime(2024, 1, 1, 23, 30, tzinfo=UTC)
        records = [
            SleepRecord(
                "Watch",
                night,
                night + timedelta(hours=8),
                SleepState.ASLEEP,
            )
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.earliest_bedtime == time(23, 30)
        assert summary.latest_wake_time == time(7, 30)
        assert summary.mid_sleep_time == datetime(2024, 1, 2, 3, 30, tzinfo=UTC)

    def test_multiple_days_aggregation(self, aggregator):
        """Test aggregating sleep across multiple days."""
        # ARRANGE
        records = []
        for day in range(3):
            night = datetime(2024, 1, day + 1, 23, 0, tzinfo=UTC)
            records.append(
                SleepRecord(
                    "Watch",
                    night,
                    night + timedelta(hours=8),
                    SleepState.ASLEEP,
                )
            )

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        assert len(result) == 3
        for day in range(1, 4):
            assert date(2024, 1, day) in result
            summary = result[date(2024, 1, day)]
            assert summary.total_sleep_hours == 8.0
