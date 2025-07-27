"""
Tests for SleepAggregator overlap handling.

These tests verify that overlapping sleep records from multiple devices
are correctly merged to avoid double-counting sleep duration.
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator


class TestSleepOverlapHandling:
    """Test cases for handling overlapping sleep records."""

    @pytest.fixture
    def aggregator(self) -> SleepAggregator:
        """Create a SleepAggregator instance."""
        return SleepAggregator()

    @pytest.fixture
    def test_date(self) -> date:
        """Test date for aggregation."""
        return date(2025, 7, 26)

    def test_merge_overlapping_sleep_records_from_two_devices(
        self, aggregator: SleepAggregator, test_date: date
    ) -> None:
        """Test that overlapping records from iPhone and Apple Watch are merged correctly."""
        # iPhone records 10pm-6am (8 hours)
        iphone_record = SleepRecord(
            source_name="iPhone",
            start_date=datetime(2025, 7, 25, 22, 0, tzinfo=UTC),  # 10pm
            end_date=datetime(2025, 7, 26, 6, 0, tzinfo=UTC),  # 6am next day
            state=SleepState.ASLEEP,
        )
        
        # Apple Watch records 10:30pm-6:30am (8 hours)
        watch_record = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2025, 7, 25, 22, 30, tzinfo=UTC),  # 10:30pm
            end_date=datetime(2025, 7, 26, 6, 30, tzinfo=UTC),  # 6:30am next day
            state=SleepState.ASLEEP,
        )
        
        # Aggregate both records
        summaries = aggregator.aggregate_daily([iphone_record, watch_record])
        
        # Should have one summary for the test date
        assert test_date in summaries
        summary = summaries[test_date]
        
        # Total sleep should be 8.5 hours (10pm to 6:30am), NOT 16 hours
        assert summary.total_sleep_hours == pytest.approx(8.5, rel=0.01)
        assert summary.total_time_in_bed_hours == pytest.approx(8.5, rel=0.01)

    def test_merge_multiple_overlapping_sources(
        self, aggregator: SleepAggregator, test_date: date
    ) -> None:
        """Test handling of iPhone, Apple Watch, and manual entry overlaps."""
        records = [
            # iPhone: 10pm-6am (8 hours)
            SleepRecord(
                source_name="iPhone",
                start_date=datetime(2025, 7, 25, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 6, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
            # Apple Watch: 10:30pm-6:30am (8 hours)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2025, 7, 25, 22, 30, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 6, 30, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
            # Manual entry: 11pm-7am (8 hours)
            SleepRecord(
                source_name="Manual Entry",
                start_date=datetime(2025, 7, 25, 23, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 7, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
        ]
        
        summaries = aggregator.aggregate_daily(records)
        summary = summaries[test_date]
        
        # Total sleep should be 9 hours (10pm to 7am), NOT 24 hours
        assert summary.total_sleep_hours == pytest.approx(9.0, rel=0.01)

    def test_non_overlapping_records_unchanged(
        self, aggregator: SleepAggregator, test_date: date
    ) -> None:
        """Test that non-overlapping records are not affected by merging."""
        records = [
            # Night sleep: 11pm-7am (8 hours)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2025, 7, 25, 23, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 7, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
            # Afternoon nap: 2pm-3pm (1 hour)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2025, 7, 26, 14, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 15, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
        ]
        
        summaries = aggregator.aggregate_daily(records)
        summary = summaries[test_date]
        
        # Total should be exactly 9 hours (no overlap to merge)
        assert summary.total_sleep_hours == pytest.approx(9.0, rel=0.01)

    def test_partial_overlap_merging(
        self, aggregator: SleepAggregator, test_date: date
    ) -> None:
        """Test merging of partially overlapping records."""
        records = [
            # First segment: 10pm-2am (4 hours)
            SleepRecord(
                source_name="iPhone",
                start_date=datetime(2025, 7, 25, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 2, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
            # Second segment: 1am-6am (5 hours, 1 hour overlap)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2025, 7, 26, 1, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 6, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
        ]
        
        summaries = aggregator.aggregate_daily(records)
        summary = summaries[test_date]
        
        # Total should be 8 hours (10pm to 6am), NOT 9 hours
        assert summary.total_sleep_hours == pytest.approx(8.0, rel=0.01)

    def test_complex_overlap_scenario(
        self, aggregator: SleepAggregator, test_date: date
    ) -> None:
        """Test a complex real-world scenario with multiple overlapping segments."""
        records = [
            # iPhone segment 1: 10pm-1am (3 hours)
            SleepRecord(
                source_name="iPhone",
                start_date=datetime(2025, 7, 25, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 1, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
            # Watch continuous: 10:30pm-6:30am (8 hours)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2025, 7, 25, 22, 30, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 6, 30, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
            # iPhone segment 2: 4am-7am (3 hours)
            SleepRecord(
                source_name="iPhone",
                start_date=datetime(2025, 7, 26, 4, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 7, 0, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
        ]
        
        summaries = aggregator.aggregate_daily(records)
        summary = summaries[test_date]
        
        # Total should be 9 hours (10pm to 7am), NOT 14 hours
        assert summary.total_sleep_hours == pytest.approx(9.0, rel=0.01)

    def test_in_bed_vs_asleep_overlap_handling(
        self, aggregator: SleepAggregator, test_date: date
    ) -> None:
        """Test that IN_BED and ASLEEP states are handled correctly during overlap."""
        records = [
            # In bed: 9:30pm-7am (9.5 hours)
            SleepRecord(
                source_name="iPhone",
                start_date=datetime(2025, 7, 25, 21, 30, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 7, 0, tzinfo=UTC),
                state=SleepState.IN_BED,
            ),
            # Asleep: 10pm-6:30am (8.5 hours)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2025, 7, 25, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 6, 30, tzinfo=UTC),
                state=SleepState.ASLEEP,
            ),
        ]
        
        summaries = aggregator.aggregate_daily(records)
        summary = summaries[test_date]
        
        # Time in bed should be 9.5 hours (full range)
        assert summary.total_time_in_bed_hours == pytest.approx(9.5, rel=0.01)
        # Sleep time should be 8.5 hours (only ASLEEP records)
        assert summary.total_sleep_hours == pytest.approx(8.5, rel=0.01)
        # Efficiency should be 8.5/9.5 = ~89.5%
        assert summary.sleep_efficiency == pytest.approx(0.895, rel=0.01)

    def test_warning_for_excessive_overlap(
        self, aggregator: SleepAggregator, test_date: date, caplog
    ) -> None:
        """Test that a warning is logged when overlap is detected."""
        # Create records that sum to >24 hours
        records = [
            SleepRecord(
                source_name=f"Device{i}",
                start_date=datetime(2025, 7, 25, 22, 0, tzinfo=UTC),
                end_date=datetime(2025, 7, 26, 8, 0, tzinfo=UTC),  # 10 hours each
                state=SleepState.ASLEEP,
            )
            for i in range(3)  # 3 devices = 30 hours total
        ]
        
        summaries = aggregator.aggregate_daily(records)
        summary = summaries[test_date]
        
        # Should be capped at reasonable value after merging
        assert summary.total_sleep_hours <= 24.0
        
        # Should have logged a warning about overlapping records
        assert any(
            "overlapping sleep records" in record.message.lower()
            for record in caplog.records
        )