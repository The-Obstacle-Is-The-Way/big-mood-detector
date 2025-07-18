"""
Unit tests for Activity Sequence Extractor

Tests minute-level activity sequence extraction for PAT (Principal Activity Time)
calculation, critical for circadian rhythm analysis in bipolar disorder.
"""

from datetime import date, datetime, timedelta

import numpy as np
import pytest
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
)


class TestActivitySequenceExtractor:
    """Test minute-level activity sequence extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with default settings."""
        return ActivitySequenceExtractor()

    def _create_step_record(self, timestamp: datetime, steps: int) -> ActivityRecord:
        """Helper to create step count records."""
        return ActivityRecord(
            source_name="test",
            start_date=timestamp,
            end_date=timestamp,
            activity_type=ActivityType.STEP_COUNT,
            value=float(steps),
            unit="steps",
        )

    def _create_interval_record(
        self, start: datetime, end: datetime, steps: int
    ) -> ActivityRecord:
        """Create record spanning a time interval."""
        return ActivityRecord(
            source_name="test",
            start_date=start,
            end_date=end,
            activity_type=ActivityType.STEP_COUNT,
            value=float(steps),
            unit="steps",
        )

    def test_empty_records_creates_zero_sequence(self, extractor):
        """Empty records should create all-zero sequence."""
        # Arrange
        target_date = date(2024, 1, 15)

        # Act
        sequence = extractor.extract_daily_sequence([], target_date)

        # Assert
        assert sequence.date == target_date
        assert len(sequence.activity_values) == 1440  # 24 * 60 minutes
        assert all(v == 0 for v in sequence.activity_values)
        assert sequence.total_activity == 0

    def test_single_record_fills_correct_minutes(self, extractor):
        """Single activity record should fill the correct minute slots."""
        # Arrange
        record = self._create_step_record(
            datetime(2024, 1, 15, 14, 30),
            1000,  # 2:30 PM
        )

        # Act
        sequence = extractor.extract_daily_sequence([record], date(2024, 1, 15))

        # Assert
        minute_index = 14 * 60 + 30  # 870
        assert sequence.activity_values[minute_index] == 1000
        assert sequence.total_activity == 1000
        # All other minutes should be zero
        assert sum(1 for v in sequence.activity_values if v > 0) == 1

    def test_multiple_records_aggregate_by_minute(self, extractor):
        """Multiple records in same minute should sum."""
        # Arrange
        records = [
            self._create_step_record(datetime(2024, 1, 15, 10, 15, 0), 500),
            self._create_step_record(
                datetime(2024, 1, 15, 10, 15, 30), 300
            ),  # Same minute
        ]

        # Act
        sequence = extractor.extract_daily_sequence(records, date(2024, 1, 15))

        # Assert
        minute_index = 10 * 60 + 15  # 615
        assert sequence.activity_values[minute_index] == 800  # 500 + 300

    def test_records_from_different_days_filtered(self, extractor):
        """Only records from target date should be included."""
        # Arrange
        records = [
            self._create_step_record(
                datetime(2024, 1, 14, 23, 59), 1000
            ),  # Previous day
            self._create_step_record(datetime(2024, 1, 15, 0, 1), 500),  # Target day
            self._create_step_record(datetime(2024, 1, 16, 0, 0), 2000),  # Next day
        ]

        # Act
        sequence = extractor.extract_daily_sequence(records, date(2024, 1, 15))

        # Assert
        assert sequence.total_activity == 500  # Only the target day record
        assert sequence.activity_values[1] == 500  # Minute 1 (00:01)

    def test_pat_calculation_basic(self, extractor):
        """Test PAT (Principal Activity Time) calculation."""
        # Arrange - simulate morning activity pattern
        records = []
        # Morning burst (8-10 AM)
        for hour in range(8, 10):
            for minute in range(0, 60, 5):
                records.append(
                    self._create_step_record(
                        datetime(2024, 1, 15, hour, minute),
                        100,  # High morning activity
                    )
                )

        # Light afternoon activity (2-3 PM)
        for minute in range(0, 60, 10):
            records.append(
                self._create_step_record(
                    datetime(2024, 1, 15, 14, minute),
                    50,  # Lower afternoon activity
                )
            )

        # Act
        pat_result = extractor.calculate_pat(records, date(2024, 1, 15))

        # Assert
        assert pat_result.pat_hour >= 8 and pat_result.pat_hour <= 10
        assert pat_result.morning_activity > pat_result.evening_activity
        assert pat_result.activity_concentration > 0  # Should have some concentration

    def test_pat_evening_type(self, extractor):
        """Test PAT for evening activity pattern."""
        # Arrange - simulate evening activity pattern
        records = []
        # Evening burst (8-10 PM)
        for hour in range(20, 22):
            for minute in range(0, 60, 5):
                records.append(
                    self._create_step_record(datetime(2024, 1, 15, hour, minute), 150)
                )

        # Act
        pat_result = extractor.calculate_pat(records, date(2024, 1, 15))

        # Assert
        assert pat_result.pat_hour >= 20 and pat_result.pat_hour <= 22
        assert pat_result.evening_activity > pat_result.morning_activity
        assert pat_result.is_evening_type is True

    def test_activity_concentration_uniform(self, extractor):
        """Uniform activity should have low concentration."""
        # Arrange - activity spread throughout day
        records = []
        for hour in range(6, 22):  # 6 AM to 10 PM
            records.append(
                self._create_step_record(
                    datetime(2024, 1, 15, hour, 0),
                    100,  # Same activity each hour
                )
            )

        # Act
        pat_result = extractor.calculate_pat(records, date(2024, 1, 15))

        # Assert
        assert (
            pat_result.activity_concentration < 0.1
        )  # Very low concentration (uniform)
        assert (
            pat_result.peak_activity_minutes <= 16
        )  # Individual minutes, not continuous

    def test_activity_concentration_focused(self, extractor):
        """Focused activity burst should have high concentration."""
        # Arrange - single intense hour
        records = []
        for minute in range(60):
            records.append(
                self._create_step_record(
                    datetime(2024, 1, 15, 15, minute),
                    200,  # 3 PM hour
                )
            )

        # Act
        pat_result = extractor.calculate_pat(records, date(2024, 1, 15))

        # Assert
        assert pat_result.activity_concentration > 0.8  # High concentration
        assert pat_result.peak_activity_minutes == 60  # One hour burst
        assert 15 <= pat_result.pat_hour < 16  # Within 3 PM hour

    def test_sequence_statistics(self, extractor):
        """Test sequence statistical properties."""
        # Arrange - create pattern with known properties
        records = [
            self._create_step_record(datetime(2024, 1, 15, 10, 0), 100),
            self._create_step_record(datetime(2024, 1, 15, 14, 0), 200),
            self._create_step_record(datetime(2024, 1, 15, 18, 0), 300),
        ]

        # Act
        sequence = extractor.extract_daily_sequence(records, date(2024, 1, 15))

        # Assert
        assert sequence.total_activity == 600
        assert sequence.max_minute_activity == 300
        assert sequence.active_minutes == 3
        assert sequence.get_percentile(100) == 300
        assert sequence.get_percentile(0) == 0

    def test_moving_average_smoothing(self, extractor):
        """Test moving average for smoothing noisy data."""
        # Arrange - create noisy pattern
        records = []
        np.random.seed(42)
        for hour in range(10, 12):
            for minute in range(0, 60, 2):
                # Add noise to base activity
                noise = np.random.randint(-20, 20)
                steps = max(0, 100 + noise)
                records.append(
                    self._create_step_record(datetime(2024, 1, 15, hour, minute), steps)
                )

        # Act
        sequence = extractor.extract_daily_sequence(records, date(2024, 1, 15))
        smoothed = sequence.get_smoothed_values(window_size=5)

        # Assert
        assert len(smoothed) == 1440
        # Smoothed should have less variance than original
        original_variance = np.var([v for v in sequence.activity_values if v > 0])
        smoothed_variance = np.var([v for v in smoothed if v > 0])
        assert smoothed_variance < original_variance

    def test_circadian_alignment_with_sleep(self, extractor):
        """Test activity alignment with sleep windows."""
        # Arrange - morning activity (good alignment)
        morning_records = []
        for hour in range(7, 12):
            morning_records.append(
                self._create_step_record(datetime(2024, 1, 15, hour, 0), 200)
            )

        # Typical sleep window: 11 PM - 7 AM
        sleep_start_hour = 23
        sleep_end_hour = 7

        # Act
        alignment_score = extractor.calculate_circadian_alignment(
            morning_records,
            date(2024, 1, 15),
            sleep_start_hour=sleep_start_hour,
            sleep_end_hour=sleep_end_hour,
        )

        # Assert
        assert alignment_score > 0.8  # Good alignment

    def test_circadian_misalignment(self, extractor):
        """Test detection of circadian misalignment."""
        # Arrange - night activity (poor alignment)
        night_records = []
        for hour in range(23, 24):
            night_records.append(
                self._create_step_record(datetime(2024, 1, 15, hour, 0), 300)
            )
        for hour in range(0, 3):
            night_records.append(
                self._create_step_record(datetime(2024, 1, 16, hour, 0), 300)
            )

        # Act
        alignment_score = extractor.calculate_circadian_alignment(
            night_records, date(2024, 1, 15), sleep_start_hour=23, sleep_end_hour=7
        )

        # Assert
        assert alignment_score < 0.3  # Poor alignment

    def test_extract_minute_sequence_basic(self, extractor):
        """Extract sequence across multiple days."""
        start = datetime(2024, 1, 15, 23, 50)
        end = start + timedelta(minutes=20)
        records = [self._create_interval_record(start, end, 200)]

        seq = extractor.extract_minute_sequence(records, days=2)

        assert len(seq) == 2880
        # Value spreads over 21 minutes
        assert pytest.approx(seq[23 * 60 + 50], abs=1e-3) == pytest.approx(
            200 / 21, abs=1e-3
        )
        assert sum(seq) == pytest.approx(200, rel=1e-6)

    def test_extract_minute_sequence_overlap(self, extractor):
        """Overlapping records should accumulate."""
        start = datetime(2024, 1, 15, 10, 0)
        rec1 = self._create_interval_record(start, start + timedelta(minutes=2), 60)
        rec2 = self._create_interval_record(
            start + timedelta(minutes=1), start + timedelta(minutes=3), 60
        )
        seq = extractor.extract_minute_sequence([rec1, rec2], days=1)

        # Minutes 10:00, 10:01, 10:02, 10:03
        assert len(seq) == 1440
        assert seq[10 * 60] > 0
        # Overlap minute 10:01 should have sum of both increments
        assert seq[10 * 60 + 1] > seq[10 * 60]
        assert sum(seq) == pytest.approx(120, rel=1e-6)

    def test_extract_minute_sequence_no_records(self, extractor):
        """No records should yield zeros."""
        seq = extractor.extract_minute_sequence([], days=3)
        assert len(seq) == 4320
        assert np.all(seq == 0)
