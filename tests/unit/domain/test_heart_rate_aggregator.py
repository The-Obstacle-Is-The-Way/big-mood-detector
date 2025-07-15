"""
Tests for Heart Rate Aggregator Domain Service

Test-driven development for clinical heart rate aggregation.
Following Clean Architecture and Uncle Bob's principles.
"""

from datetime import UTC, date, datetime

import pytest

from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)
from big_mood_detector.domain.services.heart_rate_aggregator import (
    DailyHeartSummary,
    HeartRateAggregator,
)


class TestDailyHeartSummary:
    """Test suite for DailyHeartSummary value object."""

    def test_create_daily_heart_summary(self):
        """Test creating a daily heart summary."""
        # ARRANGE & ACT
        summary = DailyHeartSummary(
            date=date(2024, 1, 1),
            avg_resting_hr=65.0,
            min_hr=48.0,
            max_hr=145.0,
            avg_hrv_sdnn=45.5,
            hr_measurements=150,
            hrv_measurements=10,
            high_hr_episodes=2,
            low_hr_episodes=1,
            circadian_hr_range=30.0,
            morning_hr=58.0,
            evening_hr=72.0,
        )

        # ASSERT
        assert summary.date == date(2024, 1, 1)
        assert summary.avg_resting_hr == 65.0
        assert summary.avg_hrv_sdnn == 45.5
        assert summary.circadian_hr_range == 30.0

    def test_daily_summary_is_immutable(self):
        """Test that daily summary cannot be modified."""
        # ARRANGE
        summary = DailyHeartSummary(
            date=date(2024, 1, 1),
            avg_resting_hr=65.0,
        )

        # ACT & ASSERT
        with pytest.raises(AttributeError):
            summary.avg_resting_hr = 70.0

    def test_clinically_significant_high_resting_hr(self):
        """Test detection of elevated resting heart rate."""
        # ARRANGE - High resting HR (>90)
        summary = DailyHeartSummary(
            date=date(2024, 1, 1),
            avg_resting_hr=95.0,  # Indicates stress/anxiety/mania
            high_hr_episodes=15,  # Many episodes
        )

        # ASSERT
        assert summary.is_clinically_significant
        assert summary.has_high_resting_hr

    def test_clinically_significant_low_hrv(self):
        """Test detection of low HRV (poor autonomic function)."""
        # ARRANGE - Low HRV
        summary = DailyHeartSummary(
            date=date(2024, 1, 1),
            avg_resting_hr=65.0,
            avg_hrv_sdnn=15.0,  # <20ms indicates poor recovery
        )

        # ASSERT
        assert summary.is_clinically_significant
        assert summary.has_low_hrv

    def test_clinically_significant_abnormal_circadian(self):
        """Test detection of abnormal circadian rhythm."""
        # ARRANGE - Flat circadian rhythm
        summary = DailyHeartSummary(
            date=date(2024, 1, 1),
            avg_resting_hr=65.0,
            circadian_hr_range=5.0,  # <10 bpm range is abnormal
            morning_hr=68.0,
            evening_hr=70.0,
        )

        # ASSERT
        assert summary.is_clinically_significant
        assert summary.has_abnormal_circadian_rhythm

    def test_normal_heart_metrics_not_significant(self):
        """Test normal heart metrics are not clinically significant."""
        # ARRANGE
        summary = DailyHeartSummary(
            date=date(2024, 1, 1),
            avg_resting_hr=65.0,
            avg_hrv_sdnn=45.0,
            circadian_hr_range=20.0,
            high_hr_episodes=0,
            low_hr_episodes=0,
        )

        # ASSERT
        assert not summary.is_clinically_significant
        assert not summary.has_high_resting_hr
        assert not summary.has_low_hrv


class TestHeartRateAggregator:
    """Test suite for HeartRateAggregator service."""

    @pytest.fixture
    def aggregator(self):
        """Provide HeartRateAggregator instance."""
        return HeartRateAggregator()

    @pytest.fixture
    def single_day_records(self):
        """Provide heart rate records for a single day."""
        day = datetime(2024, 1, 1, tzinfo=UTC)
        records = []

        # Morning resting HR
        for hour in range(6, 9):
            records.append(
                HeartRateRecord(
                    source_name="Apple Watch",
                    timestamp=day.replace(hour=hour),
                    metric_type=HeartMetricType.HEART_RATE,
                    value=58.0 + hour - 6,  # 58, 59, 60
                    unit="count/min",
                    motion_context=MotionContext.SEDENTARY,
                )
            )

        # Daytime measurements
        for hour in range(10, 18):
            records.append(
                HeartRateRecord(
                    source_name="Apple Watch",
                    timestamp=day.replace(hour=hour),
                    metric_type=HeartMetricType.HEART_RATE,
                    value=70.0 + (hour - 10) * 2,  # 70-84
                    unit="count/min",
                    motion_context=MotionContext.SEDENTARY,
                )
            )

        # HRV measurements
        for hour in [7, 12, 19]:
            records.append(
                HeartRateRecord(
                    source_name="Apple Watch",
                    timestamp=day.replace(hour=hour),
                    metric_type=HeartMetricType.HRV_SDNN,
                    value=40.0 + hour / 2,  # Varies by time
                    unit="ms",
                )
            )

        return records

    def test_aggregate_empty_records(self, aggregator):
        """Test aggregating empty list returns empty dict."""
        # ACT
        result = aggregator.aggregate_daily([])

        # ASSERT
        assert result == {}

    def test_aggregate_single_day(self, aggregator, single_day_records):
        """Test aggregating a single day's heart data."""
        # ACT
        result = aggregator.aggregate_daily(single_day_records)

        # ASSERT
        assert len(result) == 1
        assert date(2024, 1, 1) in result

        summary = result[date(2024, 1, 1)]
        assert summary.hr_measurements == 11  # 3 morning + 8 daytime
        assert summary.hrv_measurements == 3
        assert summary.avg_resting_hr > 0
        assert summary.avg_hrv_sdnn > 0

    def test_resting_hr_calculation(self, aggregator):
        """Test correct calculation of resting heart rate."""
        # ARRANGE - Only sedentary measurements
        records = [
            HeartRateRecord(
                "Apple Watch",
                datetime(2024, 1, 1, hour, tzinfo=UTC),
                HeartMetricType.HEART_RATE,
                60.0 + hour,  # 60, 61, 62
                "count/min",
                MotionContext.SEDENTARY,
            )
            for hour in range(3)
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.avg_resting_hr == 61.0  # (60 + 61 + 62) / 3

    def test_exclude_active_from_resting_hr(self, aggregator):
        """Test that active HR is excluded from resting calculation."""
        # ARRANGE
        records = [
            HeartRateRecord(
                "Apple Watch",
                datetime(2024, 1, 1, 10, tzinfo=UTC),
                HeartMetricType.HEART_RATE,
                65.0,
                "count/min",
                MotionContext.SEDENTARY,
            ),
            HeartRateRecord(
                "Apple Watch",
                datetime(2024, 1, 1, 11, tzinfo=UTC),
                HeartMetricType.HEART_RATE,
                120.0,  # High but active
                "count/min",
                MotionContext.ACTIVE,
            ),
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.avg_resting_hr == 65.0  # Only sedentary included

    def test_high_low_episode_counting(self, aggregator):
        """Test counting of high/low HR episodes."""
        # ARRANGE
        records = []
        day = datetime(2024, 1, 1, tzinfo=UTC)

        # High HR episodes (>100 at rest)
        for i in range(5):
            records.append(
                HeartRateRecord(
                    "Apple Watch",
                    day.replace(hour=10, minute=i),
                    HeartMetricType.HEART_RATE,
                    105.0,
                    "count/min",
                    MotionContext.SEDENTARY,
                )
            )

        # Low HR episodes (<50)
        for i in range(3):
            records.append(
                HeartRateRecord(
                    "Apple Watch",
                    day.replace(hour=4, minute=i),
                    HeartMetricType.HEART_RATE,
                    45.0,
                    "count/min",
                )
            )

        # Normal HR
        records.append(
            HeartRateRecord(
                "Apple Watch",
                day.replace(hour=12),
                HeartMetricType.HEART_RATE,
                72.0,
                "count/min",
            )
        )

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.high_hr_episodes == 5
        assert summary.low_hr_episodes == 3

    def test_circadian_rhythm_calculation(self, aggregator):
        """Test calculation of circadian rhythm markers."""
        # ARRANGE
        day = datetime(2024, 1, 1, tzinfo=UTC)
        records = [
            # Morning readings (6-9 AM)
            HeartRateRecord(
                "Apple Watch",
                day.replace(hour=7),
                HeartMetricType.HEART_RATE,
                55.0,
                "count/min",
                MotionContext.SEDENTARY,
            ),
            # Evening readings (6-10 PM)
            HeartRateRecord(
                "Apple Watch",
                day.replace(hour=20),
                HeartMetricType.HEART_RATE,
                75.0,
                "count/min",
                MotionContext.SEDENTARY,
            ),
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.morning_hr == 55.0
        assert summary.evening_hr == 75.0
        assert summary.circadian_hr_range == 20.0  # 75 - 55

    def test_hrv_aggregation(self, aggregator):
        """Test HRV SDNN aggregation."""
        # ARRANGE
        records = [
            HeartRateRecord(
                "Apple Watch",
                datetime(2024, 1, 1, hour, tzinfo=UTC),
                HeartMetricType.HRV_SDNN,
                40.0 + hour * 2,  # 40, 42, 44
                "ms",
            )
            for hour in range(3)
        ]

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        summary = result[date(2024, 1, 1)]
        assert summary.avg_hrv_sdnn == 42.0  # (40 + 42 + 44) / 3

    def test_multiple_days_aggregation(self, aggregator):
        """Test aggregating heart data across multiple days."""
        # ARRANGE
        records = []
        for day_offset in range(3):
            timestamp = datetime(2024, 1, day_offset + 1, 10, tzinfo=UTC)
            records.append(
                HeartRateRecord(
                    "Apple Watch",
                    timestamp,
                    HeartMetricType.HEART_RATE,
                    65.0 + day_offset,
                    "count/min",
                    MotionContext.SEDENTARY,
                )
            )

        # ACT
        result = aggregator.aggregate_daily(records)

        # ASSERT
        assert len(result) == 3
        for day_num in range(1, 4):
            assert date(2024, 1, day_num) in result

