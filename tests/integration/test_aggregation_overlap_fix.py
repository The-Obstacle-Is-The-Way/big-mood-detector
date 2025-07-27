"""
Integration tests for sleep overlap fix in aggregation pipeline.

These tests verify that the sleep duration calculation fix
works correctly through the entire aggregation pipeline.
"""

import pytest
from datetime import UTC, date, datetime
from pathlib import Path

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationConfig,
    AggregationPipeline,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartRateRecord,
    HeartMetricType,
    MotionContext,
)


@pytest.mark.integration
class TestAggregationOverlapFix:
    """Test that overlapping sleep records are handled correctly in aggregation."""

    @pytest.fixture
    def config(self) -> AggregationConfig:
        """Create test configuration."""
        return AggregationConfig(
            window_size=7,
            min_window_size=3,
            enable_dlmo_calculation=False,  # Skip expensive calculations for tests
            enable_circadian_analysis=False,
        )

    @pytest.fixture
    def pipeline(self, config: AggregationConfig) -> AggregationPipeline:
        """Create aggregation pipeline."""
        return AggregationPipeline(config=config)

    @pytest.fixture
    def overlapping_sleep_records(self) -> list[SleepRecord]:
        """Create overlapping sleep records simulating iPhone + Apple Watch."""
        records = []
        base_date = date(2025, 7, 20)
        
        # Create 7 days of overlapping sleep data
        for day_offset in range(7):
            sleep_date = datetime(
                base_date.year, 
                base_date.month,
                base_date.day + day_offset,
                22, 0, 0,  # 10 PM
                tzinfo=UTC
            )
            
            # iPhone records 10pm-6am (8 hours)
            records.append(
                SleepRecord(
                    source_name="iPhone",
                    start_date=sleep_date,
                    end_date=sleep_date.replace(hour=6).replace(day=sleep_date.day + 1),
                    state=SleepState.ASLEEP,
                )
            )
            
            # Apple Watch records 10:30pm-6:30am (8 hours, overlapping)
            records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=sleep_date.replace(minute=30),
                    end_date=sleep_date.replace(hour=6, minute=30).replace(day=sleep_date.day + 1),
                    state=SleepState.ASLEEP,
                )
            )
        
        return records

    @pytest.fixture
    def minimal_activity_records(self) -> list[ActivityRecord]:
        """Create minimal activity records for the test period."""
        records = []
        base_date = date(2025, 7, 20)
        
        for day_offset in range(7):
            activity_date = datetime(
                base_date.year,
                base_date.month, 
                base_date.day + day_offset,
                12, 0, 0,  # Noon
                tzinfo=UTC
            )
            
            records.append(
                ActivityRecord(
                    source_name="iPhone",
                    start_date=activity_date,
                    end_date=activity_date.replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=1200.0,
                    unit="count",
                )
            )
        
        return records

    def test_aggregation_with_overlapping_sleep(
        self,
        pipeline: AggregationPipeline,
        overlapping_sleep_records: list[SleepRecord],
        minimal_activity_records: list[ActivityRecord],
    ) -> None:
        """Test that aggregation correctly handles overlapping sleep records."""
        # Define date range
        start_date = date(2025, 7, 20)
        end_date = date(2025, 7, 26)
        
        # Run aggregation
        features = pipeline.aggregate_daily_features(
            sleep_records=overlapping_sleep_records,
            activity_records=minimal_activity_records,
            heart_records=[],  # No heart data for this test
            start_date=start_date,
            end_date=end_date,
        )
        
        # Should have features for days 3-6 (window size requirement)
        assert len(features) >= 4
        
        # Check each day's sleep duration
        for feature_set in features:
            # Sleep duration should be ~8.5 hours (merged overlap)
            # NOT 16 hours (sum of both records)
            seoul_features = feature_set.seoul_features
            assert seoul_features is not None
            
            # The actual sleep duration should be realistic
            assert 7.0 <= seoul_features.sleep_duration_hours <= 9.0
            
            # Specifically should NOT be 16 hours
            assert seoul_features.sleep_duration_hours < 10.0

    def test_extreme_overlap_scenario(
        self,
        pipeline: AggregationPipeline,
        minimal_activity_records: list[ActivityRecord],
    ) -> None:
        """Test handling of extreme overlap (3+ devices recording simultaneously)."""
        # Create records from 3 devices all recording the same sleep
        sleep_records = []
        test_date = date(2025, 7, 20)
        
        for day_offset in range(7):
            sleep_start = datetime(
                test_date.year,
                test_date.month,
                test_date.day + day_offset,
                22, 0, 0,
                tzinfo=UTC
            )
            
            # Three devices all recording ~8 hours of sleep
            for device, offset_minutes in [
                ("iPhone", 0),
                ("Apple Watch", 15),
                ("Sleep Tracker App", 30),
            ]:
                records_start = sleep_start.replace(
                    minute=offset_minutes
                )
                records_end = records_start.replace(
                    hour=6,
                    minute=offset_minutes
                ).replace(day=records_start.day + 1)
                
                sleep_records.append(
                    SleepRecord(
                        source_name=device,
                        start_date=records_start,
                        end_date=records_end,
                        state=SleepState.ASLEEP,
                    )
                )
        
        # Run aggregation
        features = pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=minimal_activity_records,
            heart_records=[],
            start_date=test_date,
            end_date=test_date.replace(day=26),
        )
        
        # Check results
        for feature_set in features:
            seoul_features = feature_set.seoul_features
            assert seoul_features is not None
            
            # Should be ~8.5 hours (22:00 to 06:30), NOT 24 hours
            assert 7.0 <= seoul_features.sleep_duration_hours <= 9.0
            
            # Definitely should not exceed 24 hours
            assert seoul_features.sleep_duration_hours <= 24.0

    def test_non_overlapping_naps_preserved(
        self,
        pipeline: AggregationPipeline,
        minimal_activity_records: list[ActivityRecord],
    ) -> None:
        """Test that non-overlapping naps are correctly preserved."""
        sleep_records = []
        test_date = date(2025, 7, 20)
        
        for day_offset in range(7):
            current_date = test_date.replace(day=test_date.day + day_offset)
            
            # Night sleep: 11pm-7am (8 hours)
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=datetime(
                        current_date.year,
                        current_date.month,
                        current_date.day,
                        23, 0, 0,
                        tzinfo=UTC
                    ),
                    end_date=datetime(
                        current_date.year,
                        current_date.month,
                        current_date.day + 1,
                        7, 0, 0,
                        tzinfo=UTC
                    ),
                    state=SleepState.ASLEEP,
                )
            )
            
            # Afternoon nap: 2pm-3pm (1 hour)
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=datetime(
                        current_date.year,
                        current_date.month,
                        current_date.day + 1,
                        14, 0, 0,
                        tzinfo=UTC
                    ),
                    end_date=datetime(
                        current_date.year,
                        current_date.month,
                        current_date.day + 1,
                        15, 0, 0,
                        tzinfo=UTC
                    ),
                    state=SleepState.ASLEEP,
                )
            )
        
        # Run aggregation
        features = pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=minimal_activity_records,
            heart_records=[],
            start_date=test_date,
            end_date=test_date.replace(day=26),
        )
        
        # Check results
        for feature_set in features:
            seoul_features = feature_set.seoul_features
            assert seoul_features is not None
            
            # Should be ~9 hours total (8 + 1), not merged since no overlap
            assert 8.5 <= seoul_features.sleep_duration_hours <= 9.5