"""
Test to verify the aggregation pipeline fix is actually being used.

This test verifies that the sleep duration calculation in the 
aggregation pipeline now correctly uses SleepAggregator.
"""
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.application.services.aggregation_pipeline import AggregationPipeline


class TestAggregationPipelineFix:
    """Verify the sleep duration fix is working."""
    
    def test_aggregation_pipeline_uses_sleep_aggregator(self):
        """
        Test that _calculate_features_with_stats uses SleepAggregator
        for sleep duration calculation.
        """
        # Create pipeline
        pipeline = AggregationPipeline()
        
        # Create test sleep records for 7.5 hours
        sleep_records = [
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 1, 22, 0),
                end_date=datetime(2024, 1, 2, 5, 30),
                state=SleepState.ASLEEP
            )
        ]
        
        # Create minimal test data
        current_date = date(2024, 1, 1)
        daily_metrics = {
            "sleep": {
                "sleep_percentage": 0.1875,  # 4.5/24 - window percentage
                "sleep_amplitude": 0.0,
                "long_num": 1,
                "long_len": 4.5,
                "long_st": 4.5,
                "long_wt": 0.0,
                "short_num": 0,
                "short_len": 0.0,
                "short_st": 0.0,
                "short_wt": 0.0,
            }
        }
        
        sleep_window = [daily_metrics["sleep"]]
        circadian_window = []
        activity_metrics = {
            "daily_steps": 10000,
            "activity_variance": 100,
            "sedentary_hours": 12,
            "activity_fragmentation": 0.2,
            "sedentary_bout_mean": 2.0,
            "activity_intensity_ratio": 0.3,
        }
        
        # Call the method
        features = pipeline._calculate_features_with_stats(
            current_date,
            daily_metrics,
            sleep_window,
            circadian_window,
            activity_metrics,
            sleep_records,  # Pass the sleep records
        )
        
        # Verify the fix worked
        assert features is not None
        assert features.seoul_features is not None
        
        # The key test: sleep_duration_hours should be 7.5, not 4.5
        assert features.seoul_features.sleep_duration_hours == 7.5, (
            f"Expected 7.5 hours, got {features.seoul_features.sleep_duration_hours}. "
            "The fix should use SleepAggregator, not sleep_percentage * 24!"
        )
        
        print(f"✅ SUCCESS: Sleep duration correctly calculated as {features.seoul_features.sleep_duration_hours} hours")
    
    def test_aggregation_pipeline_handles_no_sleep_data(self):
        """Test that the pipeline handles days with no sleep data."""
        pipeline = AggregationPipeline()
        
        # Empty sleep records
        sleep_records = []
        
        # Create test data
        current_date = date(2024, 1, 1)
        daily_metrics = {
            "sleep": {
                "sleep_percentage": 0.0,
                "sleep_amplitude": 0.0,
                "long_num": 0,
                "long_len": 0.0,
                "long_st": 0.0,
                "long_wt": 0.0,
                "short_num": 0,
                "short_len": 0.0,
                "short_st": 0.0,
                "short_wt": 0.0,
            }
        }
        
        features = pipeline._calculate_features_with_stats(
            current_date,
            daily_metrics,
            [daily_metrics["sleep"]],
            [],
            {"daily_steps": 0, "activity_variance": 0, "sedentary_hours": 24,
             "activity_fragmentation": 0, "sedentary_bout_mean": 24, "activity_intensity_ratio": 0},
            sleep_records,
        )
        
        assert features is not None
        assert features.seoul_features.sleep_duration_hours == 0.0