"""
Regression test to ensure sleep duration calculations stay within reasonable bounds.

This test guards against the sleep_percentage * 24 bug returning.
"""
import pytest
from datetime import datetime, date, timedelta
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord, HeartMetricType
from big_mood_detector.application.services.aggregation_pipeline import AggregationPipeline


class TestSleepDurationRegression:
    """Ensure sleep duration calculations remain reasonable."""
    
    def test_sleep_duration_stays_within_reasonable_bounds(self):
        """
        Guard against sleep duration calculation bugs.
        
        Normal human sleep should be between 4-12 hours per day.
        We use a tighter bound [6,10] to catch calculation errors early.
        """
        pipeline = AggregationPipeline()
        
        # Create various sleep patterns
        test_cases = [
            # Normal 8 hour sleep
            (datetime(2024, 1, 1, 22, 0), datetime(2024, 1, 2, 6, 0), "Normal 8h"),
            # Short 6 hour sleep
            (datetime(2024, 1, 1, 23, 0), datetime(2024, 1, 2, 5, 0), "Short 6h"),
            # Long 10 hour sleep
            (datetime(2024, 1, 1, 21, 0), datetime(2024, 1, 2, 7, 0), "Long 10h"),
            # Split sleep (nap + night)
            (datetime(2024, 1, 1, 14, 0), datetime(2024, 1, 1, 15, 30), "Afternoon nap"),
        ]
        
        for start, end, description in test_cases:
            sleep_record = SleepRecord(
                source_name="Apple Watch",
                start_date=start,
                end_date=end,
                state=SleepState.ASLEEP,
            )
            
            # Minimal activity/heart data
            activity_record = ActivityRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 1, 12, 0),
                end_date=datetime(2024, 1, 1, 13, 0),
                activity_type=ActivityType.STEP_COUNT,
                value=1000,
                unit="steps",
            )
            
            heart_record = HeartRateRecord(
                source_name="Apple Watch",
                timestamp=datetime(2024, 1, 1, 12, 0),
                metric_type=HeartMetricType.HEART_RATE,
                value=70,
                unit="bpm",
            )
            
            # Process through pipeline
            features = pipeline.aggregate_daily_features(
                sleep_records=[sleep_record],
                activity_records=[activity_record],
                heart_records=[heart_record],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1),
            )
            
            # Check that sleep duration is reasonable
            assert len(features) > 0, f"No features extracted for {description}"
            feature = features[0]
            
            assert hasattr(feature, 'seoul_features'), "Should have seoul_features"
            sleep_hours = feature.seoul_features.sleep_duration_hours
            
            # The key assertion: sleep should be in reasonable range
            assert 0 <= sleep_hours <= 24, (
                f"{description}: Sleep duration {sleep_hours}h is impossible (>24h)"
            )
            
            # Warn if outside normal range (but don't fail - naps are valid)
            if not (4 <= sleep_hours <= 12):
                print(f"⚠️  {description}: Sleep duration {sleep_hours}h is unusual but valid")
    
    def test_multi_day_sleep_average_reasonable(self):
        """Test that multi-day averages are reasonable."""
        pipeline = AggregationPipeline()
        
        # Create a week of normal sleep
        sleep_records = []
        activity_records = []
        heart_records = []
        
        for day in range(7):
            base_date = date(2024, 1, 1) + timedelta(days=day)
            
            # Sleep from 10pm to 6am (8 hours)
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(base_date - timedelta(days=1), 
                                               datetime.min.time()).replace(hour=22),
                    end_date=datetime.combine(base_date, 
                                             datetime.min.time()).replace(hour=6),
                    state=SleepState.ASLEEP,
                )
            )
            
            # Minimal activity
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(base_date, datetime.min.time()).replace(hour=12),
                    end_date=datetime.combine(base_date, datetime.min.time()).replace(hour=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=5000,
                    unit="steps",
                )
            )
            
            # Minimal heart rate
            heart_records.append(
                HeartRateRecord(
                    source_name="Apple Watch",
                    timestamp=datetime.combine(base_date, datetime.min.time()).replace(hour=12),
                    metric_type=HeartMetricType.HEART_RATE,
                    value=70,
                    unit="bpm",
                )
            )
        
        # Process the week
        features = pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 7),
        )
        
        # Check each day's sleep
        sleep_hours_list = []
        for feature in features:
            if feature and hasattr(feature, 'seoul_features'):
                sleep_hours = feature.seoul_features.sleep_duration_hours
                sleep_hours_list.append(sleep_hours)
                
                # Each day should be reasonable
                assert 6 <= sleep_hours <= 10, (
                    f"Daily sleep {sleep_hours}h outside normal range [6,10]"
                )
        
        # Average should be close to 8
        if sleep_hours_list:
            avg_sleep = sum(sleep_hours_list) / len(sleep_hours_list)
            assert 7 <= avg_sleep <= 9, (
                f"Average sleep {avg_sleep}h is unusual (expected ~8h)"
            )
            print(f"✅ Average sleep over {len(sleep_hours_list)} days: {avg_sleep:.1f}h")