"""
Test aggregation pipeline sleep duration calculation.

MISSION: Prove the bug exists, then fix it like a pro!
"""
import logging
from datetime import date, datetime

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState

# Enable debug logging to see sleep assignment
logging.basicConfig(level=logging.DEBUG)


class TestAggregationPipelineSleep:
    """Test that the pipeline calculates sleep duration correctly."""

    def test_single_7_5_hour_sleep_record(self):
        """
        Test: Pipeline should report 7.5 hours when sleep ends before 3 PM.

        NOTE: Apple Health assigns sleep to the date the user wakes up,
        unless they wake up after 3 PM (then it's assigned to next day).
        """
        # Create 3 days of data so the pipeline has enough window data
        sleep_records = []
        activity_records = []

        for day_offset in range(3):
            # Create 7.5 hour sleep record for each night
            sleep_start = datetime(2024, 1, 1 + day_offset, 22, 30)  # 10:30 PM
            sleep_end = datetime(2024, 1, 2 + day_offset, 6, 0)   # 6:00 AM next day

            sleep_records.append(SleepRecord(
                source_name="Apple Watch",
                start_date=sleep_start,
                end_date=sleep_end,
                state=SleepState.ASLEEP
            ))

            # Create minimal activity data
            activity_records.append(ActivityRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 2 + day_offset, 12, 0),
                end_date=datetime(2024, 1, 2 + day_offset, 12, 1),
                activity_type=ActivityType.STEP_COUNT,
                value=100 + day_offset * 50,
                unit="steps"
            ))

        # Create pipeline
        pipeline = AggregationPipeline()

        # Process the data - check the middle day (Jan 3)
        # which should have enough window data
        features = pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4),
            min_window_size=2  # Need at least 2 days for window
        )

        # ASSERTION: Should get 7.5 hours, not ~2 hours!
        print(f"\n🔍 Got {len(features)} features")
        for f in features:
            print(f"  - Date: {f.date}, Has Seoul features: {f.seoul_features is not None}")
            if f.seoul_features:
                print(f"    Sleep duration: {f.seoul_features.sleep_duration_hours:.1f} hours")

        assert len(features) >= 1
        # Check the middle day which should have proper sleep data
        daily_feature = next((f for f in features if f.date == date(2024, 1, 3)), None)
        assert daily_feature is not None, "Should have features for Jan 3"

        # This is the REAL test - it will fail with current code
        assert daily_feature.seoul_features is not None
        actual_sleep_hours = daily_feature.seoul_features.sleep_duration_hours

        print("\n🔴 EXPECTED: 7.5 hours")
        print(f"🔴 ACTUAL: {actual_sleep_hours:.1f} hours")

        # Let's also check what the raw metrics show
        if hasattr(daily_feature, 'raw_features'):
            print(f"🔍 Raw features: {daily_feature.raw_features}")

        # Check the Seoul features in detail
        seoul = daily_feature.seoul_features
        print(f"🔍 Sleep efficiency: {seoul.sleep_efficiency}")
        print(f"🔍 Sleep fragmentation: {seoul.sleep_fragmentation}")
        print(f"🔍 Short sleep window %: {seoul.short_sleep_window_pct}")
        print(f"🔍 Long sleep window %: {seoul.long_sleep_window_pct}")

        # Check that sleep duration is correctly calculated as ~7.5 hours
        assert 7.0 <= actual_sleep_hours <= 8.0, (
            f"Sleep duration should be ~7.5 hours, not {actual_sleep_hours:.1f}!"
        )

        print(f"\n✅ SUCCESS: Sleep duration correctly calculated as {actual_sleep_hours:.1f} hours")

    def test_multiple_sleep_fragments(self):
        """Test that fragmented sleep is summed correctly."""
        # Create multiple days of data for proper windowing
        sleep_records = []
        activity_records = []

        for day_offset in range(3):
            # Night sleep: 22:00 → 02:00 (4 hours)
            sleep1 = SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 1 + day_offset, 22, 0),
                end_date=datetime(2024, 1, 2 + day_offset, 2, 0),
                state=SleepState.ASLEEP
            )

            # Back to sleep: 03:00 → 06:30 (3.5 hours)
            sleep2 = SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 2 + day_offset, 3, 0),
                end_date=datetime(2024, 1, 2 + day_offset, 6, 30),
                state=SleepState.ASLEEP
            )

            sleep_records.extend([sleep1, sleep2])

            # Add activity data for each day
            activity_records.append(ActivityRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 2 + day_offset, 12, 0),
                end_date=datetime(2024, 1, 2 + day_offset, 12, 1),
                activity_type=ActivityType.STEP_COUNT,
                value=100,
                unit="steps"
            ))

        # Total should be 4 + 3.5 = 7.5 hours per night

        pipeline = AggregationPipeline()
        features = pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4),
            min_window_size=2
        )

        # Check the middle day
        assert len(features) >= 1
        daily_feature = next((f for f in features if f.date == date(2024, 1, 3)), None)
        assert daily_feature is not None
        actual_sleep = daily_feature.seoul_features.sleep_duration_hours

        print("\n🔴 Fragmented sleep test:")
        print("   Fragment 1: 4.0 hours")
        print("   Fragment 2: 3.5 hours")
        print("   EXPECTED TOTAL: 7.5 hours")
        print(f"   ACTUAL TOTAL: {actual_sleep:.1f} hours")

        assert 7.0 <= actual_sleep <= 8.0, (
            f"Fragmented sleep should sum to ~7.5 hours, not {actual_sleep:.1f}!"
        )
