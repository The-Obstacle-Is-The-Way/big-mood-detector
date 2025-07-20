"""
Test aggregation pipeline sleep duration calculation.

MISSION: Prove the bug exists, then fix it like a pro!
"""
from datetime import date, datetime

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


class TestAggregationPipelineSleep:
    """Test that the pipeline calculates sleep duration correctly."""

    def test_single_7_5_hour_sleep_record(self):
        """
        RED TEST: Pipeline should report 7.5 hours, not 2.5!

        This test will FAIL until we fix the bug.
        """
        # Create a single 7.5 hour sleep record
        sleep_start = datetime(2024, 1, 1, 22, 0)  # 10 PM
        sleep_end = datetime(2024, 1, 2, 5, 30)   # 5:30 AM next day

        sleep_record = SleepRecord(
            source_name="Apple Watch",
            start_date=sleep_start,
            end_date=sleep_end,
            state=SleepState.ASLEEP
        )

        # Create minimal activity data (required by pipeline)
        activity_record = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 12, 0),
            end_date=datetime(2024, 1, 1, 12, 1),
            activity_type=ActivityType.STEP_COUNT,
            value=100,
            unit="steps"
        )

        # Create pipeline
        pipeline = AggregationPipeline()

        # Process the data - override min_window_size to allow single day
        features = pipeline.aggregate_daily_features(
            sleep_records=[sleep_record],
            activity_records=[activity_record],
            heart_records=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            min_window_size=1  # Allow single day for testing
        )

        # ASSERTION: Should get 7.5 hours, not ~2 hours!
        assert len(features) == 1
        daily_feature = features[0]

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

        # The bug: This assertion will FAIL because pipeline calculates wrong!
        assert 7.0 <= actual_sleep_hours <= 8.0, (
            f"Sleep duration should be ~7.5 hours, not {actual_sleep_hours:.1f}!"
        )

    def test_multiple_sleep_fragments(self):
        """Test that fragmented sleep is summed correctly."""
        # Night sleep: 22:00 → 02:00 (4 hours)
        sleep1 = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 22, 0),
            end_date=datetime(2024, 1, 2, 2, 0),
            state=SleepState.ASLEEP
        )

        # Wake period: 02:00 → 03:00 (bathroom break)

        # Back to sleep: 03:00 → 06:30 (3.5 hours)
        sleep2 = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 3, 0),
            end_date=datetime(2024, 1, 2, 6, 30),
            state=SleepState.ASLEEP
        )

        # Total should be 4 + 3.5 = 7.5 hours

        pipeline = AggregationPipeline()
        features = pipeline.aggregate_daily_features(
            sleep_records=[sleep1, sleep2],
            activity_records=[],
            heart_records=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
            min_window_size=1  # Allow single day for testing
        )

        assert len(features) == 1
        actual_sleep = features[0].seoul_features.sleep_duration_hours

        print("\n🔴 Fragmented sleep test:")
        print("   Fragment 1: 4.0 hours")
        print("   Fragment 2: 3.5 hours")
        print("   EXPECTED TOTAL: 7.5 hours")
        print(f"   ACTUAL TOTAL: {actual_sleep:.1f} hours")

        assert 7.0 <= actual_sleep <= 8.0, (
            f"Fragmented sleep should sum to ~7.5 hours, not {actual_sleep:.1f}!"
        )
