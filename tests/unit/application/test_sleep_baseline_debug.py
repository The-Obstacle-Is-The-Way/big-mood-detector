"""
Debug test to trace exactly where sleep hours are getting lost.
"""
from datetime import date, datetime, timedelta

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


class TestSleepBaselineDebug:
    """Debug the sleep baseline calculation."""

    def test_trace_sleep_baseline_calculation(self, tmp_path):
        """Trace exactly where sleep hours are lost."""
        # Create a single day of sleep data: 7.5 hours
        sleep_record = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 22, 0),  # 10pm
            end_date=datetime(2024, 1, 2, 5, 30),   # 5:30am next day
            state=SleepState.ASLEEP,
        )

        print("\n🔍 CREATED SLEEP RECORD:")
        print(f"   Start: {sleep_record.start_date}")
        print(f"   End: {sleep_record.end_date}")
        print(f"   Duration: {sleep_record.duration_hours:.1f} hours")

        # Verify with SleepAggregator
        aggregator = SleepAggregator()
        summaries = aggregator.aggregate_daily([sleep_record])
        summary = summaries[date(2024, 1, 1)]

        print("\n✅ SLEEP AGGREGATOR SAYS:")
        print(f"   Total sleep: {summary.total_sleep_hours:.1f} hours")

        # Create minimal activity and heart data
        activity_record = ActivityRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 12, 0),
            end_date=datetime(2024, 1, 1, 13, 0),
            activity_type=ActivityType.STEP_COUNT,
            value=5000,
            unit="steps",
        )

        heart_record = HeartRateRecord(
            source_name="Apple Watch",
            timestamp=datetime(2024, 1, 1, 12, 0),
            metric_type=HeartMetricType.HEART_RATE,
            value=60,
            unit="bpm",
        )

        # Create pipeline with baseline repository
        baseline_repo = FileBaselineRepository(tmp_path / "baselines")

        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "debug_user"
        config.min_days_required = 1

        pipeline = MoodPredictionPipeline(
            config=config,
            baseline_repository=baseline_repo
        )

        # Process the data
        print("\n⚙️  PROCESSING THROUGH PIPELINE...")
        result = pipeline.process_health_data(
            sleep_records=[sleep_record],
            activity_records=[activity_record],
            heart_records=[heart_record],
            target_date=date(2024, 1, 1),
        )

        # Check the baseline
        baseline = baseline_repo.get_baseline("debug_user")

        if baseline:
            print("\n📊 BASELINE RESULT:")
            print(f"   Sleep mean: {baseline.sleep_mean:.1f} hours")
            print(f"   Data points: {baseline.data_points}")

            if abs(baseline.sleep_mean - 7.5) > 0.1:
                print(f"\n❌ BUG CONFIRMED: Baseline should be 7.5h, not {baseline.sleep_mean:.1f}h!")
            else:
                print("\n✅ BASELINE CORRECT!")
        else:
            print("\n❌ NO BASELINE CREATED!")

        # Check the features
        if result and result.daily_features:
            features = result.daily_features.get(date(2024, 1, 1))
            if features and features.seoul_features:
                print("\n🔍 SEOUL FEATURES:")
                print(f"   sleep_duration_hours: {features.seoul_features.sleep_duration_hours:.1f}")

    def test_trace_multiple_days(self, tmp_path):
        """Test with multiple days to see accumulation."""
        sleep_records = []
        activity_records = []
        heart_records = []

        # Create 3 days of 7.5h sleep
        for day_offset in range(3):
            base_date = date(2024, 1, 1) + timedelta(days=day_offset)

            # Sleep from 10pm to 5:30am
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(base_date - timedelta(days=1), datetime.min.time()).replace(hour=22),
                    end_date=datetime.combine(base_date, datetime.min.time()).replace(hour=5, minute=30),
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
                    value=10000,
                    unit="steps",
                )
            )

            # Minimal heart rate
            heart_records.append(
                HeartRateRecord(
                    source_name="Apple Watch",
                    timestamp=datetime.combine(base_date, datetime.min.time()).replace(hour=12),
                    metric_type=HeartMetricType.HEART_RATE,
                    value=60,
                    unit="bpm",
                )
            )

        # Process
        baseline_repo = FileBaselineRepository(tmp_path / "baselines")

        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "multi_day_user"
        config.min_days_required = 1

        pipeline = MoodPredictionPipeline(
            config=config,
            baseline_repository=baseline_repo
        )

        pipeline.process_health_data(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=date(2024, 1, 3),
        )

        # Check baseline
        baseline = baseline_repo.get_baseline("multi_day_user")

        if baseline:
            print("\n📊 3-DAY BASELINE:")
            print(f"   Sleep mean: {baseline.sleep_mean:.1f} hours")
            print(f"   Data points: {baseline.data_points}")

            # With 3 days at 7.5h each, mean should be 7.5h
            # But if persist_baselines is called 3x per day, we'd have 9 data points!
            if baseline.data_points != 3:
                print(f"\n⚠️  WARNING: Expected 3 data points, got {baseline.data_points}!")
                print("   This suggests persist_baselines called multiple times per day!")
