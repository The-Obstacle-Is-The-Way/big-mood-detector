"""
Debug the sleep duration math - BILLION DOLLAR MATH OLYMPIAD!

Let's trace every single number through the system.
"""

from datetime import date, datetime, timedelta

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.repositories.baseline_repository_interface import (
    UserBaseline,
)
from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
)
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


class TestSleepMathDebug:
    """TRACE THE MATH BRO! Every number matters!"""

    def test_simple_7_5_hour_sleep(self):
        """Test case: Jane sleeps 22:00 → 05:30 = 7.5 hours EXACTLY"""
        print("\n🧮 MATH OLYMPIAD: Testing 7.5 hour sleep")

        # Create ONE sleep record: 22:00 → 05:30
        start = datetime(2024, 1, 1, 22, 0)  # 10 PM
        end = datetime(2024, 1, 2, 5, 30)  # 5:30 AM next day

        sleep_record = SleepRecord(
            source_name="Apple Watch",
            start_date=start,
            end_date=end,
            state=SleepState.ASLEEP,
        )

        # MATH CHECK 1: Raw duration
        duration = sleep_record.duration_hours
        print(f"✅ MATH 1: SleepRecord.duration_hours = {duration}")
        assert duration == 7.5, f"Expected 7.5, got {duration}"

        # MATH CHECK 2: Aggregator
        aggregator = SleepAggregator()
        summaries = aggregator.aggregate_daily([sleep_record])

        assert len(summaries) == 1, f"Expected 1 summary, got {len(summaries)}"

        # Get the first (and only) summary
        summary_date = list(summaries.keys())[0]
        summary = summaries[summary_date]

        print(f"✅ MATH 2: DailySleepSummary.date = {summary.date}")
        print(
            f"✅ MATH 2: DailySleepSummary.total_sleep_hours = {summary.total_sleep_hours}"
        )
        assert (
            summary.total_sleep_hours == 7.5
        ), f"Expected 7.5, got {summary.total_sleep_hours}"

    def test_midnight_cross_assignment(self):
        """Test which day gets the sleep when crossing midnight"""
        print("\n🧮 MATH OLYMPIAD: Testing midnight cross date assignment")

        # Case 1: Sleep 22:00 → 05:30 (should be assigned to Jan 1)
        start1 = datetime(2024, 1, 1, 22, 0)
        end1 = datetime(2024, 1, 2, 5, 30)

        sleep1 = SleepRecord(
            source_name="Apple Watch",
            start_date=start1,
            end_date=end1,
            state=SleepState.ASLEEP,
        )

        aggregator = SleepAggregator()
        summaries1 = aggregator.aggregate_daily([sleep1])

        # Get the summary
        summary_date = list(summaries1.keys())[0]
        summary = summaries1[summary_date]

        print(f"Sleep from {start1} to {end1}")
        print(f"Assigned to date: {summary.date}")
        print(f"Duration: {summary.total_sleep_hours} hours")

        # The sleep should be assigned to Jan 2 (due to 3pm rule)
        assert summary.date == date(2024, 1, 2), f"Expected Jan 2, got {summary.date}"

    def test_multiple_days_aggregation(self):
        """Test 3 days of 7.5 hour sleep each"""
        print("\n🧮 MATH OLYMPIAD: Testing 3 days aggregation")

        records = []
        for day in range(3):
            start = datetime(2024, 1, day + 1, 22, 0)
            end = start + timedelta(hours=7.5)

            records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=start,
                    end_date=end,
                    state=SleepState.ASLEEP,
                )
            )

        aggregator = SleepAggregator()
        summaries = aggregator.aggregate_daily(records)

        print(f"\nCreated {len(records)} sleep records")
        print(f"Got {len(summaries)} daily summaries")

        for i, (summary_date, summary) in enumerate(summaries.items()):
            print(f"Day {i+1}: {summary_date} = {summary.total_sleep_hours} hours")
            assert (
                summary.total_sleep_hours == 7.5
            ), f"Day {i+1} wrong: {summary.total_sleep_hours}"

    def test_baseline_calculation_simple(self, tmp_path):
        """Test baseline calculation with known values"""
        print("\n🧮 MATH OLYMPIAD: Testing baseline calculation")

        # Create a baseline repository
        repo = FileBaselineRepository(tmp_path / "baselines")

        # Create feature engineer with baseline tracking
        feature_eng = AdvancedFeatureEngineer(
            baseline_repository=repo, user_id="test_jane"
        )

        # Simulate 3 days of sleep data
        sleep_values = [7.5, 7.0, 8.0]  # Average should be 7.5

        for i, hours in enumerate(sleep_values):
            print(f"\nDay {i+1}: Updating baseline with {hours} hours")
            feature_eng._update_individual_baseline("sleep", hours)

            baseline = feature_eng.individual_baselines["sleep"]
            print(f"  Count: {baseline.get('count', 0)}")
            print(f"  Sum: {baseline.get('sum', 0)}")
            print(f"  Mean: {baseline.get('mean', 0)}")
            print(f"  Std: {baseline.get('std', 0)}")

        # Check final mean
        final_mean = feature_eng.individual_baselines["sleep"]["mean"]
        expected_mean = sum(sleep_values) / len(sleep_values)

        print(f"\n✅ FINAL CHECK: Mean = {final_mean}, Expected = {expected_mean}")
        assert (
            abs(final_mean - expected_mean) < 0.01
        ), f"Mean wrong: {final_mean} != {expected_mean}"

    def test_persist_and_reload_baseline(self, tmp_path):
        """Test that baselines persist correctly"""
        print("\n🧮 MATH OLYMPIAD: Testing baseline persistence")

        repo = FileBaselineRepository(tmp_path / "baselines")

        # Create and save a baseline
        baseline = UserBaseline(
            user_id="test_jane",
            baseline_date=date.today(),
            sleep_mean=7.5,
            sleep_std=0.5,
            activity_mean=15000.0,
            activity_std=2000.0,
            circadian_phase=0.0,
            data_points=3,
        )

        print(f"Saving baseline with sleep_mean={baseline.sleep_mean}")
        repo.save_baseline(baseline)

        # Reload it
        loaded = repo.get_baseline("test_jane")
        print(f"Loaded baseline with sleep_mean={loaded.sleep_mean}")

        assert loaded.sleep_mean == 7.5, f"Persistence failed: {loaded.sleep_mean}"
        assert loaded.data_points == 3, f"Data points wrong: {loaded.data_points}"
