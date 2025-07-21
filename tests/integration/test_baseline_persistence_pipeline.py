"""
Integration test for baseline persistence in the mood prediction pipeline.

This is the real deal - testing that personal baselines actually persist
and improve predictions over time. This is what makes our system PERSONAL!
"""

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord, HeartMetricType, MotionContext
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


class TestBaselinePersistencePipeline:
    """Test that baselines persist and improve predictions over time."""

    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create a test data directory."""
        data_dir = tmp_path / "test_baseline_persistence"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @pytest.fixture
    def baseline_repository(self, test_data_dir):
        """Create a file baseline repository."""
        baselines_dir = test_data_dir / "baselines"
        return FileBaselineRepository(baselines_dir)

    def generate_realistic_data(self, base_date: date, days: int, user_pattern: dict):
        """Generate realistic health data with user-specific patterns."""
        sleep_records = []
        activity_records = []
        heart_rate_records = []

        for day_offset in range(days):
            current_date = base_date + timedelta(days=day_offset)

            # Sleep with personal variation
            sleep_duration = np.random.normal(
                user_pattern["sleep_mean"], user_pattern["sleep_std"]
            )
            sleep_duration = max(
                4.0, min(12.0, sleep_duration)
            )  # Clamp to realistic range

            sleep_records.append(
                SleepRecord(
                    source_name="test",
                    start_date=datetime.combine(
                        current_date - timedelta(days=1), datetime.min.time()
                    )
                    + timedelta(hours=22),
                    end_date=datetime.combine(current_date, datetime.min.time())
                    + timedelta(hours=22 + sleep_duration),
                    state=SleepState.ASLEEP,
                )
            )

            # Activity with personal variation
            daily_steps = np.random.normal(
                user_pattern["activity_mean"], user_pattern["activity_std"]
            )
            daily_steps = max(1000, int(daily_steps))

            # Distribute activity throughout the day
            for hour in [9, 12, 15, 18]:
                activity_records.append(
                    ActivityRecord(
                        source_name="test",
                        start_date=datetime.combine(current_date, datetime.min.time())
                        + timedelta(hours=hour),
                        end_date=datetime.combine(current_date, datetime.min.time())
                        + timedelta(hours=hour, minutes=30),
                        activity_type=ActivityType.STEP_COUNT,
                        value=daily_steps / 4,  # distribute steps across 4 activity periods
                        unit="count",
                    )
                )

            # Heart rate throughout the day
            for hour in range(0, 24, 2):
                # Circadian rhythm simulation
                if 6 <= hour <= 22:  # Awake hours
                    hr = user_pattern["hr_mean"] + 10 * np.sin((hour - 6) * np.pi / 16)
                else:  # Sleep hours
                    hr = user_pattern["hr_rest"]

                hr += np.random.normal(0, user_pattern["hr_std"])

                heart_rate_records.append(
                    HeartRateRecord(
                        source_name="test",
                        timestamp=datetime.combine(current_date, datetime.min.time())
                        + timedelta(hours=hour),
                        metric_type=HeartMetricType.HEART_RATE,
                        value=int(hr),
                        unit="count/min",
                        motion_context=MotionContext.SEDENTARY if hour < 6 or hour > 22 else MotionContext.ACTIVE,
                    )
                )

        return sleep_records, activity_records, heart_rate_records

    @pytest.mark.xfail(
        reason="Test uses outdated domain entity APIs - needs rewrite for current architecture",
        strict=True
    )
    def test_baseline_persistence_improves_predictions(self, baseline_repository):
        """
        EPIC TEST: Prove that baselines persist and predictions improve!

        This test simulates a user tracking their data over 3 weeks:
        - Week 1: Establish initial baselines
        - Week 2: Baselines should be more accurate
        - Week 3: Predictions should be most accurate
        """
        # User's true personal patterns
        user_pattern = {
            "sleep_mean": 7.2,  # This user sleeps less than average
            "sleep_std": 0.8,
            "activity_mean": 12000,  # Very active user
            "activity_std": 3000,
            "hr_mean": 68,  # Lower resting HR (fit)
            "hr_rest": 58,
            "hr_active": 95,
            "hr_std": 5,
            "hrv_mean": 55,  # Good HRV
            "hrv_std": 10,
        }

        # Create pipeline with personal calibration
        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "test_athlete_123"
        config.min_days_required = 3

        pipeline = MoodPredictionPipeline(
            config=config, baseline_repository=baseline_repository
        )

        # Week 1: Process initial data
        week1_sleep, week1_activity, week1_hr = self.generate_realistic_data(
            date(2024, 1, 1), 7, user_pattern
        )

        features_week1 = []
        for day in range(7):
            result = pipeline.process_health_data(
                sleep_records=week1_sleep[day : day + 1],
                activity_records=[
                    a for a in week1_activity if a.date == week1_sleep[day].date
                ],
                heart_records=[
                    h for h in week1_hr if h.timestamp.date() == week1_sleep[day].date
                ],
                target_date=week1_sleep[day].date,
            )
            if result:  # After min_days_required
                features_week1.append(result)

        # Check baseline was created
        baseline = baseline_repository.get_baseline("test_athlete_123")
        assert baseline is not None, "Baseline should be created after week 1"

        # Baseline should reflect user's patterns (not population average)
        assert (
            6.5 < baseline.sleep_mean < 7.5
        ), f"Sleep baseline {baseline.sleep_mean} should be ~7.2"
        assert (
            10000 < baseline.activity_mean < 14000
        ), f"Activity baseline {baseline.activity_mean} should be ~12000"

        # REGRESSION TEST: Ensure sleep calculations are reasonable
        # This guards against the sleep_percentage * 24 bug
        for feature in features_week1:
            if hasattr(feature, "seoul_features"):
                sleep_hours = feature.seoul_features.sleep_duration_hours
                assert 4.0 <= sleep_hours <= 12.0, (
                    f"Sleep duration {sleep_hours}h is outside reasonable range [4,12]. "
                    "This may indicate the sleep_percentage * 24 bug has returned!"
                )

        # Week 2: Process more data - baselines should improve
        week2_sleep, week2_activity, week2_hr = self.generate_realistic_data(
            date(2024, 1, 8), 7, user_pattern
        )

        features_week2 = []
        for day in range(7):
            result = pipeline.process_health_data(
                sleep_records=week2_sleep[day : day + 1],
                activity_records=[
                    a for a in week2_activity if a.date == week2_sleep[day].date
                ],
                heart_records=[
                    h for h in week2_hr if h.timestamp.date() == week2_sleep[day].date
                ],
                target_date=week2_sleep[day].date,
            )
            if result:
                features_week2.append(result)

        # Check baseline improved
        baseline_week2 = baseline_repository.get_baseline("test_athlete_123")
        assert baseline_week2 is not None
        assert (
            baseline_week2.data_points > baseline.data_points
        ), "More data points after week 2"

        # REGRESSION TEST: Week 2 sleep calculations
        for feature in features_week2:
            if hasattr(feature, "seoul_features"):
                sleep_hours = feature.seoul_features.sleep_duration_hours
                assert (
                    4.0 <= sleep_hours <= 12.0
                ), f"Week 2: Sleep duration {sleep_hours}h is outside reasonable range"

        # Week 3: Final week - predictions should be most personalized
        week3_sleep, week3_activity, week3_hr = self.generate_realistic_data(
            date(2024, 1, 15), 7, user_pattern
        )

        features_week3 = []
        for day in range(7):
            result = pipeline.process_health_data(
                sleep_records=week3_sleep[day : day + 1],
                activity_records=[
                    a for a in week3_activity if a.date == week3_sleep[day].date
                ],
                heart_records=[
                    h for h in week3_hr if h.timestamp.date() == week3_sleep[day].date
                ],
                target_date=week3_sleep[day].date,
            )
            if result:
                features_week3.append(result)

        # Final baseline should be most accurate
        final_baseline = baseline_repository.get_baseline("test_athlete_123")
        assert final_baseline.data_points >= 14, "Should have at least 2 weeks of data"

        # Verify baseline history shows improvement over time
        history = baseline_repository.get_baseline_history("test_athlete_123")
        assert len(history) >= 2, "Should have baseline history"

        # This is what makes it PERSONAL - baselines converge to user's true patterns!
        print("\n🎯 PERSONAL CALIBRATION SUCCESS!")
        print(f"Initial sleep baseline: {baseline.sleep_mean:.2f} hours")
        print(
            f"Final sleep baseline: {final_baseline.sleep_mean:.2f} hours (true: {user_pattern['sleep_mean']})"
        )
        print(f"Initial activity baseline: {baseline.activity_mean:.0f} steps")
        print(
            f"Final activity baseline: {final_baseline.activity_mean:.0f} steps (true: {user_pattern['activity_mean']})"
        )

    @pytest.mark.xfail(
        reason="Test uses outdated domain entity APIs - needs rewrite for current architecture",
        strict=True
    )
    def test_baseline_persistence_after_pipeline_restart(self, baseline_repository):
        """Test that baselines persist when pipeline is restarted."""
        # Create first pipeline instance
        config = PipelineConfig()
        config.enable_personal_calibration = True
        config.user_id = "restart_test_user"
        config.min_days_required = 1

        pipeline1 = MoodPredictionPipeline(
            config=config, baseline_repository=baseline_repository
        )

        # Process some data
        sleep = SleepRecord(
            date=date(2024, 1, 1),
            sleep_start=datetime(2023, 12, 31, 22, 0),
            sleep_end=datetime(2024, 1, 1, 6, 0),
            sleep_duration_hours=8.0,
            sleep_efficiency=0.85,
            awake_count=2,
            restless_count=5,
            quality_score=0.85,
        )

        activity = ActivityRecord(
            date=date(2024, 1, 1),
            timestamp=datetime(2024, 1, 1, 14, 0),
            activity_type="Walking",
            duration_minutes=30.0,
            calories=150.0,
            distance_km=2.5,
            heart_rate_avg=95.0,
        )

        hr = HeartRateRecord(
            timestamp=datetime(2024, 1, 1, 14, 0),
            heart_rate=75,
            heart_rate_variability=45.0,
            motion_context="resting",
        )

        # Process day 1
        pipeline1.process_health_data(
            sleep_records=[sleep],
            activity_records=[activity],
            heart_records=[hr],
            target_date=date(2024, 1, 1),
        )

        # Verify baseline exists
        baseline1 = baseline_repository.get_baseline("restart_test_user")
        assert baseline1 is not None

        # Simulate pipeline restart - create new instance
        pipeline2 = MoodPredictionPipeline(
            config=config, baseline_repository=baseline_repository
        )

        # Process day 2 with new pipeline instance
        sleep2 = SleepRecord(
            date=date(2024, 1, 2),
            sleep_start=datetime(2024, 1, 1, 22, 30),
            sleep_end=datetime(2024, 1, 2, 6, 30),
            sleep_duration_hours=8.0,
            sleep_efficiency=0.87,
            awake_count=1,
            restless_count=4,
            quality_score=0.87,
        )

        pipeline2.process_health_data(
            sleep_records=[sleep2],
            activity_records=[activity],  # Reuse for simplicity
            heart_records=[hr],
            target_date=date(2024, 1, 2),
        )

        # Verify baseline was loaded and updated
        baseline2 = baseline_repository.get_baseline("restart_test_user")
        assert baseline2 is not None
        assert (
            baseline2.data_points > baseline1.data_points
        ), "Baseline should accumulate data across restarts"

        print("\n✅ BASELINE PERSISTENCE WORKS!")
        print("Baseline survives pipeline restart - true personal calibration!")
