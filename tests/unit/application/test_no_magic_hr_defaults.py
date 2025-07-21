"""
Test that magic HR/HRV defaults (70 bpm, 50 ms) have been removed.

This guards against hardcoded values that would incorrectly skew
personal baselines for athletes or individuals with different physiology.
"""

import tempfile
from datetime import date, datetime
from pathlib import Path

import pytest

class TestNoMagicHRDefaults:
    """Test that HR/HRV values are not defaulted to magic numbers."""

    def test_user_baseline_no_hr_defaults(self):
        """Test UserBaseline doesn't have magic HR/HRV defaults."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        baseline = UserBaseline(
            user_id="test",
            baseline_date=date.today(),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=8000,
            activity_std=2000,
            circadian_phase=22.0,
        )

        # HR/HRV should be None by default, not magic numbers
        assert baseline.heart_rate_mean is None
        assert baseline.heart_rate_std is None
        assert baseline.hrv_mean is None
        assert baseline.hrv_std is None

    def test_file_repository_preserves_none_values(self):
        """Test FileBaselineRepository doesn't add magic defaults when loading."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline
        from big_mood_detector.infrastructure.repositories.file_baseline_repository import FileBaselineRepository

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = FileBaselineRepository(Path(tmpdir))

            # Save baseline without HR/HRV data
            baseline = UserBaseline(
                user_id="athlete",
                baseline_date=date.today(),
                sleep_mean=6.5,  # Athletes often sleep less
                sleep_std=0.5,
                activity_mean=15000,  # Very active
                activity_std=3000,
                circadian_phase=21.0,
                # No HR/HRV data - should remain None
            )

            repo.save_baseline(baseline)

            # Load it back
            loaded = repo.get_baseline("athlete")
            assert loaded is not None

            # Should still be None, not defaulted to 70/50
            assert loaded.heart_rate_mean is None
            assert loaded.hrv_mean is None

    def test_feature_engineering_no_magic_defaults(self):
        """Test AdvancedFeatureEngineer doesn't use magic HR values."""
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        engineer = AdvancedFeatureEngineer(user_id="test_user")

        # Should not update baselines with magic values
        # The old code had checks for != 70.0 and != 50.0
        # Now it should check for > 0

        # Check the baseline is empty initially
        assert "hr" not in engineer.individual_baselines
        assert "hrv" not in engineer.individual_baselines

    def test_aggregation_pipeline_no_hr_defaults(self):
        """Test aggregation pipeline doesn't insert magic HR values."""
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )
        from big_mood_detector.application.services.aggregation_pipeline import AggregationPipeline

        pipeline = AggregationPipeline()

        # Create minimal test data WITHOUT heart rate records
        sleep_records = [
            SleepRecord(
                source_name="test",
                start_date=datetime(2024, 1, i, 22, 0),
                end_date=datetime(2024, 1, i + 1, 6, 0),
                state=SleepState.ASLEEP,
            )
            for i in range(1, 4)  # 3 days to meet min_window_size
        ]

        activity_records = [
            ActivityRecord(
                source_name="test",
                start_date=datetime(2024, 1, i + 1, 12, 0),
                end_date=datetime(2024, 1, i + 1, 13, 0),
                activity_type=ActivityType.STEP_COUNT,
                value=5000,
                unit="steps",
            )
            for i in range(3)
        ]

        # Process WITHOUT heart rate data
        features = pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],  # Empty - no heart data
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 3),
            min_window_size=3,
        )

        # Should get features
        assert len(features) > 0

        # Check that HR values are 0.0 (not 70.0)
        for feature in features:
            if hasattr(feature, "seoul_features"):
                # Should be 0.0 when no data, not magic 70.0
                assert feature.seoul_features.avg_resting_hr == 0.0
                assert feature.seoul_features.hrv_sdnn == 0.0

    def test_baseline_persistence_without_hr_data(self):
        """Test baseline persistence when user has no HR data."""
        from big_mood_detector.infrastructure.repositories.file_baseline_repository import FileBaselineRepository
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = FileBaselineRepository(Path(tmpdir))
            engineer = AdvancedFeatureEngineer(
                user_id="no_hr_user", baseline_repository=repo
            )

            # Update sleep and activity baselines
            engineer._update_individual_baseline("sleep", 7.5)
            engineer._update_individual_baseline("activity", 10000)

            # Don't update HR/HRV baselines
            # Persist
            engineer.persist_baselines()

            # Load and verify
            saved = repo.get_baseline("no_hr_user")
            assert saved is not None

            # HR/HRV should be None, not magic defaults
            assert saved.heart_rate_mean is None
            assert saved.hrv_mean is None

            # But sleep/activity should be saved
            assert saved.sleep_mean == 7.5
            assert saved.activity_mean == 10000

    @pytest.mark.parametrize(
        "hr_value,hrv_value,should_update",
        [
            (0.0, 0.0, False),  # Zero values should not update
            (70.0, 50.0, True),  # Real values should update (no more magic check)
            (55.0, 65.0, True),  # Athlete values should update
            (85.0, 35.0, True),  # Stressed person values should update
            (-1.0, -1.0, False),  # Negative values should not update
        ],
    )
    def test_baseline_update_logic(self, hr_value, hrv_value, should_update):
        """Test that baseline updates only with valid data."""
        from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        engineer = AdvancedFeatureEngineer(user_id="test_user")

        # Create a mock heart summary
        heart_summary = DailyHeartSummary(
            date=date.today(),
            avg_resting_hr=hr_value,
            avg_hrv_sdnn=hrv_value,
        )

        # Call the normalization calculation which updates baselines
        engineer._calculate_normalized_features(
            current_date=date.today(),
            sleep=None,
            activity=None,
            heart=heart_summary,
        )

        # Check if baselines were updated
        if should_update:
            assert "hr" in engineer.individual_baselines
            assert "hrv" in engineer.individual_baselines
            assert engineer.individual_baselines["hr"]["mean"] == hr_value
            assert engineer.individual_baselines["hrv"]["mean"] == hrv_value
        else:
            # Should not have created baselines for invalid values
            assert "hr" not in engineer.individual_baselines
            assert "hrv" not in engineer.individual_baselines
