"""
Test for extending UserBaseline to include HR/HRV metrics.

Following TDD - write failing test first, then implement.
"""

from datetime import date, datetime

class TestUserBaselineHRHRV:
    """Test that UserBaseline can store HR/HRV metrics."""

    def test_user_baseline_should_include_hr_hrv_metrics(self):
        """
        Test that UserBaseline includes heart rate and HRV baselines.
        This should PASS now after implementation.
        """
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Create baseline with HR/HRV metrics
        baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            # NEW FIELDS - now they exist!
            heart_rate_mean=70.0,
            heart_rate_std=5.0,
            hrv_mean=45.0,
            hrv_std=10.0,
            last_updated=datetime(2024, 1, 15, 10, 0),
            data_points=30,
        )

        # Verify all fields are set correctly
        assert baseline.heart_rate_mean == 70.0
        assert baseline.heart_rate_std == 5.0
        assert baseline.hrv_mean == 45.0
        assert baseline.hrv_std == 10.0

    def test_baseline_repository_interface_compatible_with_extended_baseline(self):
        """
        Test that repository interface can handle extended baselines.
        This documents the desired behavior.
        """
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Create current baseline (without HR/HRV)
        baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 15, 10, 0),
            data_points=30,
        )

        # Verify current fields exist
        assert baseline.sleep_mean == 7.5
        assert baseline.activity_mean == 8000.0

        # Verify new fields are None (no magic defaults!)
        assert baseline.heart_rate_mean is None
        assert baseline.hrv_mean is None
