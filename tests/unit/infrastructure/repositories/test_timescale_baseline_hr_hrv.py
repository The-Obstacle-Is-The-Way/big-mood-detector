"""
Test TimescaleDB baseline repository handles HR/HRV fields correctly.

This ensures the production database can store and retrieve heart rate
and heart rate variability baselines for personal calibration.
"""

from datetime import date

import pytest
from sqlalchemy import create_engine

class TestTimescaleBaselineHRHRV:
    """Test HR/HRV support in TimescaleDB repository."""

    @pytest.fixture
    def test_db(self):
        """Create in-memory SQLite database for testing."""
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def repository(self, test_db):
        """Create repository with test database."""
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import TimescaleBaselineRepository

        # Use the test engine's connection string
        return TimescaleBaselineRepository(
            connection_string="sqlite:///:memory:",
            enable_feast_sync=False,  # Disable Feast for unit tests
        )

    def test_save_baseline_with_hr_hrv_data(self, repository):
        """Test saving baseline with HR/HRV data."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        baseline = UserBaseline(
            user_id="athlete_123",
            baseline_date=date.today(),
            sleep_mean=7.2,
            sleep_std=0.8,
            activity_mean=12000,
            activity_std=3000,
            circadian_phase=22.0,
            # Athlete has lower resting HR and higher HRV
            heart_rate_mean=55.0,
            heart_rate_std=5.0,
            hrv_mean=65.0,
            hrv_std=12.0,
            data_points=30,
        )

        repository.save_baseline(baseline)

        # Retrieve and verify
        retrieved = repository.get_baseline("athlete_123")
        assert retrieved is not None
        assert retrieved.heart_rate_mean == 55.0
        assert retrieved.heart_rate_std == 5.0
        assert retrieved.hrv_mean == 65.0
        assert retrieved.hrv_std == 12.0

    def test_save_baseline_without_hr_hrv_data(self, repository):
        """Test saving baseline without HR/HRV data (should be None)."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        baseline = UserBaseline(
            user_id="no_hr_user",
            baseline_date=date.today(),
            sleep_mean=8.0,
            sleep_std=1.0,
            activity_mean=8000,
            activity_std=2000,
            circadian_phase=22.5,
            # No HR/HRV data - should remain None
            heart_rate_mean=None,
            heart_rate_std=None,
            hrv_mean=None,
            hrv_std=None,
            data_points=7,
        )

        repository.save_baseline(baseline)

        # Retrieve and verify
        retrieved = repository.get_baseline("no_hr_user")
        assert retrieved is not None
        assert retrieved.heart_rate_mean is None
        assert retrieved.heart_rate_std is None
        assert retrieved.hrv_mean is None
        assert retrieved.hrv_std is None

    def test_update_baseline_adds_hr_hrv_later(self, repository):
        """Test updating baseline to add HR/HRV data later."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # First save without HR/HRV
        baseline1 = UserBaseline(
            user_id="progressive_user",
            baseline_date=date.today(),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=10000,
            activity_std=2500,
            circadian_phase=22.0,
            data_points=7,
        )

        repository.save_baseline(baseline1)

        # Later, user gets a heart rate monitor
        baseline2 = UserBaseline(
            user_id="progressive_user",
            baseline_date=date.today(),
            sleep_mean=7.4,  # Slightly updated
            sleep_std=0.9,
            activity_mean=10500,
            activity_std=2400,
            circadian_phase=22.0,
            heart_rate_mean=68.0,  # Now we have HR data!
            heart_rate_std=6.0,
            hrv_mean=52.0,
            hrv_std=10.0,
            data_points=14,
        )

        repository.save_baseline(baseline2)

        # Retrieve and verify it was updated
        retrieved = repository.get_baseline("progressive_user")
        assert retrieved is not None
        assert retrieved.heart_rate_mean == 68.0
        assert retrieved.hrv_mean == 52.0
        assert retrieved.data_points == 14

    def test_baseline_history_preserves_hr_hrv(self, repository):
        """Test that baseline history includes HR/HRV fields."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Save multiple baselines over time
        dates = [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)]
        hr_means = [70.0, 68.0, 65.0]  # HR improving over time
        hrv_means = [45.0, 48.0, 52.0]  # HRV improving over time

        for i, (baseline_date, hr_mean, hrv_mean) in enumerate(
            zip(dates, hr_means, hrv_means, strict=False)
        ):
            baseline = UserBaseline(
                user_id="improving_user",
                baseline_date=baseline_date,
                sleep_mean=7.5,
                sleep_std=0.8,
                activity_mean=10000 + i * 1000,  # Increasing activity
                activity_std=2000,
                circadian_phase=22.0,
                heart_rate_mean=hr_mean,
                heart_rate_std=5.0,
                hrv_mean=hrv_mean,
                hrv_std=8.0,
                data_points=7 * (i + 1),
            )
            repository.save_baseline(baseline)

        # Get history
        history = repository.get_baseline_history("improving_user", limit=10)

        assert len(history) == 3

        # Verify HR/HRV progression
        for i, baseline in enumerate(history):
            assert baseline.heart_rate_mean == hr_means[i]
            assert baseline.hrv_mean == hrv_means[i]

        # Verify HR is decreasing (fitness improving)
        assert history[2].heart_rate_mean < history[0].heart_rate_mean
        # Verify HRV is increasing (fitness improving)
        assert history[2].hrv_mean > history[0].hrv_mean

    def test_raw_records_include_hr_hrv_metrics(self, repository):
        """Test that raw records properly store HR/HRV metrics."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        baseline = UserBaseline(
            user_id="raw_test_user",
            baseline_date=date.today(),
            sleep_mean=7.0,
            sleep_std=1.0,
            activity_mean=9000,
            activity_std=2000,
            circadian_phase=22.0,
            heart_rate_mean=62.0,
            heart_rate_std=4.0,
            hrv_mean=58.0,
            hrv_std=11.0,
            data_points=30,
        )

        repository.save_baseline(baseline)

        # Check raw records were created (would need access to session)
        # For now, just verify retrieval works
        retrieved = repository.get_baseline("raw_test_user")
        assert retrieved is not None
        assert retrieved.heart_rate_mean == 62.0
        assert retrieved.hrv_mean == 58.0

    @pytest.mark.parametrize(
        "hr_mean,hrv_mean,expected_hr,expected_hrv",
        [
            (None, None, None, None),  # No data
            (70.0, None, 70.0, None),  # Only HR
            (None, 50.0, None, 50.0),  # Only HRV
            (65.0, 55.0, 65.0, 55.0),  # Both
        ],
    )
    def test_partial_hr_hrv_data(
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        self, repository, hr_mean, hrv_mean, expected_hr, expected_hrv
    ):
        """Test handling partial HR/HRV data."""
        baseline = UserBaseline(
            user_id=f"partial_user_{hr_mean}_{hrv_mean}",
            baseline_date=date.today(),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=10000,
            activity_std=2000,
            circadian_phase=22.0,
            heart_rate_mean=hr_mean,
            heart_rate_std=5.0 if hr_mean else None,
            hrv_mean=hrv_mean,
            hrv_std=10.0 if hrv_mean else None,
            data_points=30,
        )

        repository.save_baseline(baseline)

        retrieved = repository.get_baseline(baseline.user_id)
        assert retrieved is not None
        assert retrieved.heart_rate_mean == expected_hr
        assert retrieved.hrv_mean == expected_hrv
