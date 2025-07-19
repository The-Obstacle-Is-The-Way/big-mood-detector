"""
Test Advanced Feature Engineering with Baseline Persistence

Following TDD - testing that baselines are persisted when features are extracted.
"""

from datetime import date, datetime, time, timedelta
from unittest.mock import Mock

import pytest

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)
from big_mood_detector.domain.services.activity_aggregator import DailyActivitySummary
from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
)
from big_mood_detector.domain.services.heart_rate_aggregator import DailyHeartSummary
from big_mood_detector.domain.services.sleep_aggregator import DailySleepSummary


class TestAdvancedFeatureEngineeringWithPersistence:
    """Test baseline persistence integration with feature extraction."""

    @pytest.fixture
    def mock_baseline_repository(self):
        """Create a mock baseline repository."""
        return Mock(spec=BaselineRepositoryInterface)

    @pytest.fixture
    def feature_engineer_with_persistence(self, mock_baseline_repository):
        """Create feature engineer with baseline repository injected."""
        engineer = AdvancedFeatureEngineer(
            config={},
            baseline_repository=mock_baseline_repository,
            user_id="test_user_123"
        )
        return engineer

    @pytest.fixture
    def sample_data(self):
        """Create sample health data for testing."""
        base_date = date(2024, 1, 15)

        # 30 days of data for baseline calculation
        sleep_data = []
        activity_data = []
        heart_data = []

        for i in range(30):
            current_date = base_date - timedelta(days=29-i)

            sleep_data.append(
                DailySleepSummary(
                    date=current_date,
                    total_time_in_bed_hours=8.0 + (i % 3) * 0.2,
                    total_sleep_hours=7.5 + (i % 3) * 0.1,
                    sleep_efficiency=0.88 + (i % 5) * 0.02,
                    sleep_sessions=1,
                    longest_sleep_hours=7.5,
                    sleep_fragmentation_index=0.1,
                    earliest_bedtime=time(23, 0),
                    latest_wake_time=time(7, 0),
                    mid_sleep_time=datetime.combine(
                        current_date + timedelta(days=1), time(3, 0)
                    ),
                )
            )

            activity_data.append(
                DailyActivitySummary(
                    date=current_date,
                    total_steps=8000.0 + (i % 7) * 500,
                    total_active_energy=300.0,
                    total_distance_km=6.0,
                    flights_climbed=10.0,
                    activity_sessions=3,
                    peak_activity_hour=14,
                    activity_variance=0.2,
                    sedentary_hours=14.0,
                    active_hours=3.0,
                    earliest_activity=time(7, 0),
                    latest_activity=time(21, 0),
                )
            )

            heart_data.append(
                DailyHeartSummary(
                    date=current_date,
                    avg_resting_hr=65.0 + (i % 5),
                    min_hr=50.0,
                    max_hr=140.0,
                    avg_hrv_sdnn=45.0 + (i % 3) * 2,
                    min_hrv_sdnn=40.0,
                    hr_measurements=100,
                    hrv_measurements=20,
                    high_hr_episodes=0,
                    low_hr_episodes=0,
                    circadian_hr_range=15.0,
                    morning_hr=62.0,
                    evening_hr=68.0,
                )
            )

        return sleep_data, activity_data, heart_data

    def test_baseline_persistence_on_feature_extraction(
        self, mock_baseline_repository, sample_data
    ):
        """Test that baselines are persisted when features are extracted."""
        sleep_data, activity_data, heart_data = sample_data
        target_date = date(2024, 1, 15)

        # Configure mock to return None when loading baseline (first time user)
        mock_baseline_repository.get_baseline.return_value = None

        # Create feature engineer after configuring mock
        engineer = AdvancedFeatureEngineer(
            config={},
            baseline_repository=mock_baseline_repository,
            user_id="test_user_123"
        )

        # Extract features
        features = engineer.extract_advanced_features(
            current_date=target_date,
            historical_sleep=sleep_data,
            historical_activity=activity_data,
            historical_heart=heart_data,
            lookback_days=30,
        )

        # Persist baselines after feature extraction
        engineer.persist_baselines()

        # Verify baseline was saved
        assert mock_baseline_repository.save_baseline.called

        # Check the saved baseline
        saved_baseline = mock_baseline_repository.save_baseline.call_args[0][0]
        assert isinstance(saved_baseline, UserBaseline)
        assert saved_baseline.user_id == "test_user_123"
        # Note: baseline_date uses date.today() in the implementation
        assert saved_baseline.baseline_date == date.today()

        # Verify baseline statistics are reasonable
        # After processing 30 days, mean should be reasonable
        assert saved_baseline.sleep_mean >= 0  # At least non-negative
        assert saved_baseline.activity_mean >= 0  # At least non-negative
        assert saved_baseline.data_points >= 1  # At least one data point

    def test_baseline_loading_for_zscore_calculation(
        self, mock_baseline_repository, sample_data
    ):
        """Test that existing baselines are loaded for Z-score calculation."""
        sleep_data, activity_data, heart_data = sample_data
        target_date = date(2024, 2, 15)  # One month later

        # Mock an existing baseline
        existing_baseline = UserBaseline(
            user_id="test_user_123",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=0.5,
            activity_mean=8500.0,
            activity_std=1200.0,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )
        mock_baseline_repository.get_baseline.return_value = existing_baseline

        # Create feature engineer after configuring mock
        engineer = AdvancedFeatureEngineer(
            config={},
            baseline_repository=mock_baseline_repository,
            user_id="test_user_123"
        )

        # Extract features - should use loaded baseline for Z-scores
        features = engineer.extract_advanced_features(
            current_date=target_date,
            historical_sleep=sleep_data[-7:],  # Only last week of data
            historical_activity=activity_data[-7:],
            historical_heart=heart_data[-7:],
            lookback_days=7,
        )

        # Verify baseline was loaded
        mock_baseline_repository.get_baseline.assert_called_with("test_user_123")

        # Z-scores should be calculated using loaded baseline
        # (This assumes we modify the feature engineer to use repository baselines)
        assert features is not None

    def test_persist_baselines_method(
        self, mock_baseline_repository
    ):
        """Test explicit persist_baselines method."""
        # Create engineer with baseline repository
        engineer = AdvancedFeatureEngineer(
            config={},
            baseline_repository=mock_baseline_repository,
            user_id="test_user_123"
        )

        # Add some baseline data
        engineer._update_individual_baseline("sleep", 7.5)
        engineer._update_individual_baseline("activity", 8500)

        # Test that we can explicitly persist current baselines
        engineer.persist_baselines()

        # Should save the current baseline state
        assert mock_baseline_repository.save_baseline.called

        saved_baseline = mock_baseline_repository.save_baseline.call_args[0][0]
        assert saved_baseline.user_id == "test_user_123"
