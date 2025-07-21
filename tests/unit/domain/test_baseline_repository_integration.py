"""
Test Baseline Repository Integration

Tests for integrating baseline persistence with feature extraction.
Following TDD - write the test first.
"""

from datetime import date, datetime
from unittest.mock import Mock

import pytest

class TestBaselineRepositoryIntegration:
    """Test integration of baseline repository with feature engineering."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock baseline repository."""
        return Mock(spec=BaselineRepositoryInterface)

    @pytest.fixture
    def temp_baseline_dir(self, tmp_path):
        """Create temporary directory for file repository."""
        return tmp_path / "baselines"

    def test_feature_engineer_accepts_repository(self, mock_repository):
        """Test that AdvancedFeatureEngineer can be initialized with repository."""
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        # Should accept baseline_repository parameter
        engineer = AdvancedFeatureEngineer(
            config={}, baseline_repository=mock_repository
        )

        assert engineer.baseline_repository == mock_repository

    def test_loads_baseline_on_init_if_user_provided(self, mock_repository):
        """Test that baselines are loaded from repository on initialization."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        # Mock existing baseline
        existing_baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=0.5,
            activity_mean=8500.0,
            activity_std=1200.0,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )
        mock_repository.get_baseline.return_value = existing_baseline

        # Initialize with user_id
        engineer = AdvancedFeatureEngineer(
            config={}, baseline_repository=mock_repository, user_id="test_user"
        )

        # Should have loaded baseline
        mock_repository.get_baseline.assert_called_with("test_user")

        # Individual baselines should be populated
        assert "sleep" in engineer.individual_baselines
        assert engineer.individual_baselines["sleep"]["mean"] == 7.5
        assert engineer.individual_baselines["sleep"]["std"] == 0.5

    def test_persist_baselines_saves_to_repository(self, mock_repository):
        """Test that persist_baselines method saves to repository."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        # Set up mock to return None (no existing baseline)
        mock_repository.get_baseline.return_value = None

        engineer = AdvancedFeatureEngineer(
            config={}, baseline_repository=mock_repository, user_id="test_user"
        )

        # Initialize the engineer's baselines
        engineer._load_baselines_from_repository()

        # Update some baselines
        engineer._update_individual_baseline("sleep", 7.5)
        engineer._update_individual_baseline("sleep", 7.8)
        engineer._update_individual_baseline("sleep", 7.2)

        engineer._update_individual_baseline("activity", 8000)
        engineer._update_individual_baseline("activity", 8500)
        engineer._update_individual_baseline("activity", 9000)

        # Persist baselines
        engineer.persist_baselines()

        # Should save baseline
        assert mock_repository.save_baseline.called
        saved_baseline = mock_repository.save_baseline.call_args[0][0]

        assert isinstance(saved_baseline, UserBaseline)
        assert saved_baseline.user_id == "test_user"
        assert saved_baseline.sleep_mean == pytest.approx(7.5, rel=0.1)
        assert saved_baseline.activity_mean == pytest.approx(8500, rel=0.1)
        assert saved_baseline.data_points > 0

    def test_no_persistence_without_repository(self):
        """Test that engineer works without repository (backward compatibility)."""
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        # Should work without repository
        engineer = AdvancedFeatureEngineer(config={})

        assert engineer.baseline_repository is None

        # persist_baselines should be no-op
        engineer.persist_baselines()  # Should not raise

    def test_integration_with_file_repository(self, temp_baseline_dir):
        """Test end-to-end integration with actual file repository."""
        from big_mood_detector.infrastructure.repositories.file_baseline_repository import FileBaselineRepository
        from big_mood_detector.domain.services.advanced_feature_engineering import AdvancedFeatureEngineer

        repository = FileBaselineRepository(temp_baseline_dir)

        # First engineer saves baselines
        engineer1 = AdvancedFeatureEngineer(
            config={}, baseline_repository=repository, user_id="test_user"
        )

        # Add some data
        for i in range(10):
            engineer1._update_individual_baseline("sleep", 7.5 + i * 0.1)
            engineer1._update_individual_baseline("activity", 8000 + i * 100)

        engineer1.persist_baselines()

        # Second engineer loads baselines
        engineer2 = AdvancedFeatureEngineer(
            config={}, baseline_repository=repository, user_id="test_user"
        )

        # Should have loaded the saved baselines
        assert "sleep" in engineer2.individual_baselines
        assert engineer2.individual_baselines["sleep"]["mean"] > 0
        assert engineer2.individual_baselines["activity"]["mean"] > 0
