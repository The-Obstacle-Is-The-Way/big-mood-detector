"""
Test File-based Baseline Repository

Simple file-based implementation for baseline storage.
Following YAGNI principle - start simple, refactor when needed.
"""

from datetime import date, datetime

import pytest

class TestFileBaselineRepository:
    """Test file-based baseline repository implementation."""

    @pytest.fixture
    def temp_baseline_dir(self, tmp_path):
        """Create a temporary directory for baseline storage."""
        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        return baseline_dir

    @pytest.fixture
    def repository(self, temp_baseline_dir):
        """Create repository instance."""

        return FileBaselineRepository(temp_baseline_dir)

    def test_save_and_retrieve_baseline(self, repository):
        """Test saving and retrieving a baseline."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Given a baseline
        baseline = UserBaseline(
            user_id="user123",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=0.8,
            activity_mean=8000.0,
            activity_std=1500.0,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )

        # When we save it
        repository.save_baseline(baseline)

        # Then we can retrieve it
        retrieved = repository.get_baseline("user123")
        assert retrieved is not None
        assert retrieved.user_id == baseline.user_id
        assert retrieved.sleep_mean == baseline.sleep_mean
        assert retrieved.activity_mean == baseline.activity_mean

    def test_get_nonexistent_baseline(self, repository):
        """Test retrieving baseline for user with no data."""
        result = repository.get_baseline("nonexistent")
        assert result is None

    def test_update_baseline(self, repository):
        """Test updating an existing baseline."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Given an initial baseline
        baseline1 = UserBaseline(
            user_id="user123",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.0,
            sleep_std=0.5,
            activity_mean=7000.0,
            activity_std=1000.0,
            circadian_phase=-0.5,
            last_updated=datetime(2024, 1, 1, 12, 0),
            data_points=30,
        )
        repository.save_baseline(baseline1)

        # When we save a new baseline for the same user
        baseline2 = UserBaseline(
            user_id="user123",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=0.8,
            activity_mean=8000.0,
            activity_std=1500.0,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )
        repository.save_baseline(baseline2)

        # Then we get the most recent one
        retrieved = repository.get_baseline("user123")
        assert retrieved.baseline_date == date(2024, 1, 15)
        assert retrieved.sleep_mean == 7.5

    def test_get_baseline_history(self, repository):
        """Test retrieving baseline history."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Given multiple baselines
        dates = [date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 1)]
        for i, baseline_date in enumerate(dates):
            baseline = UserBaseline(
                user_id="user123",
                baseline_date=baseline_date,
                sleep_mean=7.0 + i * 0.5,
                sleep_std=0.5 + i * 0.1,
                activity_mean=7000.0 + i * 1000,
                activity_std=1000.0 + i * 100,
                circadian_phase=-0.5 + i * 0.25,
                last_updated=datetime.combine(baseline_date, datetime.min.time()),
                data_points=30,
            )
            repository.save_baseline(baseline)

        # When we get history
        history = repository.get_baseline_history("user123", limit=2)

        # Then we get the most recent ones
        assert len(history) == 2
        assert history[0].baseline_date == date(2024, 1, 15)
        assert history[1].baseline_date == date(2024, 2, 1)

    def test_empty_history(self, repository):
        """Test getting history for user with no baselines."""
        history = repository.get_baseline_history("nonexistent")
        assert history == []

    def test_multiple_users(self, repository):
        """Test storing baselines for multiple users."""
        from big_mood_detector.domain.repositories.baseline_repository_interface import UserBaseline

        # Given baselines for different users
        baseline1 = UserBaseline(
            user_id="user1",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=0.8,
            activity_mean=8000.0,
            activity_std=1500.0,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )
        baseline2 = UserBaseline(
            user_id="user2",
            baseline_date=date(2024, 1, 15),
            sleep_mean=8.0,
            sleep_std=1.0,
            activity_mean=6000.0,
            activity_std=1200.0,
            circadian_phase=0.5,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )

        # When we save them
        repository.save_baseline(baseline1)
        repository.save_baseline(baseline2)

        # Then each user has their own baseline
        user1_baseline = repository.get_baseline("user1")
        user2_baseline = repository.get_baseline("user2")

        assert user1_baseline.sleep_mean == 7.5
        assert user2_baseline.sleep_mean == 8.0
