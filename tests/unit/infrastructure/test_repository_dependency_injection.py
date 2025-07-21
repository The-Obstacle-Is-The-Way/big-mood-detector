"""
Test Dependency Injection for Repositories

Verify repository implementations are properly registered.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

class TestRepositoryDependencyInjection:
    """Test DI container repository registrations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_settings(self, temp_dir):
        """Create mock settings with data directory."""
        settings = Mock()
        settings.data_dir = temp_dir
        return settings

    def test_repository_interfaces_resolve_to_implementations(self, mock_settings):
        """Test that repository interfaces resolve to their concrete implementations."""
        from big_mood_detector.domain.repositories.heart_rate_repository import HeartRateRepositoryInterface
        from big_mood_detector.domain.repositories.sleep_repository import SleepRepositoryInterface
        from big_mood_detector.infrastructure.repositories.file_heart_rate_repository import FileHeartRateRepository
        from big_mood_detector.infrastructure.repositories.file_activity_repository import FileActivityRepository
        from big_mood_detector.domain.repositories.activity_repository import ActivityRepositoryInterface
        from big_mood_detector.infrastructure.di.container import setup_dependencies
        from big_mood_detector.infrastructure.repositories.file_sleep_repository import FileSleepRepository

        container = setup_dependencies(mock_settings)

        # Resolve interfaces
        activity_repo = container.resolve(ActivityRepositoryInterface)
        heart_rate_repo = container.resolve(HeartRateRepositoryInterface)
        sleep_repo = container.resolve(SleepRepositoryInterface)

        # Verify correct implementations
        assert isinstance(activity_repo, FileActivityRepository)
        assert isinstance(heart_rate_repo, FileHeartRateRepository)
        assert isinstance(sleep_repo, FileSleepRepository)

    def test_concrete_repositories_are_singletons(self, mock_settings):
        """Test that concrete repositories are registered as singletons."""
        from big_mood_detector.infrastructure.repositories.file_activity_repository import FileActivityRepository
        from big_mood_detector.infrastructure.di.container import setup_dependencies

        container = setup_dependencies(mock_settings)

        # Resolve same type multiple times
        repo1 = container.resolve(FileActivityRepository)
        repo2 = container.resolve(FileActivityRepository)

        # Should be same instance (singleton)
        assert repo1 is repo2

    def test_repository_data_directories_are_configured(self, mock_settings, temp_dir):
        """Test that repositories use the configured data directory."""
        from big_mood_detector.infrastructure.repositories.file_activity_repository import FileActivityRepository
        from big_mood_detector.infrastructure.di.container import setup_dependencies

        container = setup_dependencies(mock_settings)

        activity_repo = container.resolve(FileActivityRepository)

        # Verify data directory is set correctly
        assert activity_repo.data_dir == temp_dir
        # Verify subdirectory was created
        assert (temp_dir / "activity_records").exists()
