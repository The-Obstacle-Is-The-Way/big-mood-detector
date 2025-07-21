"""
Test DI Container Baseline Repository Registration

Tests for registering baseline repository in the dependency injection container.
"""

import pytest

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
)
from big_mood_detector.infrastructure.di.container import (
    Container,
    DependencyNotFoundError,
    setup_dependencies,
)
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


class TestDIContainerBaselineRepository:
    """Test baseline repository registration in DI container."""

    @pytest.fixture
    def container(self):
        """Create a fresh container for testing."""
        return Container()

    @pytest.fixture
    def mock_settings(self, tmp_path):
        """Mock settings for dependency setup."""
        from types import SimpleNamespace

        return SimpleNamespace(
            data_dir=tmp_path / "data",
            DATA_DIR=tmp_path / "data",
        )

    def test_baseline_repository_not_registered_by_default(self, container):
        """Test that BaselineRepository is not registered by default."""
        with pytest.raises(DependencyNotFoundError):
            container.resolve(BaselineRepositoryInterface)

    def test_register_baseline_repository(self, container, tmp_path):
        """Test registering baseline repository."""
        # Create baselines directory
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()

        # Register the repository
        container.register_singleton(
            BaselineRepositoryInterface, lambda: FileBaselineRepository(baselines_dir)
        )

        # Should resolve successfully
        repo = container.resolve(BaselineRepositoryInterface)
        assert repo is not None
        assert isinstance(repo, FileBaselineRepository)

        # Should be singleton
        repo2 = container.resolve(BaselineRepositoryInterface)
        assert repo is repo2

    def test_setup_dependencies_includes_baseline_repository(self, mock_settings):
        """Test that setup_dependencies registers baseline repository."""
        # Clear any existing container
        import big_mood_detector.infrastructure.di.container as di_module

        di_module._container = None
        di_module.get_container.cache_clear()

        # Setup dependencies
        container = setup_dependencies(mock_settings)

        # Should be able to resolve baseline repository
        repo = container.resolve(BaselineRepositoryInterface)
        assert repo is not None
        assert isinstance(repo, FileBaselineRepository)

    def test_advanced_feature_engineer_with_repository(self, mock_settings):
        """Test that AdvancedFeatureEngineer gets baseline repository injected."""
        # Setup dependencies
        import big_mood_detector.infrastructure.di.container as di_module

        di_module._container = None
        di_module.get_container.cache_clear()

        setup_dependencies(mock_settings)

        # Resolve AdvancedFeatureEngineer

        # Note: We'd need to update the DI registration to inject the repository
        # For now, this is a placeholder test
        # engineer = container.resolve(AdvancedFeatureEngineer)
        # assert engineer.baseline_repository is not None
