"""
Test baseline repository factory for selecting between File and TimescaleDB.

This ensures we can configure which baseline repository to use via
environment variables or settings.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
)


class TestBaselineRepositoryFactory:
    """Test repository selection and configuration."""

    def test_create_file_repository_by_default(self):
        """Test that FileBaselineRepository is created by default."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))

            # Should create file repository by default
            repo = factory.create_repository()

            assert isinstance(repo, BaselineRepositoryInterface)
            # Check it's specifically a FileBaselineRepository
            assert repo.__class__.__name__ == "FileBaselineRepository"

    def test_create_timescale_repository_with_env_var(self):
        """Test creating TimescaleDB repository via environment variable."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))

            # Set environment variable
            with patch.dict(os.environ, {"BASELINE_REPOSITORY_TYPE": "timescale"}):
                # Mock connection string
                with patch.dict(
                    os.environ, {"TIMESCALE_CONNECTION_STRING": "postgresql://test"}
                ):
                    repo = factory.create_repository()

            assert isinstance(repo, BaselineRepositoryInterface)
            # Check it's specifically a TimescaleBaselineRepository
            assert repo.__class__.__name__ == "TimescaleBaselineRepository"

    def test_create_repository_with_explicit_type(self):
        """Test creating repository with explicit type parameter."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))

            # Create file repository explicitly
            file_repo = factory.create_repository(repository_type="file")
            assert file_repo.__class__.__name__ == "FileBaselineRepository"

            # Create timescale repository explicitly
            with patch.dict(
                os.environ, {"TIMESCALE_CONNECTION_STRING": "postgresql://test"}
            ):
                ts_repo = factory.create_repository(repository_type="timescale")
                assert ts_repo.__class__.__name__ == "TimescaleBaselineRepository"

    def test_invalid_repository_type_raises_error(self):
        """Test that invalid repository type raises ValueError."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))

            with pytest.raises(ValueError, match="Unknown repository type"):
                factory.create_repository(repository_type="invalid")

    def test_timescale_requires_connection_string(self):
        """Test that TimescaleDB requires connection string."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))

            # Try to create TimescaleDB without connection string
            with pytest.raises(ValueError, match="TIMESCALE_CONNECTION_STRING"):
                factory.create_repository(repository_type="timescale")

    def test_factory_configuration_from_settings(self):
        """Test factory can be configured from settings object."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
            BaselineRepositorySettings,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings
            settings = BaselineRepositorySettings(
                repository_type="file",
                base_path=Path(tmpdir),
                timescale_connection_string=None,
                enable_feast_sync=False,
            )

            factory = BaselineRepositoryFactory.from_settings(settings)
            repo = factory.create_repository()

            assert repo.__class__.__name__ == "FileBaselineRepository"

    def test_singleton_pattern_returns_same_instance(self):
        """Test that factory can return singleton instances."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))

            # Get repository twice
            repo1 = factory.get_repository()
            repo2 = factory.get_repository()

            # Should be the same instance
            assert repo1 is repo2

    def test_create_vs_get_repository(self):
        """Test difference between create and get methods."""
        from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
            BaselineRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))

            # create_repository always returns new instance
            repo1 = factory.create_repository()
            repo2 = factory.create_repository()
            assert repo1 is not repo2

            # get_repository returns singleton
            repo3 = factory.get_repository()
            repo4 = factory.get_repository()
            assert repo3 is repo4
