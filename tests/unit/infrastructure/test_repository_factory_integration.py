"""
Integration test showing repository factory usage with feature engineering.

This demonstrates how the factory-created repository integrates with
the AdvancedFeatureEngineer for baseline persistence.
"""
import tempfile
from pathlib import Path

import pytest

from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
)
from big_mood_detector.infrastructure.repositories.baseline_repository_factory import (
    BaselineRepositoryFactory,
)


class TestRepositoryFactoryWithFeatureEngineering:
    """Test repository factory integration with feature engineering."""

    def test_feature_engineer_uses_factory_repository(self):
        """Test that AdvancedFeatureEngineer can use factory-created repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create factory and get repository
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))
            repository = factory.get_repository()

            # Create two feature engineers with same repository
            engineer1 = AdvancedFeatureEngineer(
                user_id="test_user",
                baseline_repository=repository,
            )

            AdvancedFeatureEngineer(
                user_id="test_user",
                baseline_repository=repository,
            )

            # Update baselines with engineer1
            for i in range(5):
                engineer1._update_individual_baseline("sleep", 7.5 + i * 0.1)
                engineer1._update_individual_baseline("activity", 8000 + i * 100)

            # Persist with engineer1
            engineer1.persist_baselines()

            # Engineer2 should load baselines automatically on init
            # since it has the same user_id and repository

            # Verify baseline values
            baseline = repository.get_baseline("test_user")
            assert baseline is not None
            assert baseline.sleep_mean > 7.5
            assert baseline.activity_mean > 8000

    def test_factory_supports_multiple_users(self):
        """Test that factory-created repository supports multiple users."""
        with tempfile.TemporaryDirectory() as tmpdir:
            factory = BaselineRepositoryFactory(base_path=Path(tmpdir))
            repository = factory.get_repository()

            # Create baselines for multiple users
            users = ["alice", "bob", "charlie"]

            for i, user_id in enumerate(users):
                engineer = AdvancedFeatureEngineer(
                    user_id=user_id,
                    baseline_repository=repository,
                )

                # Give each user different baseline values
                sleep_base = 6.5 + i * 0.5
                activity_base = 7000 + i * 1000

                for j in range(7):
                    engineer._update_individual_baseline("sleep", sleep_base + j * 0.1)
                    engineer._update_individual_baseline("activity", activity_base + j * 100)

                engineer.persist_baselines()

            # Verify all baselines exist and are different
            baselines = {}
            for user_id in users:
                baseline = repository.get_baseline(user_id)
                assert baseline is not None
                baselines[user_id] = baseline

            # Check that baselines are personalized
            assert baselines["alice"].sleep_mean != baselines["bob"].sleep_mean
            assert baselines["bob"].activity_mean != baselines["charlie"].activity_mean

    def test_environment_variable_configuration(self):
        """Test that factory respects environment variables."""
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test file repository selection
            with patch.dict(os.environ, {"BASELINE_REPOSITORY_TYPE": "file"}):
                factory = BaselineRepositoryFactory(base_path=Path(tmpdir))
                repo = factory.create_repository()
                assert repo.__class__.__name__ == "FileBaselineRepository"

            # Test invalid type handling
            with patch.dict(os.environ, {"BASELINE_REPOSITORY_TYPE": "invalid"}):
                factory = BaselineRepositoryFactory(base_path=Path(tmpdir))
                with pytest.raises(ValueError, match="Unknown repository type"):
                    factory.create_repository()
