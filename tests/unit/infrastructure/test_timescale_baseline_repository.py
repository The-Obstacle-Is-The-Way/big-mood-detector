"""
TDD Tests for TimescaleDB Baseline Repository

Following Uncle Bob's Red-Green approach to implement production-grade
baseline persistence with TimescaleDB + Feast integration.

Test-driven development phases:
1. Repository interface registration
2. Save path (write to TimescaleDB)
3. Load path (read from continuous aggregates)  
4. Pipeline integration
5. Online sync (Feast + Redis)
"""

import pytest
from unittest.mock import Mock, patch
from datetime import date, datetime
from pathlib import Path

from big_mood_detector.infrastructure.di.container import setup_dependencies
from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)


class TestTimescaleBaselineRepositoryDI:
    """Test dependency injection registration for TimescaleDB repository."""

    def test_timescale_baseline_repository_not_registered_by_default(self):
        """
        RED: Test that resolving TimescaleBaselineRepository raises DependencyNotFoundError.
        
        This test will fail initially because we haven't implemented 
        TimescaleBaselineRepository yet.
        """
        from unittest.mock import Mock
        
        mock_settings = Mock()
        mock_settings.data_dir = Path("data")
        
        container = setup_dependencies(mock_settings)
        
        with pytest.raises(Exception) as exc_info:
            # This should fail because TimescaleBaselineRepository doesn't exist yet
            container.resolve("TimescaleBaselineRepository")
        
        # Verify it's the right kind of error
        assert "TimescaleBaselineRepository" in str(exc_info.value)

    def test_can_register_timescale_baseline_repository(self):
        """
        RED: Test that we can register TimescaleBaselineRepository in DI container.
        
        This will fail until we create the TimescaleBaselineRepository class.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        from unittest.mock import Mock
        
        mock_settings = Mock()
        mock_settings.data_dir = Path("data")
        
        container = setup_dependencies(mock_settings)
        
        # Register the TimescaleDB repository
        container.register_singleton(
            TimescaleBaselineRepository,
            lambda: TimescaleBaselineRepository(
                connection_string="postgresql://test:test@localhost:5432/test_db"
            )
        )
        
        # Should be able to resolve it
        repo = container.resolve(TimescaleBaselineRepository)
        assert repo is not None
        assert isinstance(repo, BaselineRepositoryInterface)


class TestTimescaleBaselineRepositorySave:
    """Test baseline persistence to TimescaleDB (Red-Green Step 2)."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session for testing."""
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_session = Mock()
            mock_engine.return_value.sessionmaker.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def sample_baseline(self):
        """Sample baseline for testing."""
        return UserBaseline(
            user_id="test_user_123",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 15, 10, 30),
            data_points=30
        )

    def test_save_baseline_inserts_row_to_timescale(self, mock_db_session, sample_baseline):
        """
        RED: Test that repo.save(baseline) inserts a row to TimescaleDB.
        
        Will fail until we implement the save method.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        repo = TimescaleBaselineRepository("postgresql://test:test@localhost:5432/test_db")
        
        # Save the baseline
        repo.save_baseline(sample_baseline)
        
        # Verify database interaction
        assert mock_db_session.add.called
        assert mock_db_session.commit.called
        
        # Verify the saved data structure
        saved_call = mock_db_session.add.call_args[0][0]
        assert saved_call.user_id == "test_user_123"
        assert saved_call.sleep_mean == 7.5
        assert saved_call.sleep_std == 1.2

    def test_save_baseline_creates_bitemporal_record(self, mock_db_session, sample_baseline):
        """
        RED: Test that baseline saves create immutable bitemporal records.
        
        Following your suggestion for effective_ts versioning.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        repo = TimescaleBaselineRepository("postgresql://test:test@localhost:5432/test_db")
        
        # Save the baseline
        repo.save_baseline(sample_baseline)
        
        # Verify bitemporal structure
        saved_record = mock_db_session.add.call_args[0][0]
        assert hasattr(saved_record, 'effective_ts')
        assert hasattr(saved_record, 'user_id')
        assert hasattr(saved_record, 'feature_name')
        assert hasattr(saved_record, 'window')
        assert hasattr(saved_record, 'mean')
        assert hasattr(saved_record, 'std')
        assert hasattr(saved_record, 'n')


class TestTimescaleBaselineRepositoryLoad:
    """Test baseline retrieval from continuous aggregates (Red-Green Step 3)."""

    def test_get_baseline_queries_continuous_aggregate(self):
        """
        RED: Test that repo.get(user, metric, window, as_of) returns BaselineSnapshot.
        
        Will fail until we implement the get method with continuous aggregates.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_session = Mock()
            mock_engine.return_value.sessionmaker.return_value = mock_session
            
            # Mock query result from continuous aggregate
            mock_result = Mock()
            mock_result.user_id = "test_user_123"
            mock_result.sleep_mean = 7.5
            mock_result.sleep_std = 1.2
            mock_result.activity_mean = 8000.0
            mock_result.activity_std = 2000.0
            mock_result.circadian_phase = 22.0
            mock_result.data_points = 30
            mock_result.as_of = datetime(2024, 1, 15, 10, 30)
            
            mock_session.query.return_value.filter.return_value.first.return_value = mock_result
            
            repo = TimescaleBaselineRepository("postgresql://test:test@localhost:5432/test_db")
            
            # Get baseline
            baseline = repo.get_baseline("test_user_123")
            
            # Verify it returns a UserBaseline object
            assert isinstance(baseline, UserBaseline)
            assert baseline.user_id == "test_user_123"
            assert baseline.sleep_mean == 7.5
            assert baseline.sleep_std == 1.2

    def test_get_baseline_handles_missing_data(self):
        """
        RED: Test that repo.get() handles missing baselines gracefully.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        with patch("sqlalchemy.create_engine") as mock_engine:
            mock_session = Mock()
            mock_engine.return_value.sessionmaker.return_value = mock_session
            
            # Mock no result found
            mock_session.query.return_value.filter.return_value.first.return_value = None
            
            repo = TimescaleBaselineRepository("postgresql://test:test@localhost:5432/test_db")
            
            # Should return None for missing baselines
            baseline = repo.get_baseline("nonexistent_user")
            assert baseline is None


class TestTimescaleBaselineRepositoryIntegration:
    """Test end-to-end integration with AdvancedFeatureEngineer (Red-Green Step 4)."""

    def test_advanced_feature_engineer_uses_timescale_repository(self):
        """
        RED: Test that AdvancedFeatureEngineer calls repo.save and Z-scores match.
        
        This ensures the pipeline integration works correctly.
        """
        from big_mood_detector.domain.services.advanced_feature_engineering import (
            AdvancedFeatureEngineer
        )
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        # Mock the repository
        mock_repo = Mock(spec=TimescaleBaselineRepository)
        mock_baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 1, 0, 0),
            data_points=30
        )
        mock_repo.get_baseline.return_value = mock_baseline
        
        # Create engineer with TimescaleDB repository
        engineer = AdvancedFeatureEngineer(
            baseline_repository=mock_repo,
            user_id="test_user"
        )
        
        # Process some data (this should load and use baselines)
        # We'll need sample data here...
        
        # Verify repository interactions
        mock_repo.get_baseline.assert_called_with("test_user")
        
        # After processing, baselines should be persisted
        engineer.persist_baselines()
        mock_repo.save_baseline.assert_called()


class TestFeastOnlineSync:
    """Test Feast online store synchronization (Red-Green Step 5)."""

    def test_save_baseline_triggers_feast_sync(self):
        """
        RED: Test that after repo.save(), Feast online store contains same values.
        
        This tests your suggestion for online inference via Feast + Redis.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        with patch("feast.FeatureStore") as mock_feast:
            mock_store = Mock()
            mock_feast.return_value = mock_store
            
            repo = TimescaleBaselineRepository(
                connection_string="postgresql://test:test@localhost:5432/test_db",
                enable_feast_sync=True
            )
            
            baseline = UserBaseline(
                user_id="test_user_123",
                baseline_date=date(2024, 1, 15),
                sleep_mean=7.5,
                sleep_std=1.2,
                activity_mean=8000.0,
                activity_std=2000.0,
                circadian_phase=22.0,
                last_updated=datetime(2024, 1, 15, 10, 30),
                data_points=30
            )
            
            # Save baseline (should trigger Feast sync)
            repo.save_baseline(baseline)
            
            # Verify Feast push was called
            mock_store.push.assert_called()
            
            # Verify the pushed data structure
            pushed_data = mock_store.push.call_args[0][0]
            assert "user_id" in pushed_data.columns
            assert "sleep_mean" in pushed_data.columns
            assert "sleep_std" in pushed_data.columns

    def test_get_baseline_prefers_online_store(self):
        """
        RED: Test that repo.get() checks Feast online store first for performance.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        with patch("feast.FeatureStore") as mock_feast:
            mock_store = Mock()
            mock_feast.return_value = mock_store
            
            # Mock online store response
            mock_features = {
                "sleep_mean": [7.5],
                "sleep_std": [1.2],
                "activity_mean": [8000.0],
                "activity_std": [2000.0],
                "circadian_phase": [22.0],
                "data_points": [30]
            }
            mock_store.get_online_features.return_value.to_dict.return_value = mock_features
            
            repo = TimescaleBaselineRepository(
                connection_string="postgresql://test:test@localhost:5432/test_db",
                enable_feast_sync=True
            )
            
            # Get baseline (should hit online store first)
            baseline = repo.get_baseline("test_user_123")
            
            # Verify online store was queried
            mock_store.get_online_features.assert_called()
            
            # Verify correct baseline returned
            assert baseline.sleep_mean == 7.5
            assert baseline.activity_mean == 8000.0 