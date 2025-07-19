"""
TimescaleDB Baseline Repository Tests

Following best practices for testing repositories:
1. Use contract tests to ensure fake and real implementations behave identically
2. Don't overmock - test against real behavior
3. Use fast fake for development, real DB for integration validation
"""

import pytest
from datetime import date, datetime
from pathlib import Path

from big_mood_detector.infrastructure.di.container import setup_dependencies
from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)
from big_mood_detector.infrastructure.repositories.file_baseline_repository import FileBaselineRepository


class BaselineRepositoryContract:
    """
    Contract test for all BaselineRepository implementations.
    
    Both FileBaselineRepository and TimescaleBaselineRepository 
    must pass these tests to ensure behavioral compatibility.
    """
    
    def get_repository(self) -> BaselineRepositoryInterface:
        """Override in subclasses to provide repository implementation"""
        raise NotImplementedError
    
    def get_sample_baseline(self) -> UserBaseline:
        """Standard test baseline"""
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
    
    def test_save_and_retrieve_baseline(self):
        """Core contract: save baseline and retrieve it"""
        repo = self.get_repository()
        baseline = self.get_sample_baseline()
        
        # Save baseline
        repo.save_baseline(baseline)
        
        # Retrieve baseline
        retrieved = repo.get_baseline(baseline.user_id)
        
        assert retrieved is not None
        assert retrieved.user_id == baseline.user_id
        assert retrieved.sleep_mean == baseline.sleep_mean
        assert retrieved.activity_mean == baseline.activity_mean
    
    def test_get_nonexistent_baseline_returns_none(self):
        """Contract: missing baselines return None"""
        repo = self.get_repository()
        
        result = repo.get_baseline("nonexistent_user")
        
        assert result is None
    
    def test_get_baseline_history_empty_for_new_user(self):
        """Contract: new users have empty history"""
        repo = self.get_repository()
        
        history = repo.get_baseline_history("new_user")
        
        assert history == []
    
    def test_get_baseline_history_returns_chronological_order(self):
        """Contract: history is returned oldest first"""
        repo = self.get_repository()
        
        # Create separate baseline objects with different dates
        baseline1 = UserBaseline(
            user_id="history_test_user",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.0,
            sleep_std=1.0,
            activity_mean=7500.0,
            activity_std=1800.0,
            circadian_phase=21.5,
            last_updated=datetime(2024, 1, 1, 10, 0),
            data_points=28
        )
        
        baseline2 = UserBaseline(
            user_id="history_test_user",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 15, 10, 0),
            data_points=30
        )
        
        # Save in reverse chronological order
        repo.save_baseline(baseline2)
        repo.save_baseline(baseline1)
        
        # Retrieve history
        history = repo.get_baseline_history(baseline1.user_id)
        
        assert len(history) >= 2
        # Should be oldest first
        assert history[0].baseline_date <= history[1].baseline_date


class TestFileBaselineRepository(BaselineRepositoryContract):
    """Test FileBaselineRepository against the contract"""
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path
    
    def get_repository(self) -> BaselineRepositoryInterface:
        # Use real FileBaselineRepository - no mocking!
        return FileBaselineRepository(base_path=Path("./temp_test_baselines"))


class TestTimescaleBaselineRepository(BaselineRepositoryContract):
    """Test TimescaleBaselineRepository against the contract"""
    
    @pytest.fixture(autouse=True)
    def setup_test_container(self):
        """Set up test database using Testcontainers"""
        # This would use a real PostgreSQL container with TimescaleDB
        # No mocking - test against real database behavior!
        pytest.skip("Requires Docker setup - implement when ready for integration tests")
    
    def get_repository(self) -> BaselineRepositoryInterface:
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository
        )
        
        # Real TimescaleDB connection - configured for testing speed
        return TimescaleBaselineRepository(
            connection_string="postgresql://test:test@localhost:54321/test_baselines",
            enable_feast_sync=False  # Disable for pure repository tests
        )


class TestBaselineRepositoryIntegration:
    """
    Integration tests that verify real implementations work correctly.
    These run slower but provide crucial confidence.
    """
    
    def test_file_and_timescale_repos_are_interchangeable(self):
        """
        Integration test: Both implementations should behave identically
        for the same operations.
        """
        pytest.skip("Run only when both implementations are ready")
        
        file_repo = FileBaselineRepository(Path("./test_data"))
        # timescale_repo = TimescaleBaselineRepository(test_connection_string)
        
        baseline = UserBaseline(
            user_id="interop_test_user",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 15, 10, 30),
            data_points=30
        )
        
        # Both should handle the same data identically
        file_repo.save_baseline(baseline)
        # timescale_repo.save_baseline(baseline)
        
        file_result = file_repo.get_baseline(baseline.user_id)
        # timescale_result = timescale_repo.get_baseline(baseline.user_id)
        
        # Results should be equivalent (allowing for minor serialization differences)
        assert file_result.user_id == baseline.user_id
        # assert timescale_result.user_id == baseline.user_id 