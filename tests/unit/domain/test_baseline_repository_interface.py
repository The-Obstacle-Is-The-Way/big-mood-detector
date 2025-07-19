"""
Test Baseline Repository Interface

Following TDD and Uncle Bob's principles:
- Interface Segregation (small, focused interface)
- Dependency Inversion (depend on abstractions)
- Single Responsibility (only baseline storage)
"""

from datetime import date, datetime
from typing import Optional

import pytest

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)


class TestBaselineRepositoryInterface:
    """Test the baseline repository interface contract."""
    
    def test_baseline_value_object(self):
        """Test that UserBaseline is a proper value object."""
        baseline = UserBaseline(
            user_id="user123",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=0.8,
            activity_mean=8000,
            activity_std=1500,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )
        
        # Should be immutable (frozen)
        with pytest.raises(AttributeError):
            baseline.sleep_mean = 8.0
        
        # Should have all required fields
        assert baseline.user_id == "user123"
        assert baseline.sleep_mean == 7.5
        assert baseline.data_points == 30
    
    def test_repository_interface_definition(self):
        """Test that interface is properly defined."""
        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            BaselineRepositoryInterface()
        
        # Should define required methods
        assert hasattr(BaselineRepositoryInterface, 'save_baseline')
        assert hasattr(BaselineRepositoryInterface, 'get_baseline')
        assert hasattr(BaselineRepositoryInterface, 'get_baseline_history')
    
    def test_mock_implementation(self):
        """Test with a mock implementation to verify interface."""
        class MockBaselineRepository(BaselineRepositoryInterface):
            def __init__(self):
                self._baselines: dict[str, list[UserBaseline]] = {}
            
            def save_baseline(self, baseline: UserBaseline) -> None:
                if baseline.user_id not in self._baselines:
                    self._baselines[baseline.user_id] = []
                self._baselines[baseline.user_id].append(baseline)
            
            def get_baseline(self, user_id: str) -> Optional[UserBaseline]:
                if user_id in self._baselines and self._baselines[user_id]:
                    return self._baselines[user_id][-1]
                return None
            
            def get_baseline_history(
                self, user_id: str, limit: int = 10
            ) -> list[UserBaseline]:
                if user_id in self._baselines:
                    return self._baselines[user_id][-limit:]
                return []
        
        # Test the mock implementation
        repo = MockBaselineRepository()
        
        # Initially no baseline
        assert repo.get_baseline("user123") is None
        
        # Save a baseline
        baseline1 = UserBaseline(
            user_id="user123",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.0,
            sleep_std=0.5,
            activity_mean=7000,
            activity_std=1000,
            circadian_phase=-0.5,
            last_updated=datetime(2024, 1, 1, 12, 0),
            data_points=30,
        )
        repo.save_baseline(baseline1)
        
        # Should retrieve it
        retrieved = repo.get_baseline("user123")
        assert retrieved == baseline1
        
        # Save another baseline
        baseline2 = UserBaseline(
            user_id="user123",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=0.8,
            activity_mean=8000,
            activity_std=1500,
            circadian_phase=0.0,
            last_updated=datetime(2024, 1, 15, 12, 0),
            data_points=30,
        )
        repo.save_baseline(baseline2)
        
        # Should get the most recent
        assert repo.get_baseline("user123") == baseline2
        
        # Should get history
        history = repo.get_baseline_history("user123")
        assert len(history) == 2
        assert history[0] == baseline1
        assert history[1] == baseline2