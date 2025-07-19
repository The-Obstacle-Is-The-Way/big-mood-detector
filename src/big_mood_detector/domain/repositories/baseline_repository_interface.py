"""
Baseline Repository Interface

Domain interface for baseline persistence.
Following Uncle Bob's Clean Architecture - this interface
belongs in the domain layer and implementations will be in infrastructure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True)
class UserBaseline:
    """
    Value object representing a user's baseline measurements.
    
    Baselines are statistical summaries of a user's typical patterns,
    used for Z-score normalization in mood prediction.
    """
    user_id: str
    baseline_date: date
    sleep_mean: float
    sleep_std: float
    activity_mean: float
    activity_std: float
    circadian_phase: float
    last_updated: datetime
    data_points: int  # Number of days used in calculation


class BaselineRepositoryInterface(ABC):
    """
    Repository interface for persisting user baselines.
    
    Following Interface Segregation Principle - minimal interface
    with only essential operations for baseline storage and retrieval.
    """
    
    @abstractmethod
    def save_baseline(self, baseline: UserBaseline) -> None:
        """
        Save or update a user's baseline.
        
        Args:
            baseline: The baseline data to persist
        """
        pass
    
    @abstractmethod
    def get_baseline(self, user_id: str) -> Optional[UserBaseline]:
        """
        Retrieve the most recent baseline for a user.
        
        Args:
            user_id: The user identifier
            
        Returns:
            The most recent baseline or None if not found
        """
        pass
    
    @abstractmethod
    def get_baseline_history(
        self, user_id: str, limit: int = 10
    ) -> list[UserBaseline]:
        """
        Get historical baselines for trend analysis.
        
        Args:
            user_id: The user identifier
            limit: Maximum number of baselines to return
            
        Returns:
            List of baselines ordered by date (oldest first)
        """
        pass