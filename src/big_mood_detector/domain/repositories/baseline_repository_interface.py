"""
Baseline Repository Interface

Domain interface for baseline persistence.
Following Uncle Bob's Clean Architecture - this interface
belongs in the domain layer and implementations will be in infrastructure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime


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
    # Heart rate baselines (NEW)
    heart_rate_mean: float = 70.0  # Default resting HR
    heart_rate_std: float = 10.0
    hrv_mean: float = 50.0  # Default HRV
    hrv_std: float = 15.0
    last_updated: datetime | None = None  # Will be set in __post_init__
    data_points: int = 30  # Number of days used in calculation

    def __post_init__(self) -> None:
        """Set default values for datetime fields."""
        if self.last_updated is None:
            object.__setattr__(self, 'last_updated', datetime.now())


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
    def get_baseline(self, user_id: str) -> UserBaseline | None:
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
