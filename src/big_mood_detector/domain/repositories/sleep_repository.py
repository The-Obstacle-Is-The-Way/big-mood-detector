"""
Sleep Repository Interface

Abstract repository for sleep data persistence.
Following Repository Pattern and Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.value_objects.time_period import TimePeriod


class SleepRepositoryInterface(ABC):
    """
    Abstract interface for sleep data repository.

    Dependency Inversion: High-level domain doesn't depend on low-level storage.
    """

    @abstractmethod
    async def save(self, sleep_record: SleepRecord) -> None:
        """Persist a sleep record."""
        pass

    @abstractmethod
    async def save_batch(self, sleep_records: List[SleepRecord]) -> None:
        """Persist multiple sleep records efficiently."""
        pass

    @abstractmethod
    async def get_by_id(self, record_id: str) -> Optional[SleepRecord]:
        """Retrieve a sleep record by ID."""
        pass

    @abstractmethod
    async def get_by_period(self, period: TimePeriod) -> List[SleepRecord]:
        """Retrieve all sleep records within a time period."""
        pass

    @abstractmethod
    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[SleepRecord]:
        """Retrieve sleep records within a date range."""
        pass

    @abstractmethod
    async def get_latest(self, limit: int = 10) -> List[SleepRecord]:
        """Retrieve the most recent sleep records."""
        pass

    @abstractmethod
    async def delete_by_period(self, period: TimePeriod) -> int:
        """Delete sleep records within a time period. Returns count deleted."""
        pass
