"""
Activity Repository Interface

Domain repository for activity record persistence.
Part of the domain layer - defines interface only.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.value_objects.time_period import TimePeriod


class ActivityRepositoryInterface(ABC):
    """Abstract interface for activity data persistence."""

    @abstractmethod
    async def save(self, activity_record: ActivityRecord) -> None:
        """Persist an activity record."""
        pass

    @abstractmethod
    async def save_batch(self, activity_records: list[ActivityRecord]) -> None:
        """Persist multiple activity records efficiently."""
        pass

    @abstractmethod
    async def get_by_id(self, record_id: str) -> ActivityRecord | None:
        """Retrieve an activity record by ID."""
        pass

    @abstractmethod
    async def get_by_period(self, period: TimePeriod) -> list[ActivityRecord]:
        """Retrieve all activity records within a time period."""
        pass

    @abstractmethod
    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[ActivityRecord]:
        """Retrieve activity records within a date range."""
        pass

    @abstractmethod
    async def get_by_type(
        self, activity_type: ActivityType, period: TimePeriod
    ) -> list[ActivityRecord]:
        """Retrieve activity records of a specific type within a period."""
        pass

    @abstractmethod
    async def get_latest(self, limit: int = 10) -> list[ActivityRecord]:
        """Retrieve the most recent activity records."""
        pass

    @abstractmethod
    async def delete_by_period(self, period: TimePeriod) -> int:
        """Delete activity records within a time period. Returns count deleted."""
        pass