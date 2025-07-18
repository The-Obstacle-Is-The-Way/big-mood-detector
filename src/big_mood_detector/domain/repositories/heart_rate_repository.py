"""
Heart Rate Repository Interface

Domain repository for heart rate record persistence.
Part of the domain layer - defines interface only.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
)
from big_mood_detector.domain.value_objects.time_period import TimePeriod


class HeartRateRepositoryInterface(ABC):
    """Abstract interface for heart rate data persistence."""

    @abstractmethod
    async def save(self, heart_rate_record: HeartRateRecord) -> None:
        """Persist a heart rate record."""
        pass

    @abstractmethod
    async def save_batch(self, heart_rate_records: list[HeartRateRecord]) -> None:
        """Persist multiple heart rate records efficiently."""
        pass

    @abstractmethod
    async def get_by_id(self, record_id: str) -> HeartRateRecord | None:
        """Retrieve a heart rate record by ID."""
        pass

    @abstractmethod
    async def get_by_period(self, period: TimePeriod) -> list[HeartRateRecord]:
        """Retrieve all heart rate records within a time period."""
        pass

    @abstractmethod
    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[HeartRateRecord]:
        """Retrieve heart rate records within a date range."""
        pass

    @abstractmethod
    async def get_by_metric_type(
        self, metric_type: HeartMetricType, period: TimePeriod
    ) -> list[HeartRateRecord]:
        """Retrieve heart rate records of a specific metric type within a period."""
        pass

    @abstractmethod
    async def get_latest(self, limit: int = 10) -> list[HeartRateRecord]:
        """Retrieve the most recent heart rate records."""
        pass

    @abstractmethod
    async def get_clinically_significant(
        self, period: TimePeriod
    ) -> list[HeartRateRecord]:
        """Retrieve only clinically significant measurements within a period."""
        pass

    @abstractmethod
    async def delete_by_period(self, period: TimePeriod) -> int:
        """Delete heart rate records within a time period. Returns count deleted."""
        pass
