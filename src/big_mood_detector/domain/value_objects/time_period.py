"""
Time Period Value Object

Immutable value object representing a time period with validation.
Following Domain-Driven Design principles.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass(frozen=True)
class TimePeriod:
    """
    Immutable time period with business logic.

    Value Object pattern - equality based on values, not identity.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        """Validate time period invariants."""
        if self.end <= self.start:
            raise ValueError(f"Invalid time period: {self.start} to {self.end}")

    @property
    def duration(self) -> timedelta:
        """Get duration as timedelta."""
        return self.end - self.start

    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return self.duration.total_seconds() / 3600

    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.duration.total_seconds() / 60

    def overlaps_with(self, other: "TimePeriod") -> bool:
        """Check if this period overlaps with another."""
        return not (self.end <= other.start or other.end <= self.start)

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within this period."""
        return self.start <= timestamp <= self.end

    def merge_with(self, other: "TimePeriod") -> Optional["TimePeriod"]:
        """
        Merge with another period if they overlap or are adjacent.

        Returns None if periods cannot be merged.
        """
        if not self.overlaps_with(other):
            # Check if adjacent (within 1 minute)
            gap = min(
                abs((self.end - other.start).total_seconds()),
                abs((other.end - self.start).total_seconds()),
            )
            if gap > 60:  # More than 1 minute gap
                return None

        return TimePeriod(
            start=min(self.start, other.start), end=max(self.end, other.end)
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.start.isoformat()} to {self.end.isoformat()} ({self.duration_hours:.1f}h)"
