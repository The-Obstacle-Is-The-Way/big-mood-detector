"""
Repository Models

Data models for persistence layer that extend domain entities with storage metadata.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord


@dataclass
class StoredSleepRecord:
    """Sleep record with storage metadata."""

    id: str
    record: SleepRecord
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]

    @classmethod
    def from_domain(
        cls, record: SleepRecord, record_id: str | None = None
    ) -> "StoredSleepRecord":
        """Create stored record from domain entity."""
        if record_id is None:
            # Generate ID from record properties
            record_id = f"{record.source_name}_{record.start_date.isoformat()}_{record.end_date.isoformat()}"
            # Make it filesystem safe
            record_id = record_id.replace(":", "-").replace("/", "-")

        now = datetime.utcnow()
        return cls(
            id=record_id, record=record, created_at=now, updated_at=now, metadata={}
        )

    def to_domain(self) -> SleepRecord:
        """Extract domain entity."""
        return self.record


@dataclass
class StoredActivityRecord:
    """Activity record with storage metadata."""

    id: str
    record: ActivityRecord
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]

    @classmethod
    def from_domain(
        cls, record: ActivityRecord, record_id: str | None = None
    ) -> "StoredActivityRecord":
        """Create stored record from domain entity."""
        if record_id is None:
            # Generate ID from record properties
            record_id = f"{record.source_name}_{record.start_date.isoformat()}_{record.end_date.isoformat()}_{record.activity_type.value}"
            # Make it filesystem safe
            record_id = record_id.replace(":", "-").replace("/", "-")

        now = datetime.utcnow()
        return cls(
            id=record_id, record=record, created_at=now, updated_at=now, metadata={}
        )

    def to_domain(self) -> ActivityRecord:
        """Extract domain entity."""
        return self.record


@dataclass
class StoredHeartRateRecord:
    """Heart rate record with storage metadata."""

    id: str
    record: HeartRateRecord
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]

    @classmethod
    def from_domain(
        cls, record: HeartRateRecord, record_id: str | None = None
    ) -> "StoredHeartRateRecord":
        """Create stored record from domain entity."""
        if record_id is None:
            # Generate ID from record properties
            record_id = f"{record.source_name}_{record.timestamp.isoformat()}_{record.metric_type.value}"
            # Make it filesystem safe
            record_id = record_id.replace(":", "-").replace("/", "-")

        now = datetime.utcnow()
        return cls(
            id=record_id, record=record, created_at=now, updated_at=now, metadata={}
        )

    def to_domain(self) -> HeartRateRecord:
        """Extract domain entity."""
        return self.record
