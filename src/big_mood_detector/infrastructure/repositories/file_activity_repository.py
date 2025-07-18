"""
File-based Activity Repository Implementation

Concrete implementation of activity repository using JSON files.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.repositories.activity_repository import (
    ActivityRepositoryInterface,
)
from big_mood_detector.domain.value_objects.time_period import TimePeriod
from big_mood_detector.infrastructure.logging import get_module_logger
from big_mood_detector.infrastructure.repositories.models import StoredActivityRecord

logger = get_module_logger(__name__)


class FileActivityRepository(ActivityRepositoryInterface):
    """File-based implementation of activity repository."""

    def __init__(self, data_dir: Path):
        """Initialize repository with data directory."""
        self.data_dir = Path(data_dir)
        self.records_dir = self.data_dir / "activity_records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info("file_repository_initialized", data_dir=str(data_dir))

    async def save(self, activity_record: ActivityRecord) -> None:
        """Persist an activity record."""
        if activity_record is None:
            raise ValueError("Cannot save None record")

        # Wrap in stored record
        stored = StoredActivityRecord.from_domain(activity_record)

        async with self._lock:
            file_path = self.records_dir / f"{stored.id}.json"
            data = self._serialize_stored_record(stored)

            # Write atomically
            temp_path = file_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2, default=str))
            temp_path.replace(file_path)

            logger.debug("record_saved", record_id=stored.id)

    async def save_batch(self, activity_records: list[ActivityRecord]) -> None:
        """Persist multiple activity records efficiently."""
        # Use gather for concurrent saves
        await asyncio.gather(*[self.save(record) for record in activity_records])
        logger.info("batch_saved", count=len(activity_records))

    async def get_by_id(self, record_id: str) -> ActivityRecord | None:
        """Retrieve an activity record by ID."""
        file_path = self.records_dir / f"{record_id}.json"

        if not file_path.exists():
            return None

        try:
            data = json.loads(file_path.read_text())
            stored = self._deserialize_stored_record(data)
            return stored.to_domain()
        except Exception as e:
            logger.error("failed_to_load_record", record_id=record_id, error=str(e))
            return None

    async def get_by_period(self, period: TimePeriod) -> list[ActivityRecord]:
        """Retrieve all activity records within a time period."""
        return await self.get_by_date_range(period.start, period.end)

    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[ActivityRecord]:
        """Retrieve activity records within a date range."""
        records = []

        # Read all record files
        for file_path in self.records_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                stored = self._deserialize_stored_record(data)
                record = stored.to_domain()

                # Check if record falls within date range
                if start_date <= record.start_date <= end_date:
                    records.append(record)
            except Exception as e:
                logger.error(
                    "failed_to_load_record_file", file=str(file_path), error=str(e)
                )
                continue

        # Sort by start time
        records.sort(key=lambda r: r.start_date)
        return records

    async def get_by_type(
        self, activity_type: ActivityType, period: TimePeriod
    ) -> list[ActivityRecord]:
        """Retrieve activity records of a specific type within a period."""
        all_records = await self.get_by_period(period)
        return [r for r in all_records if r.activity_type == activity_type]

    async def get_latest(self, limit: int = 10) -> list[ActivityRecord]:
        """Retrieve the most recent activity records."""
        all_records = []

        # Read all records
        for file_path in self.records_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                stored = self._deserialize_stored_record(data)
                all_records.append(stored)
            except Exception as e:
                logger.error(
                    "failed_to_load_record_file", file=str(file_path), error=str(e)
                )
                continue

        # Sort by start time descending and take limit
        all_records.sort(key=lambda s: s.record.start_date, reverse=True)
        return [s.to_domain() for s in all_records[:limit]]

    async def delete_by_period(self, period: TimePeriod) -> int:
        """Delete activity records within a time period. Returns count deleted."""
        deleted_count = 0

        async with self._lock:
            for file_path in self.records_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text())
                    stored = self._deserialize_stored_record(data)
                    record = stored.to_domain()

                    # Check if record falls within period
                    if period.contains(record.start_date):
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug("record_deleted", record_id=stored.id)
                except Exception as e:
                    logger.error(
                        "failed_to_process_record_file",
                        file=str(file_path),
                        error=str(e),
                    )
                    continue

        logger.info(
            "records_deleted_by_period", count=deleted_count, period=str(period)
        )
        return deleted_count

    def _serialize_stored_record(self, stored: StoredActivityRecord) -> dict[str, Any]:
        """Serialize stored record to JSON-compatible dict."""
        record = stored.record
        return {
            "id": stored.id,
            "created_at": stored.created_at.isoformat(),
            "updated_at": stored.updated_at.isoformat(),
            "metadata": stored.metadata,
            "record": {
                "source_name": record.source_name,
                "start_date": record.start_date.isoformat(),
                "end_date": record.end_date.isoformat(),
                "activity_type": record.activity_type.value,
                "value": record.value,
                "unit": record.unit,
            },
        }

    def _deserialize_stored_record(self, data: dict[str, Any]) -> StoredActivityRecord:
        """Deserialize JSON dict to stored record."""
        record_data = data["record"]
        record = ActivityRecord(
            source_name=record_data["source_name"],
            start_date=datetime.fromisoformat(record_data["start_date"]),
            end_date=datetime.fromisoformat(record_data["end_date"]),
            activity_type=ActivityType(record_data["activity_type"]),
            value=float(record_data["value"]),
            unit=record_data["unit"],
        )

        return StoredActivityRecord(
            id=data["id"],
            record=record,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
