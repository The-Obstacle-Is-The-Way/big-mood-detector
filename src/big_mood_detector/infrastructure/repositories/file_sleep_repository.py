"""
File-based Sleep Repository Implementation

Concrete implementation of sleep repository using JSON files.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.repositories.sleep_repository import (
    SleepRepositoryInterface,
)
from big_mood_detector.domain.value_objects.time_period import TimePeriod
import logging

from big_mood_detector.infrastructure.logging import get_module_logger
from big_mood_detector.infrastructure.repositories.models import StoredSleepRecord

logger = get_module_logger(__name__)


class FileSleepRepository(SleepRepositoryInterface):
    """File-based implementation of sleep repository."""

    def __init__(self, data_dir: Path):
        """Initialize repository with data directory."""
        self.data_dir = Path(data_dir)
        self.records_dir = self.data_dir / "sleep_records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info("file_repository_initialized", data_dir=str(data_dir))

    async def save(self, sleep_record: SleepRecord) -> None:
        """Persist a sleep record."""
        if sleep_record is None:
            raise ValueError("Cannot save None record")

        # Wrap in stored record
        stored = StoredSleepRecord.from_domain(sleep_record)

        async with self._lock:
            file_path = self.records_dir / f"{stored.id}.json"
            data = self._serialize_stored_record(stored)

            # Write atomically
            temp_path = file_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2, default=str))
            temp_path.replace(file_path)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("record_saved", record_id=stored.id)

    async def save_batch(self, sleep_records: list[SleepRecord]) -> None:
        """Persist multiple sleep records efficiently."""
        # Use gather for concurrent saves
        await asyncio.gather(*[self.save(record) for record in sleep_records])
        logger.info("batch_saved", count=len(sleep_records))

    async def get_by_id(self, record_id: str) -> SleepRecord | None:
        """Retrieve a sleep record by ID."""
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

    async def get_by_period(self, period: TimePeriod) -> list[SleepRecord]:
        """Retrieve all sleep records within a time period."""
        return await self.get_by_date_range(period.start, period.end)

    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[SleepRecord]:
        """Retrieve sleep records within a date range."""
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

    async def get_latest(self, limit: int = 10) -> list[SleepRecord]:
        """Retrieve the most recent sleep records."""
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
        """Delete sleep records within a time period. Returns count deleted."""
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
                        if logger.isEnabledFor(logging.DEBUG):
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

    def _serialize_stored_record(self, stored: StoredSleepRecord) -> dict[str, Any]:
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
                "state": record.state.value,
            },
        }

    def _deserialize_stored_record(self, data: dict[str, Any]) -> StoredSleepRecord:
        """Deserialize JSON dict to stored record."""
        record_data = data["record"]
        record = SleepRecord(
            source_name=record_data["source_name"],
            start_date=datetime.fromisoformat(record_data["start_date"]),
            end_date=datetime.fromisoformat(record_data["end_date"]),
            state=SleepState(record_data["state"]),
        )

        return StoredSleepRecord(
            id=data["id"],
            record=record,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
