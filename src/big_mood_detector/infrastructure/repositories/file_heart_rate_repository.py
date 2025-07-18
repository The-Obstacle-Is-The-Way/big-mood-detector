"""
File-based Heart Rate Repository Implementation

Concrete implementation of heart rate repository using JSON files.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
    MotionContext,
)
from big_mood_detector.domain.repositories.heart_rate_repository import (
    HeartRateRepositoryInterface,
)
from big_mood_detector.domain.value_objects.time_period import TimePeriod
import logging

from big_mood_detector.infrastructure.logging import get_module_logger
from big_mood_detector.infrastructure.repositories.models import StoredHeartRateRecord

logger = get_module_logger(__name__)


class FileHeartRateRepository(HeartRateRepositoryInterface):
    """File-based implementation of heart rate repository."""

    def __init__(self, data_dir: Path):
        """Initialize repository with data directory."""
        self.data_dir = Path(data_dir)
        self.records_dir = self.data_dir / "heart_rate_records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info("file_repository_initialized", data_dir=str(data_dir))

    async def save(self, heart_rate_record: HeartRateRecord) -> None:
        """Persist a heart rate record."""
        if heart_rate_record is None:
            raise ValueError("Cannot save None record")

        # Wrap in stored record
        stored = StoredHeartRateRecord.from_domain(heart_rate_record)

        async with self._lock:
            file_path = self.records_dir / f"{stored.id}.json"
            data = self._serialize_stored_record(stored)

            # Write atomically
            temp_path = file_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2, default=str))
            temp_path.replace(file_path)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("record_saved", record_id=stored.id)

    async def save_batch(self, heart_rate_records: list[HeartRateRecord]) -> None:
        """Persist multiple heart rate records efficiently."""
        # Use gather for concurrent saves
        await asyncio.gather(*[self.save(record) for record in heart_rate_records])
        logger.info("batch_saved", count=len(heart_rate_records))

    async def get_by_id(self, record_id: str) -> HeartRateRecord | None:
        """Retrieve a heart rate record by ID."""
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

    async def get_by_period(self, period: TimePeriod) -> list[HeartRateRecord]:
        """Retrieve all heart rate records within a time period."""
        return await self.get_by_date_range(period.start, period.end)

    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[HeartRateRecord]:
        """Retrieve heart rate records within a date range."""
        records = []

        # Read all record files
        for file_path in self.records_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                stored = self._deserialize_stored_record(data)
                record = stored.to_domain()

                # Check if record falls within date range
                if start_date <= record.timestamp <= end_date:
                    records.append(record)
            except Exception as e:
                logger.error(
                    "failed_to_load_record_file", file=str(file_path), error=str(e)
                )
                continue

        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp)
        return records

    async def get_by_metric_type(
        self, metric_type: HeartMetricType, period: TimePeriod
    ) -> list[HeartRateRecord]:
        """Retrieve heart rate records of a specific metric type within a period."""
        all_records = await self.get_by_period(period)
        return [r for r in all_records if r.metric_type == metric_type]

    async def get_latest(self, limit: int = 10) -> list[HeartRateRecord]:
        """Retrieve the most recent heart rate records."""
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

        # Sort by timestamp descending and take limit
        all_records.sort(key=lambda s: s.record.timestamp, reverse=True)
        return [s.to_domain() for s in all_records[:limit]]

    async def get_clinically_significant(
        self, period: TimePeriod
    ) -> list[HeartRateRecord]:
        """Retrieve only clinically significant measurements within a period."""
        all_records = await self.get_by_period(period)
        return [r for r in all_records if r.is_clinically_significant]

    async def delete_by_period(self, period: TimePeriod) -> int:
        """Delete heart rate records within a time period. Returns count deleted."""
        deleted_count = 0

        async with self._lock:
            for file_path in self.records_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text())
                    stored = self._deserialize_stored_record(data)
                    record = stored.to_domain()

                    # Check if record falls within period
                    if period.contains(record.timestamp):
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

    def _serialize_stored_record(self, stored: StoredHeartRateRecord) -> dict[str, Any]:
        """Serialize stored record to JSON-compatible dict."""
        record = stored.record
        return {
            "id": stored.id,
            "created_at": stored.created_at.isoformat(),
            "updated_at": stored.updated_at.isoformat(),
            "metadata": stored.metadata,
            "record": {
                "source_name": record.source_name,
                "timestamp": record.timestamp.isoformat(),
                "metric_type": record.metric_type.value,
                "value": record.value,
                "unit": record.unit,
                "motion_context": record.motion_context.value,
            },
        }

    def _deserialize_stored_record(self, data: dict[str, Any]) -> StoredHeartRateRecord:
        """Deserialize JSON dict to stored record."""
        record_data = data["record"]
        record = HeartRateRecord(
            source_name=record_data["source_name"],
            timestamp=datetime.fromisoformat(record_data["timestamp"]),
            metric_type=HeartMetricType(record_data["metric_type"]),
            value=float(record_data["value"]),
            unit=record_data["unit"],
            motion_context=MotionContext.from_string(record_data.get("motion_context")),
        )

        return StoredHeartRateRecord(
            id=data["id"],
            record=record,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )
