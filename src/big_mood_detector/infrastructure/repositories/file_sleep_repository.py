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
from big_mood_detector.infrastructure.logging import get_module_logger

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
        
        if not sleep_record.id:
            raise ValueError("Sleep record must have an ID")
        
        async with self._lock:
            file_path = self.records_dir / f"{sleep_record.id}.json"
            data = self._serialize_record(sleep_record)
            
            # Write atomically
            temp_path = file_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2, default=str))
            temp_path.replace(file_path)
            
            logger.debug("record_saved", record_id=sleep_record.id)

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
            return self._deserialize_record(data)
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
                record = self._deserialize_record(data)
                
                # Check if record falls within date range
                if start_date <= record.start_time <= end_date:
                    records.append(record)
            except Exception as e:
                logger.error("failed_to_load_record_file", file=str(file_path), error=str(e))
                continue
        
        # Sort by start time
        records.sort(key=lambda r: r.start_time)
        return records

    async def get_latest(self, limit: int = 10) -> list[SleepRecord]:
        """Retrieve the most recent sleep records."""
        all_records = []
        
        # Read all records
        for file_path in self.records_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                record = self._deserialize_record(data)
                all_records.append(record)
            except Exception as e:
                logger.error("failed_to_load_record_file", file=str(file_path), error=str(e))
                continue
        
        # Sort by start time descending and take limit
        all_records.sort(key=lambda r: r.start_time, reverse=True)
        return all_records[:limit]

    async def delete_by_period(self, period: TimePeriod) -> int:
        """Delete sleep records within a time period. Returns count deleted."""
        deleted_count = 0
        
        async with self._lock:
            for file_path in self.records_dir.glob("*.json"):
                try:
                    data = json.loads(file_path.read_text())
                    record = self._deserialize_record(data)
                    
                    # Check if record falls within period
                    if period.contains(record.start_time):
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug("record_deleted", record_id=record.id)
                except Exception as e:
                    logger.error("failed_to_process_record_file", file=str(file_path), error=str(e))
                    continue
        
        logger.info("records_deleted_by_period", count=deleted_count, period=str(period))
        return deleted_count

    def _serialize_record(self, record: SleepRecord) -> dict[str, Any]:
        """Serialize sleep record to JSON-compatible dict."""
        return {
            "id": record.id,
            "start_time": record.start_time.isoformat(),
            "end_time": record.end_time.isoformat(),
            "state": record.state.value,
            "heart_rate_samples": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "value": s.value,
                    "confidence": s.confidence,
                }
                for s in record.heart_rate_samples
            ],
            "motion_samples": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "intensity": s.intensity,
                    "confidence": s.confidence,
                }
                for s in record.motion_samples
            ],
            "sound_samples": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "decibels": s.decibels,
                    "classification": s.classification,
                }
                for s in record.sound_samples
            ],
            "metadata": record.metadata,
        }

    def _deserialize_record(self, data: dict[str, Any]) -> SleepRecord:
        """Deserialize JSON dict to sleep record."""
        from big_mood_detector.domain.entities.biometric_sample import (
            HeartRateSample,
            MotionSample,
            SoundSample,
        )
        
        return SleepRecord(
            id=data["id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            state=SleepState(data["state"]),
            heart_rate_samples=[
                HeartRateSample(
                    timestamp=datetime.fromisoformat(s["timestamp"]),
                    value=s["value"],
                    confidence=s.get("confidence", 1.0),
                )
                for s in data.get("heart_rate_samples", [])
            ],
            motion_samples=[
                MotionSample(
                    timestamp=datetime.fromisoformat(s["timestamp"]),
                    intensity=s["intensity"],
                    confidence=s.get("confidence", 1.0),
                )
                for s in data.get("motion_samples", [])
            ],
            sound_samples=[
                SoundSample(
                    timestamp=datetime.fromisoformat(s["timestamp"]),
                    decibels=s["decibels"],
                    classification=s.get("classification"),
                )
                for s in data.get("sound_samples", [])
            ],
            metadata=data.get("metadata", {}),
        )