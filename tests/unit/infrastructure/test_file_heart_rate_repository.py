"""
Test File-based Heart Rate Repository

TDD for heart rate data persistence implementation.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from big_mood_detector.domain.entities.heart_rate_record import (
    HeartRateRecord,
    HeartMetricType,
    MotionContext,
)
from big_mood_detector.domain.value_objects.time_period import TimePeriod
from big_mood_detector.infrastructure.repositories.file_heart_rate_repository import (
    FileHeartRateRepository,
)
from big_mood_detector.infrastructure.repositories.models import StoredHeartRateRecord


class TestFileHeartRateRepository:
    """Test file-based heart rate repository implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def repository(self, temp_dir):
        """Create repository instance."""
        return FileHeartRateRepository(data_dir=temp_dir)

    @pytest.fixture
    def sample_heart_rate_record(self):
        """Create sample heart rate record."""
        return HeartRateRecord(
            source_name="test-watch",
            timestamp=datetime(2024, 1, 1, 12, 30, 45),
            metric_type=HeartMetricType.HEART_RATE,
            value=75.0,
            unit="count/min",
            motion_context=MotionContext.SEDENTARY,
        )

    async def test_repository_initialization(self, temp_dir):
        """Test repository can be initialized with directory."""
        repo = FileHeartRateRepository(data_dir=temp_dir)
        assert repo.data_dir == temp_dir
        assert (temp_dir / "heart_rate_records").exists()

    async def test_save_single_record(self, repository, sample_heart_rate_record):
        """Test saving a single heart rate record."""
        await repository.save(sample_heart_rate_record)
        
        # Verify file was created
        stored = StoredHeartRateRecord.from_domain(sample_heart_rate_record)
        expected_file = repository.data_dir / "heart_rate_records" / f"{stored.id}.json"
        assert expected_file.exists()

    async def test_save_and_retrieve_by_id(self, repository, sample_heart_rate_record):
        """Test saving and retrieving by ID."""
        await repository.save(sample_heart_rate_record)
        
        # Generate the ID that would be created
        stored = StoredHeartRateRecord.from_domain(sample_heart_rate_record)
        
        retrieved = await repository.get_by_id(stored.id)
        assert retrieved is not None
        assert retrieved.source_name == sample_heart_rate_record.source_name
        assert retrieved.timestamp == sample_heart_rate_record.timestamp
        assert retrieved.metric_type == sample_heart_rate_record.metric_type
        assert retrieved.value == sample_heart_rate_record.value
        assert retrieved.unit == sample_heart_rate_record.unit
        assert retrieved.motion_context == sample_heart_rate_record.motion_context

    async def test_get_by_id_not_found(self, repository):
        """Test retrieving non-existent record returns None."""
        result = await repository.get_by_id("non-existent")
        assert result is None

    async def test_save_batch(self, repository):
        """Test saving multiple records efficiently."""
        records = [
            HeartRateRecord(
                source_name=f"device-{i}",
                timestamp=datetime(2024, 1, 1, 12, i, 0),
                metric_type=HeartMetricType.HEART_RATE,
                value=60.0 + i * 5,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            )
            for i in range(1, 4)
        ]
        
        await repository.save_batch(records)
        
        # Verify all records saved
        for record in records:
            stored = StoredHeartRateRecord.from_domain(record)
            retrieved = await repository.get_by_id(stored.id)
            assert retrieved is not None
            assert retrieved.source_name == record.source_name

    async def test_get_by_date_range(self, repository):
        """Test retrieving records by date range."""
        # Create records across several hours
        records = []
        for i in range(5):
            record = HeartRateRecord(
                source_name=f"test-{i}",
                timestamp=datetime(2024, 1, 1, 10 + i, 0, 0),
                metric_type=HeartMetricType.HEART_RATE,
                value=70.0 + i,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            )
            records.append(record)
            await repository.save(record)
        
        # Get records from hours 11-13
        start = datetime(2024, 1, 1, 11, 0, 0)
        end = datetime(2024, 1, 1, 13, 30, 0)
        result = await repository.get_by_date_range(start, end)
        
        assert len(result) == 3
        assert all(start <= r.timestamp <= end for r in result)

    async def test_get_by_period(self, repository):
        """Test retrieving records by time period."""
        # Save some records
        records = []
        for i in range(3):
            record = HeartRateRecord(
                source_name=f"test-{i}",
                timestamp=datetime(2024, 1, 1, 14, i * 15, 0),
                metric_type=HeartMetricType.HRV_SDNN,
                value=30.0 + i * 5,
                unit="ms",
                motion_context=MotionContext.SEDENTARY,
            )
            records.append(record)
            await repository.save(record)
        
        # Get by period
        period = TimePeriod(
            start=datetime(2024, 1, 1, 14, 0, 0),
            end=datetime(2024, 1, 1, 14, 25, 0)  # Exclude the 14:30 record
        )
        result = await repository.get_by_period(period)
        
        assert len(result) == 2
        assert all(period.contains(r.timestamp) for r in result)

    async def test_get_by_metric_type(self, repository):
        """Test retrieving records by metric type."""
        # Save records of different types
        hr_record = HeartRateRecord(
            source_name="test",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metric_type=HeartMetricType.HEART_RATE,
            value=72.0,
            unit="count/min",
            motion_context=MotionContext.SEDENTARY,
        )
        hrv_record = HeartRateRecord(
            source_name="test",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            metric_type=HeartMetricType.HRV_SDNN,
            value=35.0,
            unit="ms",
            motion_context=MotionContext.SEDENTARY,
        )
        
        await repository.save(hr_record)
        await repository.save(hrv_record)
        
        # Get only HRV records
        period = TimePeriod(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 23, 59, 59)
        )
        result = await repository.get_by_metric_type(HeartMetricType.HRV_SDNN, period)
        
        assert len(result) == 1
        assert result[0].metric_type == HeartMetricType.HRV_SDNN

    async def test_get_latest(self, repository):
        """Test retrieving most recent records."""
        # Save records with different timestamps
        records = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(5):
            record = HeartRateRecord(
                source_name=f"test-{i}",
                timestamp=base_time + timedelta(minutes=i),
                metric_type=HeartMetricType.HEART_RATE,
                value=65.0 + i,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            )
            records.append(record)
            await repository.save(record)
        
        # Get latest 3
        result = await repository.get_latest(limit=3)
        
        assert len(result) == 3
        # Should be sorted by timestamp descending
        assert result[0].source_name == "test-4"
        assert result[1].source_name == "test-3"
        assert result[2].source_name == "test-2"

    async def test_get_clinically_significant(self, repository):
        """Test retrieving only clinically significant measurements."""
        # Save a mix of normal and clinically significant records
        records = [
            # Normal heart rate
            HeartRateRecord(
                source_name="test",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                metric_type=HeartMetricType.HEART_RATE,
                value=75.0,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            ),
            # High heart rate (tachycardia)
            HeartRateRecord(
                source_name="test",
                timestamp=datetime(2024, 1, 1, 12, 1, 0),
                metric_type=HeartMetricType.HEART_RATE,
                value=105.0,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            ),
            # Low HRV
            HeartRateRecord(
                source_name="test",
                timestamp=datetime(2024, 1, 1, 12, 2, 0),
                metric_type=HeartMetricType.HRV_SDNN,
                value=15.0,
                unit="ms",
                motion_context=MotionContext.SEDENTARY,
            ),
            # Normal HRV
            HeartRateRecord(
                source_name="test",
                timestamp=datetime(2024, 1, 1, 12, 3, 0),
                metric_type=HeartMetricType.HRV_SDNN,
                value=40.0,
                unit="ms",
                motion_context=MotionContext.SEDENTARY,
            ),
        ]
        
        for record in records:
            await repository.save(record)
        
        # Get only clinically significant records
        period = TimePeriod(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1, 23, 59, 59)
        )
        result = await repository.get_clinically_significant(period)
        
        assert len(result) == 2
        assert all(r.is_clinically_significant for r in result)

    async def test_delete_by_period(self, repository):
        """Test deleting records within a period."""
        # Save some records
        records = []
        for i in range(5):
            record = HeartRateRecord(
                source_name=f"test-{i}",
                timestamp=datetime(2024, 1, 1, 12, i * 10, 0),
                metric_type=HeartMetricType.HEART_RATE,
                value=70.0 + i,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            )
            records.append(record)
            await repository.save(record)
        
        # Delete records from minutes 10-30
        period = TimePeriod(
            start=datetime(2024, 1, 1, 12, 10, 0),
            end=datetime(2024, 1, 1, 12, 30, 0)
        )
        deleted_count = await repository.delete_by_period(period)
        
        assert deleted_count == 3
        
        # Verify correct records were deleted
        stored_0 = StoredHeartRateRecord.from_domain(records[0])
        stored_1 = StoredHeartRateRecord.from_domain(records[1])
        stored_2 = StoredHeartRateRecord.from_domain(records[2])
        stored_3 = StoredHeartRateRecord.from_domain(records[3])
        stored_4 = StoredHeartRateRecord.from_domain(records[4])
        
        assert await repository.get_by_id(stored_0.id) is not None
        assert await repository.get_by_id(stored_1.id) is None  # Deleted
        assert await repository.get_by_id(stored_2.id) is None  # Deleted
        assert await repository.get_by_id(stored_3.id) is None  # Deleted
        assert await repository.get_by_id(stored_4.id) is not None

    async def test_concurrent_access(self, repository):
        """Test repository handles concurrent access safely."""
        import asyncio
        
        # Create multiple records concurrently
        async def save_record(i):
            record = HeartRateRecord(
                source_name=f"concurrent-{i}",
                timestamp=datetime(2024, 1, 1, 12, 0, 0) + timedelta(seconds=i),
                metric_type=HeartMetricType.HEART_RATE,
                value=60.0 + i,
                unit="count/min",
                motion_context=MotionContext.SEDENTARY,
            )
            await repository.save(record)
            return StoredHeartRateRecord.from_domain(record)
        
        # Save 10 records concurrently
        stored_records = await asyncio.gather(*[save_record(i) for i in range(10)])
        
        # Verify all saved correctly
        for i, stored in enumerate(stored_records):
            record = await repository.get_by_id(stored.id)
            assert record is not None
            assert record.source_name == f"concurrent-{i}"

    async def test_data_persistence(self, temp_dir, sample_heart_rate_record):
        """Test data persists across repository instances."""
        # Save with first instance
        repo1 = FileHeartRateRepository(data_dir=temp_dir)
        await repo1.save(sample_heart_rate_record)
        
        # Get the stored ID
        stored = StoredHeartRateRecord.from_domain(sample_heart_rate_record)
        
        # Load with new instance
        repo2 = FileHeartRateRepository(data_dir=temp_dir)
        retrieved = await repo2.get_by_id(stored.id)
        
        assert retrieved is not None
        assert retrieved.source_name == sample_heart_rate_record.source_name

    async def test_invalid_data_handling(self, repository):
        """Test repository handles invalid data gracefully."""
        # Try to save None
        with pytest.raises(ValueError):
            await repository.save(None)