"""
Test File-based Sleep Repository

TDD for sleep data persistence implementation.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.value_objects.time_period import TimePeriod
from big_mood_detector.infrastructure.repositories.file_sleep_repository import (
    FileSleepRepository,
)


class TestFileSleepRepository:
    """Test file-based sleep repository implementation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def repository(self, temp_dir):
        """Create repository instance."""
        return FileSleepRepository(data_dir=temp_dir)

    @pytest.fixture
    def sample_sleep_record(self):
        """Create sample sleep record."""
        return SleepRecord(
            id="test-123",
            start_time=datetime(2024, 1, 1, 22, 0),
            end_time=datetime(2024, 1, 2, 6, 0),
            state=SleepState.ASLEEP,
            heart_rate_samples=[],
            motion_samples=[],
            sound_samples=[],
            metadata={},
        )

    async def test_repository_initialization(self, temp_dir):
        """Test repository can be initialized with directory."""
        repo = FileSleepRepository(data_dir=temp_dir)
        assert repo.data_dir == temp_dir
        assert (temp_dir / "sleep_records").exists()

    async def test_save_single_record(self, repository, sample_sleep_record):
        """Test saving a single sleep record."""
        await repository.save(sample_sleep_record)
        
        # Verify file was created
        expected_file = repository.data_dir / "sleep_records" / f"{sample_sleep_record.id}.json"
        assert expected_file.exists()

    async def test_save_and_retrieve_by_id(self, repository, sample_sleep_record):
        """Test saving and retrieving by ID."""
        await repository.save(sample_sleep_record)
        
        retrieved = await repository.get_by_id(sample_sleep_record.id)
        assert retrieved is not None
        assert retrieved.id == sample_sleep_record.id
        assert retrieved.start_time == sample_sleep_record.start_time
        assert retrieved.end_time == sample_sleep_record.end_time
        assert retrieved.state == sample_sleep_record.state

    async def test_get_by_id_not_found(self, repository):
        """Test retrieving non-existent record returns None."""
        result = await repository.get_by_id("non-existent")
        assert result is None

    async def test_save_batch(self, repository):
        """Test saving multiple records efficiently."""
        records = [
            SleepRecord(
                id=f"test-{i}",
                start_time=datetime(2024, 1, i, 22, 0),
                end_time=datetime(2024, 1, i + 1, 6, 0),
                state=SleepState.ASLEEP,
                heart_rate_samples=[],
                motion_samples=[],
                sound_samples=[],
                metadata={},
            )
            for i in range(1, 4)
        ]
        
        await repository.save_batch(records)
        
        # Verify all records saved
        for record in records:
            retrieved = await repository.get_by_id(record.id)
            assert retrieved is not None
            assert retrieved.id == record.id

    async def test_get_by_date_range(self, repository):
        """Test retrieving records by date range."""
        # Create records across several days
        records = []
        for i in range(5):
            record = SleepRecord(
                id=f"test-{i}",
                start_time=datetime(2024, 1, i + 1, 22, 0),
                end_time=datetime(2024, 1, i + 2, 6, 0),
                state=SleepState.ASLEEP,
                heart_rate_samples=[],
                motion_samples=[],
                sound_samples=[],
                metadata={},
            )
            records.append(record)
            await repository.save(record)
        
        # Get records from days 2-4
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 4, 23, 59, 59)
        result = await repository.get_by_date_range(start, end)
        
        assert len(result) == 3
        assert all(start <= r.start_time <= end for r in result)

    async def test_get_by_period(self, repository):
        """Test retrieving records by time period."""
        # Save some records
        records = []
        for i in range(3):
            record = SleepRecord(
                id=f"test-{i}",
                start_time=datetime(2024, 1, i + 1, 22, 0),
                end_time=datetime(2024, 1, i + 2, 6, 0),
                state=SleepState.ASLEEP,
                heart_rate_samples=[],
                motion_samples=[],
                sound_samples=[],
                metadata={},
            )
            records.append(record)
            await repository.save(record)
        
        # Get by period
        period = TimePeriod(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3, 23, 59, 59)
        )
        result = await repository.get_by_period(period)
        
        assert len(result) == 2
        assert all(period.contains(r.start_time) for r in result)

    async def test_get_latest(self, repository):
        """Test retrieving most recent records."""
        # Save records with different timestamps
        records = []
        base_time = datetime(2024, 1, 1)
        for i in range(5):
            record = SleepRecord(
                id=f"test-{i}",
                start_time=base_time + timedelta(days=i),
                end_time=base_time + timedelta(days=i, hours=8),
                state=SleepState.ASLEEP,
                heart_rate_samples=[],
                motion_samples=[],
                sound_samples=[],
                metadata={},
            )
            records.append(record)
            await repository.save(record)
        
        # Get latest 3
        result = await repository.get_latest(limit=3)
        
        assert len(result) == 3
        # Should be sorted by start_time descending
        assert result[0].id == "test-4"
        assert result[1].id == "test-3"
        assert result[2].id == "test-2"

    async def test_delete_by_period(self, repository):
        """Test deleting records within a period."""
        # Save some records
        records = []
        for i in range(5):
            record = SleepRecord(
                id=f"test-{i}",
                start_time=datetime(2024, 1, i + 1, 22, 0),
                end_time=datetime(2024, 1, i + 2, 6, 0),
                state=SleepState.ASLEEP,
                heart_rate_samples=[],
                motion_samples=[],
                sound_samples=[],
                metadata={},
            )
            records.append(record)
            await repository.save(record)
        
        # Delete records from days 2-3
        period = TimePeriod(
            start=datetime(2024, 1, 2),
            end=datetime(2024, 1, 3, 23, 59, 59)
        )
        deleted_count = await repository.delete_by_period(period)
        
        assert deleted_count == 2
        
        # Verify correct records were deleted
        assert await repository.get_by_id("test-0") is not None
        assert await repository.get_by_id("test-1") is None  # Deleted
        assert await repository.get_by_id("test-2") is None  # Deleted
        assert await repository.get_by_id("test-3") is not None
        assert await repository.get_by_id("test-4") is not None

    async def test_concurrent_access(self, repository):
        """Test repository handles concurrent access safely."""
        import asyncio
        
        # Create multiple records concurrently
        async def save_record(i):
            record = SleepRecord(
                id=f"concurrent-{i}",
                start_time=datetime(2024, 1, 1, 22, 0),
                end_time=datetime(2024, 1, 2, 6, 0),
                state=SleepState.ASLEEP,
                heart_rate_samples=[],
                motion_samples=[],
                sound_samples=[],
                metadata={"index": i},
            )
            await repository.save(record)
        
        # Save 10 records concurrently
        await asyncio.gather(*[save_record(i) for i in range(10)])
        
        # Verify all saved correctly
        for i in range(10):
            record = await repository.get_by_id(f"concurrent-{i}")
            assert record is not None
            assert record.metadata["index"] == i

    async def test_data_persistence(self, temp_dir, sample_sleep_record):
        """Test data persists across repository instances."""
        # Save with first instance
        repo1 = FileSleepRepository(data_dir=temp_dir)
        await repo1.save(sample_sleep_record)
        
        # Load with new instance
        repo2 = FileSleepRepository(data_dir=temp_dir)
        retrieved = await repo2.get_by_id(sample_sleep_record.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_sleep_record.id

    async def test_invalid_data_handling(self, repository):
        """Test repository handles invalid data gracefully."""
        # Try to save None
        with pytest.raises(ValueError):
            await repository.save(None)
        
        # Try to save invalid record
        with pytest.raises(ValueError):
            invalid_record = SleepRecord(
                id="",  # Invalid empty ID
                start_time=datetime(2024, 1, 1),
                end_time=datetime(2024, 1, 2),
                state=SleepState.ASLEEP,
                heart_rate_samples=[],
                motion_samples=[],
                sound_samples=[],
                metadata={},
            )
            await repository.save(invalid_record)