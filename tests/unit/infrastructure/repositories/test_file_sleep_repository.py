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
from big_mood_detector.infrastructure.repositories.models import StoredSleepRecord


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
            source_name="test-source",
            start_date=datetime(2024, 1, 1, 22, 0),
            end_date=datetime(2024, 1, 2, 6, 0),
            state=SleepState.ASLEEP,
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
        stored = StoredSleepRecord.from_domain(sample_sleep_record)
        expected_file = repository.data_dir / "sleep_records" / f"{stored.id}.json"
        assert expected_file.exists()

    async def test_save_and_retrieve_by_id(self, repository, sample_sleep_record):
        """Test saving and retrieving by ID."""
        await repository.save(sample_sleep_record)

        # Generate the ID that would be created
        stored = StoredSleepRecord.from_domain(sample_sleep_record)

        retrieved = await repository.get_by_id(stored.id)
        assert retrieved is not None
        assert retrieved.source_name == sample_sleep_record.source_name
        assert retrieved.start_date == sample_sleep_record.start_date
        assert retrieved.end_date == sample_sleep_record.end_date
        assert retrieved.state == sample_sleep_record.state

    async def test_get_by_id_not_found(self, repository):
        """Test retrieving non-existent record returns None."""
        result = await repository.get_by_id("non-existent")
        assert result is None

    async def test_save_batch(self, repository):
        """Test saving multiple records efficiently."""
        records = [
            SleepRecord(
                source_name=f"test-source-{i}",
                start_date=datetime(2024, 1, i, 22, 0),
                end_date=datetime(2024, 1, i + 1, 6, 0),
                state=SleepState.ASLEEP,
            )
            for i in range(1, 4)
        ]

        await repository.save_batch(records)

        # Verify all records saved
        for record in records:
            stored = StoredSleepRecord.from_domain(record)
            retrieved = await repository.get_by_id(stored.id)
            assert retrieved is not None
            assert retrieved.source_name == record.source_name

    async def test_get_by_date_range(self, repository):
        """Test retrieving records by date range."""
        # Create records across several days
        records = []
        for i in range(5):
            record = SleepRecord(
                source_name=f"test-{i}",
                start_date=datetime(2024, 1, i + 1, 22, 0),
                end_date=datetime(2024, 1, i + 2, 6, 0),
                state=SleepState.ASLEEP,
            )
            records.append(record)
            await repository.save(record)

        # Get records from days 2-4
        start = datetime(2024, 1, 2)
        end = datetime(2024, 1, 4, 23, 59, 59)
        result = await repository.get_by_date_range(start, end)

        assert len(result) == 3
        assert all(start <= r.start_date <= end for r in result)

    async def test_get_by_period(self, repository):
        """Test retrieving records by time period."""
        # Save some records
        records = []
        for i in range(3):
            record = SleepRecord(
                source_name=f"test-{i}",
                start_date=datetime(2024, 1, i + 1, 22, 0),
                end_date=datetime(2024, 1, i + 2, 6, 0),
                state=SleepState.ASLEEP,
            )
            records.append(record)
            await repository.save(record)

        # Get by period
        period = TimePeriod(
            start=datetime(2024, 1, 2), end=datetime(2024, 1, 3, 23, 59, 59)
        )
        result = await repository.get_by_period(period)

        assert len(result) == 2
        assert all(period.contains(r.start_date) for r in result)

    async def test_get_latest(self, repository):
        """Test retrieving most recent records."""
        # Save records with different timestamps
        records = []
        base_time = datetime(2024, 1, 1)
        for i in range(5):
            record = SleepRecord(
                source_name=f"test-{i}",
                start_date=base_time + timedelta(days=i),
                end_date=base_time + timedelta(days=i, hours=8),
                state=SleepState.ASLEEP,
            )
            records.append(record)
            await repository.save(record)

        # Get latest 3
        result = await repository.get_latest(limit=3)

        assert len(result) == 3
        # Should be sorted by start_date descending
        assert result[0].source_name == "test-4"
        assert result[1].source_name == "test-3"
        assert result[2].source_name == "test-2"

    async def test_delete_by_period(self, repository):
        """Test deleting records within a period."""
        # Save some records
        records = []
        for i in range(5):
            record = SleepRecord(
                source_name=f"test-{i}",
                start_date=datetime(2024, 1, i + 1, 22, 0),
                end_date=datetime(2024, 1, i + 2, 6, 0),
                state=SleepState.ASLEEP,
            )
            records.append(record)
            await repository.save(record)

        # Delete records from days 2-3
        period = TimePeriod(
            start=datetime(2024, 1, 2), end=datetime(2024, 1, 3, 23, 59, 59)
        )
        deleted_count = await repository.delete_by_period(period)

        assert deleted_count == 2

        # Verify correct records were deleted
        stored_0 = StoredSleepRecord.from_domain(records[0])
        stored_1 = StoredSleepRecord.from_domain(records[1])
        stored_2 = StoredSleepRecord.from_domain(records[2])
        stored_3 = StoredSleepRecord.from_domain(records[3])
        stored_4 = StoredSleepRecord.from_domain(records[4])

        assert await repository.get_by_id(stored_0.id) is not None
        assert await repository.get_by_id(stored_1.id) is None  # Deleted
        assert await repository.get_by_id(stored_2.id) is None  # Deleted
        assert await repository.get_by_id(stored_3.id) is not None
        assert await repository.get_by_id(stored_4.id) is not None

    async def test_concurrent_access(self, repository):
        """Test repository handles concurrent access safely."""
        import asyncio

        # Create multiple records concurrently
        async def save_record(i):
            record = SleepRecord(
                source_name=f"concurrent-{i}",
                start_date=datetime(2024, 1, 1, 22, 0) + timedelta(minutes=i),
                end_date=datetime(2024, 1, 2, 6, 0) + timedelta(minutes=i),
                state=SleepState.ASLEEP,
            )
            await repository.save(record)
            return StoredSleepRecord.from_domain(record)

        # Save 10 records concurrently
        stored_records = await asyncio.gather(*[save_record(i) for i in range(10)])

        # Verify all saved correctly
        for i, stored in enumerate(stored_records):
            record = await repository.get_by_id(stored.id)
            assert record is not None
            assert record.source_name == f"concurrent-{i}"

    async def test_data_persistence(self, temp_dir, sample_sleep_record):
        """Test data persists across repository instances."""
        # Save with first instance
        repo1 = FileSleepRepository(data_dir=temp_dir)
        await repo1.save(sample_sleep_record)

        # Get the stored ID
        stored = StoredSleepRecord.from_domain(sample_sleep_record)

        # Load with new instance
        repo2 = FileSleepRepository(data_dir=temp_dir)
        retrieved = await repo2.get_by_id(stored.id)

        assert retrieved is not None
        assert retrieved.source_name == sample_sleep_record.source_name

    async def test_invalid_data_handling(self, repository):
        """Test repository handles invalid data gracefully."""
        # Try to save None
        with pytest.raises(ValueError):
            await repository.save(None)
