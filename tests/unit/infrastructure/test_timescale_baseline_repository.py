"""
TimescaleDB Baseline Repository Tests

Following best practices for testing repositories:
1. Use contract tests to ensure fake and real implementations behave identically
2. Don't overmock - test against real behavior
3. Use fast fake for development, real DB for integration validation
"""

import shutil
import time
from datetime import date, datetime
from pathlib import Path

import pytest

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


class BaselineRepositoryContract:
    """
    Contract test for all BaselineRepository implementations.

    Both FileBaselineRepository and TimescaleBaselineRepository
    must pass these tests to ensure behavioral compatibility.
    """

    def get_repository(self) -> BaselineRepositoryInterface:
        """Override in subclasses to provide repository implementation"""
        raise NotImplementedError

    def get_sample_baseline(self) -> UserBaseline:
        """Standard test baseline"""
        return UserBaseline(
            user_id="test_user_123",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 15, 10, 30),
            data_points=30,
        )

    def test_save_and_retrieve_baseline_with_hr_hrv(self):
        """Test save and retrieve with optional HR/HRV fields"""
        repository = self.get_repository()

        # Create baseline with HR/HRV data
        baseline = UserBaseline(
            user_id="test_user_hr",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            heart_rate_mean=65.0,
            heart_rate_std=5.0,
            hrv_mean=55.0,
            hrv_std=8.0,
            last_updated=datetime(2024, 1, 15, 10, 30),
            data_points=30,
        )

        # Save and retrieve
        repository.save_baseline(baseline)
        retrieved = repository.get_baseline(baseline.user_id)

        # Verify HR/HRV fields
        assert retrieved is not None
        assert retrieved.heart_rate_mean == 65.0
        assert retrieved.heart_rate_std == 5.0
        assert retrieved.hrv_mean == 55.0
        assert retrieved.hrv_std == 8.0

    def test_save_and_retrieve_baseline(self):
        """Test basic save and retrieve operations"""
        repository = self.get_repository()
        baseline = self.get_sample_baseline()

        # Save baseline
        repository.save_baseline(baseline)

        # Retrieve baseline
        retrieved = repository.get_baseline(baseline.user_id)

        # Verify values
        assert retrieved is not None
        assert retrieved.user_id == baseline.user_id
        assert retrieved.baseline_date == baseline.baseline_date
        assert retrieved.sleep_mean == baseline.sleep_mean
        assert retrieved.sleep_std == baseline.sleep_std
        assert retrieved.activity_mean == baseline.activity_mean
        assert retrieved.activity_std == baseline.activity_std
        assert retrieved.circadian_phase == baseline.circadian_phase
        assert retrieved.data_points == baseline.data_points

    def test_get_nonexistent_baseline_returns_none(self):
        """Test retrieving non-existent baseline returns None"""
        repository = self.get_repository()

        result = repository.get_baseline("nonexistent_user")

        assert result is None

    def test_get_baseline_history_empty_for_new_user(self):
        """Test history returns empty list for new user"""
        repository = self.get_repository()

        history = repository.get_baseline_history("new_user")

        assert history == []

    def test_get_baseline_history_returns_chronological_order(self):
        """Test baseline history is returned in chronological order (oldest first)"""
        repository = self.get_repository()

        # Create baselines with different dates
        baseline1 = UserBaseline(
            user_id="history_test_user",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.0,
            sleep_std=1.0,
            activity_mean=7500.0,
            activity_std=1800.0,
            circadian_phase=21.5,
            last_updated=datetime(2024, 1, 1, 10, 0),
            data_points=28,
        )

        baseline2 = UserBaseline(
            user_id="history_test_user",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 15, 10, 0),
            data_points=30,
        )

        # Save in reverse chronological order to test sorting
        repository.save_baseline(baseline2)
        repository.save_baseline(baseline1)

        # Get history
        history = repository.get_baseline_history("history_test_user")

        # Should be in chronological order (oldest first)
        assert len(history) == 2
        assert history[0].baseline_date <= history[1].baseline_date


class TestFileBaselineRepository(BaselineRepositoryContract):
    """Test FileBaselineRepository against the contract"""

    @pytest.fixture(autouse=True)
    def cleanup_test_files(self):
        """Clean up test files before and after each test to ensure isolation"""
        test_path = Path("./temp_test_baselines")

        # Clean before test
        if test_path.exists():
            shutil.rmtree(test_path)

        yield  # Run the test

        # Clean after test
        if test_path.exists():
            shutil.rmtree(test_path)

    def get_repository(self) -> BaselineRepositoryInterface:
        return FileBaselineRepository(base_path=Path("./temp_test_baselines"))


@pytest.mark.integration
class TestTimescaleBaselineRepository(BaselineRepositoryContract):
    """Test TimescaleBaselineRepository against the contract using TestContainers"""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Set up PostgreSQL container with TimescaleDB extension"""
        try:
            from testcontainers.postgres import PostgresContainer
        except ImportError:
            pytest.skip(
                "testcontainers not available - run: pip install testcontainers"
            )

        # Use TimescaleDB image which includes PostgreSQL + TimescaleDB extension
        container = PostgresContainer(
            image="timescale/timescaledb:latest-pg16",
            username="test_user",
            password="test_password",
            dbname="test_baselines",
        )

        container.start()

        # Wait for container to be ready
        time.sleep(2)

        yield container

        container.stop()

    @pytest.fixture
    def repository(self, postgres_container):
        """Create TimescaleBaselineRepository with test container"""
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository,
        )

        connection_string = postgres_container.get_connection_url()

        # Create repository (this will initialize tables)
        repo = TimescaleBaselineRepository(
            connection_string=connection_string,
            enable_feast_sync=False,  # Disable Feast for pure repository tests
        )

        return repo

    def get_repository(self) -> BaselineRepositoryInterface:
        """Required by contract - will be overridden by pytest fixture"""
        # This method won't be called directly in pytest,
        # but we need it for the contract interface
        raise NotImplementedError("Use repository fixture instead")

    def test_save_and_retrieve_baseline(self, repository):
        """Test basic save and retrieve operations with real TimescaleDB"""
        baseline = self.get_sample_baseline()

        # Save baseline
        repository.save_baseline(baseline)

        # Retrieve baseline
        retrieved = repository.get_baseline(baseline.user_id)

        # Verify values
        assert retrieved is not None
        assert retrieved.user_id == baseline.user_id
        assert retrieved.baseline_date == baseline.baseline_date
        assert retrieved.sleep_mean == baseline.sleep_mean
        assert retrieved.sleep_std == baseline.sleep_std
        assert retrieved.activity_mean == baseline.activity_mean
        assert retrieved.activity_std == baseline.activity_std
        assert retrieved.circadian_phase == baseline.circadian_phase
        assert retrieved.data_points == baseline.data_points

    def test_get_nonexistent_baseline_returns_none(self, repository):
        """Test retrieving non-existent baseline returns None"""
        result = repository.get_baseline("nonexistent_user")
        assert result is None

    def test_get_baseline_history_empty_for_new_user(self, repository):
        """Test history returns empty list for new user"""
        history = repository.get_baseline_history("new_user")
        assert history == []

    def test_get_baseline_history_returns_chronological_order(self, repository):
        """Test baseline history is returned in chronological order"""
        # Create baselines with different dates
        baseline1 = UserBaseline(
            user_id="history_test_user_ts",
            baseline_date=date(2024, 1, 1),
            sleep_mean=7.0,
            sleep_std=1.0,
            activity_mean=7500.0,
            activity_std=1800.0,
            circadian_phase=21.5,
            last_updated=datetime(2024, 1, 1, 10, 0),
            data_points=28,
        )

        baseline2 = UserBaseline(
            user_id="history_test_user_ts",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            last_updated=datetime(2024, 1, 15, 10, 0),
            data_points=30,
        )

        # Save in reverse chronological order to test sorting
        repository.save_baseline(baseline2)
        repository.save_baseline(baseline1)

        # Get history
        history = repository.get_baseline_history("history_test_user_ts")

        # Should be in chronological order (oldest first)
        assert len(history) == 2
        assert history[0].baseline_date <= history[1].baseline_date

    def test_hypertable_functionality(self, repository):
        """Test TimescaleDB-specific hypertable functionality"""
        self.get_sample_baseline()

        # Save multiple baselines for different dates
        for day_offset in range(5):
            baseline_copy = UserBaseline(
                user_id=f"hypertable_user_{day_offset}",
                baseline_date=date(2024, 1, 1 + day_offset),
                sleep_mean=7.0 + day_offset * 0.1,
                sleep_std=1.0,
                activity_mean=8000.0,
                activity_std=2000.0,
                circadian_phase=22.0,
                last_updated=datetime(2024, 1, 1 + day_offset, 10, 0),
                data_points=30,
            )
            repository.save_baseline(baseline_copy)

        # Verify we can retrieve different users' baselines
        for day_offset in range(5):
            user_id = f"hypertable_user_{day_offset}"
            retrieved = repository.get_baseline(user_id)
            assert retrieved is not None
            assert retrieved.user_id == user_id
            assert abs(retrieved.sleep_mean - (7.0 + day_offset * 0.1)) < 0.001

    def test_save_and_retrieve_baseline_with_hr_hrv(self, repository):
        """Test save and retrieve with optional HR/HRV fields"""
        # Create baseline with HR/HRV data
        baseline = UserBaseline(
            user_id="test_user_hr",
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.2,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            heart_rate_mean=65.0,
            heart_rate_std=5.0,
            hrv_mean=55.0,
            hrv_std=8.0,
            last_updated=datetime(2024, 1, 15, 10, 30),
            data_points=30,
        )

        # Save
        repository.save_baseline(baseline)

        # Retrieve
        retrieved = repository.get_baseline(baseline.user_id)

        # Verify all fields including HR/HRV
        assert retrieved is not None
        assert retrieved.heart_rate_mean == 65.0
        assert retrieved.heart_rate_std == 5.0
        assert retrieved.hrv_mean == 55.0
        assert retrieved.hrv_std == 8.0


@pytest.mark.integration
class TestBaselineRepositoryIntegration:
    """
    Integration tests that verify real implementations work correctly.
    These run slower but provide crucial confidence.
    """

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Set up PostgreSQL container with TimescaleDB extension"""
        try:
            from testcontainers.postgres import PostgresContainer
        except ImportError:
            pytest.skip(
                "testcontainers not available - run: pip install testcontainers"
            )

        container = PostgresContainer(
            image="timescale/timescaledb:latest-pg16",
            username="test_user",
            password="test_password",
            dbname="test_baselines",
        )

        container.start()
        time.sleep(2)  # Wait for startup

        yield container

        container.stop()

    def test_file_and_timescale_repos_are_interchangeable(self, postgres_container):
        """
        Integration test: Both implementations should behave identically
        for the same operations.
        """
        from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
            TimescaleBaselineRepository,
        )

        # Set up both repositories
        test_path = Path("./test_interop_data")

        # Clean up before test
        if test_path.exists():
            shutil.rmtree(test_path)

        try:
            file_repo = FileBaselineRepository(test_path)
            timescale_repo = TimescaleBaselineRepository(
                connection_string=postgres_container.get_connection_url(),
                enable_feast_sync=False,
            )

            baseline = UserBaseline(
                user_id="interop_test_user",
                baseline_date=date(2024, 1, 15),
                sleep_mean=7.5,
                sleep_std=1.2,
                activity_mean=8000.0,
                activity_std=2000.0,
                circadian_phase=22.0,
                last_updated=datetime(2024, 1, 15, 10, 30),
                data_points=30,
            )

            # Both should handle the same data identically
            file_repo.save_baseline(baseline)
            timescale_repo.save_baseline(baseline)

            file_result = file_repo.get_baseline(baseline.user_id)
            timescale_result = timescale_repo.get_baseline(baseline.user_id)

            # Results should be equivalent (allowing for minor serialization differences)
            assert file_result.user_id == timescale_result.user_id
            assert file_result.baseline_date == timescale_result.baseline_date
            assert abs(file_result.sleep_mean - timescale_result.sleep_mean) < 0.001
            assert abs(file_result.sleep_std - timescale_result.sleep_std) < 0.001
            assert (
                abs(file_result.activity_mean - timescale_result.activity_mean) < 0.001
            )
            assert abs(file_result.activity_std - timescale_result.activity_std) < 0.001
            assert (
                abs(file_result.circadian_phase - timescale_result.circadian_phase)
                < 0.001
            )
            assert file_result.data_points == timescale_result.data_points

        finally:
            # Clean up after test
            if test_path.exists():
                shutil.rmtree(test_path)
