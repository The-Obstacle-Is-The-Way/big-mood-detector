"""Test concurrency handling in TimescaleDB repository."""

import concurrent.futures
import threading
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    UserBaseline,
)
from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
    TimescaleBaselineRepository,
)


class TestTimescaleConcurrency:
    """Test concurrent operations in TimescaleDB repository."""

    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory that simulates concurrent DB access."""
        sessions = []
        session_lock = threading.Lock()
        
        def create_session():
            session = MagicMock()
            
            # Track sessions for verification
            with session_lock:
                sessions.append(session)
            
            # Mock the begin() context manager
            session.begin.return_value.__enter__ = Mock(return_value=None)
            session.begin.return_value.__exit__ = Mock(return_value=None)
            
            # Mock execute to simulate successful UPSERT
            session.execute = Mock(return_value=None)
            
            # Mock query for baseline retrieval
            session.query = Mock()
            
            return session
        
        factory = Mock(side_effect=create_session)
        factory.sessions = sessions  # Attach for inspection
        return factory

    @pytest.fixture
    def repository(self, mock_session_factory):
        """Create repository with mocked dependencies."""
        with patch("big_mood_detector.infrastructure.repositories.timescale_baseline_repository.create_engine"):
            with patch("big_mood_detector.infrastructure.repositories.timescale_baseline_repository.sessionmaker") as mock_sessionmaker:
                mock_sessionmaker.return_value = mock_session_factory
                repo = TimescaleBaselineRepository(
                    connection_string="postgresql://test",
                    enable_feast_sync=False
                )
                # Ensure Feast is completely disabled
                repo.feast_client = None
                return repo

    def test_concurrent_save_baseline_same_user(self, repository):
        """Test that concurrent saves for the same user don't cause integrity errors."""
        user_id = "test_user"
        baseline_date = date(2024, 1, 15)
        
        # Create two slightly different baselines for the same user
        baseline1 = UserBaseline(
            user_id=user_id,
            baseline_date=baseline_date,
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            heart_rate_mean=65.0,
            heart_rate_std=5.0,
            hrv_mean=50.0,
            hrv_std=10.0,
            last_updated=datetime.now(timezone.utc),
            data_points=30
        )
        
        baseline2 = UserBaseline(
            user_id=user_id,
            baseline_date=baseline_date,
            sleep_mean=7.6,  # Slightly different value
            sleep_std=1.1,
            activity_mean=8100.0,
            activity_std=2100.0,
            circadian_phase=22.1,
            heart_rate_mean=66.0,
            heart_rate_std=5.1,
            hrv_mean=51.0,
            hrv_std=10.1,
            last_updated=datetime.now(timezone.utc),
            data_points=31
        )
        
        # Track which baseline "won" (was saved last)
        results = []
        errors = []
        
        def save_baseline(baseline, index):
            """Save baseline and track results."""
            try:
                repository.save_baseline(baseline)
                results.append((index, baseline))
            except Exception as e:
                errors.append((index, e))
        
        # Run concurrent saves
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(save_baseline, baseline1, 1)
            future2 = executor.submit(save_baseline, baseline2, 2)
            
            # Wait for both to complete
            concurrent.futures.wait([future1, future2])
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent saves failed: {errors}"
        
        # Verify both operations completed
        assert len(results) == 2
        
        # Verify sessions were properly managed
        sessions = repository.SessionLocal.sessions
        assert len(sessions) == 2  # Two sessions were created
        
        # Verify each session was properly closed
        for session in sessions:
            session.close.assert_called_once()

    def test_concurrent_save_different_users(self, repository):
        """Test that concurrent saves for different users work correctly."""
        baselines = []
        for i in range(5):
            baseline = UserBaseline(
                user_id=f"user_{i}",
                baseline_date=date(2024, 1, 15),
                sleep_mean=7.0 + i * 0.1,
                sleep_std=1.0,
                activity_mean=8000.0 + i * 100,
                activity_std=2000.0,
                circadian_phase=22.0,
                heart_rate_mean=60.0 + i,
                heart_rate_std=5.0,
                hrv_mean=45.0 + i,
                hrv_std=10.0,
                last_updated=datetime.now(timezone.utc),
                data_points=30
            )
            baselines.append(baseline)
        
        results = []
        errors = []
        
        def save_baseline(baseline, index):
            """Save baseline and track results."""
            try:
                repository.save_baseline(baseline)
                results.append((index, baseline))
            except Exception as e:
                errors.append((index, e))
        
        # Run concurrent saves for different users
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(save_baseline, baseline, i)
                for i, baseline in enumerate(baselines)
            ]
            concurrent.futures.wait(futures)
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent saves failed: {errors}"
        
        # Verify all operations completed
        assert len(results) == 5
        
        # Verify sessions were properly managed
        sessions = repository.SessionLocal.sessions
        assert len(sessions) == 5  # Five sessions were created
        
        # Verify each session was properly closed
        for session in sessions:
            session.close.assert_called_once()

    def test_concurrent_read_write(self, repository):
        """Test concurrent reads and writes don't interfere."""
        user_id = "test_user"
        baseline = UserBaseline(
            user_id=user_id,
            baseline_date=date(2024, 1, 15),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=8000.0,
            activity_std=2000.0,
            circadian_phase=22.0,
            heart_rate_mean=65.0,
            heart_rate_std=5.0,
            hrv_mean=50.0,
            hrv_std=10.0,
            last_updated=datetime.now(timezone.utc),
            data_points=30
        )
        
        results = {"reads": [], "writes": [], "errors": []}
        
        def write_operation(index):
            """Perform write operation."""
            try:
                repository.save_baseline(baseline)
                results["writes"].append(index)
            except Exception as e:
                results["errors"].append(("write", index, e))
        
        def read_operation(index):
            """Perform read operation."""
            try:
                # Mock the read to return None (not found)
                with patch.object(repository, "_get_from_timescale", return_value=None):
                    with patch.object(repository, "_get_from_feast", return_value=None):
                        result = repository.get_baseline(user_id)
                        results["reads"].append((index, result))
            except Exception as e:
                results["errors"].append(("read", index, e))
        
        # Mix reads and writes
        operations = []
        for i in range(10):
            if i % 2 == 0:
                operations.append(("write", i, write_operation))
            else:
                operations.append(("read", i, read_operation))
        
        # Run operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(op_func, index)
                for op_type, index, op_func in operations
            ]
            concurrent.futures.wait(futures)
        
        # Verify no errors occurred
        assert len(results["errors"]) == 0, f"Concurrent operations failed: {results['errors']}"
        
        # Verify operations completed
        assert len(results["writes"]) == 5
        assert len(results["reads"]) == 5

    def test_concurrent_upsert_stress(self, repository):
        """Stress test with many concurrent upserts to same record."""
        user_id = "stress_test_user"
        baseline_date = date(2024, 1, 15)
        num_threads = 20
        
        results = []
        errors = []
        
        def upsert_baseline(thread_id):
            """Perform upsert with unique values per thread."""
            try:
                baseline = UserBaseline(
                    user_id=user_id,
                    baseline_date=baseline_date,
                    sleep_mean=7.0 + thread_id * 0.01,  # Unique per thread
                    sleep_std=1.0,
                    activity_mean=8000.0 + thread_id,
                    activity_std=2000.0,
                    circadian_phase=22.0,
                    heart_rate_mean=60.0 + thread_id * 0.1,
                    heart_rate_std=5.0,
                    hrv_mean=45.0 + thread_id * 0.1,
                    hrv_std=10.0,
                    last_updated=datetime.now(timezone.utc),
                    data_points=30 + thread_id
                )
                repository.save_baseline(baseline)
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))
        
        # Run many concurrent upserts
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(upsert_baseline, i)
                for i in range(num_threads)
            ]
            concurrent.futures.wait(futures)
        
        # Verify no integrity errors occurred
        assert len(errors) == 0, f"Upserts failed: {errors}"
        
        # Verify all operations completed
        assert len(results) == num_threads
        
        # Verify proper session management
        sessions = repository.SessionLocal.sessions
        assert len(sessions) == num_threads
        for session in sessions:
            session.close.assert_called_once()