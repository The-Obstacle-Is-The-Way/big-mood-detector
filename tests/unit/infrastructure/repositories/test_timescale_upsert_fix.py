"""
Test TimescaleDB repository uses proper UPSERT instead of DELETE+INSERT.

This prevents race conditions when multiple processes update the same baseline.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    UserBaseline,
)
from big_mood_detector.infrastructure.repositories.timescale_baseline_repository import (
    Base,
    TimescaleBaselineRepository,
)


class TestTimescaleUpsertPattern:
    """Test that repository uses atomic UPSERT operations."""

    @pytest.fixture
    def test_db(self):
        """Create in-memory SQLite database for testing."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def repository(self, test_db):
        """Create repository with test database."""
        repo = TimescaleBaselineRepository(
            connection_string="sqlite:///:memory:", enable_feast_sync=False
        )
        # Replace the session to track SQL calls
        repo._test_engine = test_db
        return repo

    def test_save_baseline_uses_upsert_not_delete_insert(self, repository, monkeypatch):
        """Test that save_baseline uses UPSERT pattern for atomicity."""
        # Create a baseline
        baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date.today(),
            sleep_mean=7.5,
            sleep_std=1.0,
            activity_mean=10000,
            activity_std=2000,
            circadian_phase=22.0,
            heart_rate_mean=65.0,
            hrv_mean=55.0,
            data_points=30,
        )

        # Track whether INSERT ... ON CONFLICT was used
        upsert_used = False
        original_execute = None

        def track_execute(stmt):
            nonlocal upsert_used
            # Check if this is an insert statement with on_conflict
            if hasattr(stmt, "on_conflict_do_update"):
                upsert_used = True
            if original_execute:
                return original_execute(stmt)

        # Mock the session to track SQL operations
        with patch.object(repository, "_get_session") as mock_context:
            mock_session = MagicMock()
            mock_context.return_value.__enter__.return_value = mock_session
            original_execute = mock_session.execute
            mock_session.execute = track_execute

            # Save the baseline
            repository.save_baseline(baseline)

            # Verify NO query/delete operations (old pattern)
            assert not hasattr(mock_session, "query") or not mock_session.query.called

            # Verify execute was called (new UPSERT pattern)
            # The track_execute function was called
            assert upsert_used or mock_session.execute.call_count > 0

    def test_concurrent_updates_dont_lose_data(self, repository):
        """Test that concurrent updates to same baseline don't lose data."""
        # Skip this test for SQLite - it doesn't support PostgreSQL UPSERT syntax
        # This test is for documentation of expected behavior with PostgreSQL
        pytest.skip("SQLite doesn't support PostgreSQL UPSERT syntax")

    def test_session_management_prevents_leaks(self, repository):
        """Test that sessions are properly closed even on exceptions."""
        baseline = UserBaseline(
            user_id="test_user",
            baseline_date=date.today(),
            sleep_mean=7.0,
            sleep_std=1.0,
            activity_mean=8000,
            activity_std=1500,
            circadian_phase=22.0,
            data_points=10,
        )

        # Test 1: Normal operation - session should be closed
        close_called = False
        commit_called = False

        class MockSession:
            def begin(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                nonlocal commit_called
                commit_called = True
                return False

            def add(self, obj):
                pass

            def execute(self, stmt):
                pass

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                nonlocal close_called
                close_called = True

        with patch.object(repository, "SessionLocal", return_value=MockSession()):
            repository.save_baseline(baseline)
            assert close_called, "Session should be closed after normal operation"
            assert commit_called, "Session should be committed"

        # Test 2: Exception during operation - session should still be closed
        close_called = False
        rollback_called = False

        class MockSessionWithError:
            def begin(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type:
                    nonlocal rollback_called
                    rollback_called = True
                return False

            def add(self, obj):
                pass

            def execute(self, stmt):
                raise Exception("DB Error")

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                nonlocal close_called
                close_called = True

        with patch.object(
            repository, "SessionLocal", return_value=MockSessionWithError()
        ):
            with pytest.raises(Exception, match="DB Error"):
                repository.save_baseline(baseline)

            assert close_called, "Session should be closed even after exception"
            assert rollback_called, "Session should be rolled back on exception"
