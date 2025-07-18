"""
Test SQLite Episode Repository

Tests for SQLite-based persistence of episode labels.
"""

import sqlite3
import tempfile
from datetime import date
from pathlib import Path

import pytest

from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler


class TestSQLiteEpisodeRepository:
    """Test SQLite repository for episode persistence."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        yield db_path
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    def test_repository_can_be_imported(self):
        """Test that SQLite repository can be imported."""
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        assert SQLiteEpisodeRepository is not None

    def test_create_repository_with_database(self, temp_db):
        """Test creating repository initializes database."""
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        repo = SQLiteEpisodeRepository(db_path=temp_db)
        
        # Verify tables were created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check episodes table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'"
        )
        assert cursor.fetchone() is not None
        
        # Check baseline_periods table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='baseline_periods'"
        )
        assert cursor.fetchone() is not None
        
        conn.close()

    def test_save_and_load_episodes(self, temp_db):
        """Test saving and loading episodes."""
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        repo = SQLiteEpisodeRepository(db_path=temp_db)
        
        # Create an episode labeler and add some episodes
        labeler = EpisodeLabeler()
        labeler.add_episode(
            date=date(2024, 3, 15),
            episode_type="depressive",
            severity=3,
            notes="Test episode",
            rater_id="test_user"
        )
        labeler.add_episode(
            start_date=date(2024, 3, 20),
            end_date=date(2024, 3, 25),
            episode_type="hypomanic",
            severity=2,
            rater_id="test_user"
        )
        
        # Save to repository
        repo.save_labeler(labeler)
        
        # Load into a new labeler
        new_labeler = EpisodeLabeler()
        repo.load_into_labeler(new_labeler)
        
        # Verify episodes were loaded
        assert len(new_labeler.episodes) == 2
        assert new_labeler.episodes[0]["episode_type"] == "depressive"
        assert new_labeler.episodes[1]["episode_type"] == "hypomanic"

    def test_save_and_load_baselines(self, temp_db):
        """Test saving and loading baseline periods."""
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        repo = SQLiteEpisodeRepository(db_path=temp_db)
        
        # Create labeler with baseline periods
        labeler = EpisodeLabeler()
        labeler.add_baseline(
            start_date=date(2024, 4, 1),
            end_date=date(2024, 4, 14),
            notes="Stable period",
            rater_id="test_user"
        )
        
        # Save and reload
        repo.save_labeler(labeler)
        
        new_labeler = EpisodeLabeler()
        repo.load_into_labeler(new_labeler)
        
        # Verify baseline was loaded
        assert len(new_labeler.baseline_periods) == 1
        assert new_labeler.baseline_periods[0]["notes"] == "Stable period"

    def test_update_existing_episodes(self, temp_db):
        """Test updating existing episodes doesn't create duplicates."""
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        repo = SQLiteEpisodeRepository(db_path=temp_db)
        
        # Save initial episode
        labeler = EpisodeLabeler()
        labeler.add_episode(
            date=date(2024, 3, 15),
            episode_type="depressive",
            severity=3
        )
        repo.save_labeler(labeler)
        
        # Add another episode and save again
        labeler.add_episode(
            date=date(2024, 3, 16),
            episode_type="manic",
            severity=4
        )
        repo.save_labeler(labeler)
        
        # Load and verify no duplicates
        new_labeler = EpisodeLabeler()
        repo.load_into_labeler(new_labeler)
        
        assert len(new_labeler.episodes) == 2

    def test_thread_safe_operations(self, temp_db):
        """Test repository is thread-safe."""
        import threading
        
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        repo = SQLiteEpisodeRepository(db_path=temp_db)
        errors = []
        
        def add_episode(date_val):
            try:
                labeler = EpisodeLabeler()
                labeler.add_episode(
                    date=date_val,
                    episode_type="depressive",
                    severity=3
                )
                repo.save_labeler(labeler)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=add_episode, 
                args=(date(2024, 3, i + 1),)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0

    def test_query_by_date_range(self, temp_db):
        """Test querying episodes by date range."""
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        repo = SQLiteEpisodeRepository(db_path=temp_db)
        
        # Add episodes across different dates
        labeler = EpisodeLabeler()
        labeler.add_episode(date=date(2024, 1, 15), episode_type="depressive", severity=3)
        labeler.add_episode(date=date(2024, 2, 15), episode_type="manic", severity=4)
        labeler.add_episode(date=date(2024, 3, 15), episode_type="hypomanic", severity=2)
        repo.save_labeler(labeler)
        
        # Query specific range
        episodes = repo.get_episodes_by_date_range(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 3, 1)
        )
        
        assert len(episodes) == 1
        assert episodes[0]["episode_type"] == "manic"

    def test_get_episodes_by_rater(self, temp_db):
        """Test querying episodes by rater ID."""
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        repo = SQLiteEpisodeRepository(db_path=temp_db)
        
        # Add episodes from different raters
        labeler1 = EpisodeLabeler()
        labeler1.add_episode(
            date=date(2024, 3, 15),
            episode_type="depressive",
            severity=3,
            rater_id="rater1"
        )
        repo.save_labeler(labeler1)
        
        labeler2 = EpisodeLabeler()
        labeler2.add_episode(
            date=date(2024, 3, 16),
            episode_type="manic",
            severity=4,
            rater_id="rater2"
        )
        repo.save_labeler(labeler2)
        
        # Query by specific rater
        episodes = repo.get_episodes_by_rater("rater1")
        
        assert len(episodes) == 1
        assert episodes[0]["rater_id"] == "rater1"