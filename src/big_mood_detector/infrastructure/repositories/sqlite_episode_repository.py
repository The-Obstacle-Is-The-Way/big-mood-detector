"""
SQLite repository for episode labels.

Provides thread-safe storage for mood episode labels and baseline periods.
"""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from big_mood_detector.infrastructure.fine_tuning.personal_calibrator import (
    EpisodeLabeler,
)
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)


class SQLiteEpisodeRepository:
    """SQLite-based repository for episode labels."""

    def __init__(self, db_path: str | Path = "labels.db"):
        """Initialize repository with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create episodes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    episode_type TEXT NOT NULL,
                    severity INTEGER,
                    notes TEXT,
                    rater_id TEXT DEFAULT 'default',
                    labeled_at TEXT,
                    duration_days INTEGER
                )
            """
            )

            # Create baseline periods table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS baseline_periods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    notes TEXT,
                    rater_id TEXT DEFAULT 'default',
                    labeled_at TEXT,
                    duration_days INTEGER
                )
            """
            )

            # Create indices
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_dates ON episodes(start_date, end_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_rater ON episodes(rater_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_baseline_dates ON baseline_periods(start_date, end_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_baseline_rater ON baseline_periods(rater_id)"
            )

            conn.commit()
            conn.close()

    def _save_labeler_data(self, cursor: sqlite3.Cursor, labeler: EpisodeLabeler) -> None:
        """Helper to save labeler data to database."""
        # Save episodes
        for episode in labeler.episodes:
            # Handle both single-date and range episodes
            if "date" in episode:
                start_date = end_date = episode["date"]
            else:
                start_date = episode["start_date"]
                end_date = episode["end_date"]

            cursor.execute(
                """
                INSERT OR REPLACE INTO episodes
                (start_date, end_date, episode_type, severity, notes,
                 rater_id, labeled_at, duration_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    start_date,
                    end_date,
                    episode["episode_type"],
                    episode["severity"],
                    episode.get("notes", ""),
                    episode.get("rater_id", "default"),
                    episode.get("labeled_at", datetime.now().isoformat()),
                    episode.get("duration_days", 1),
                ),
            )

        # Save baseline periods
        for baseline in labeler.baseline_periods:
            cursor.execute(
                """
                INSERT OR REPLACE INTO baseline_periods
                (start_date, end_date, notes, rater_id, labeled_at, duration_days)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    baseline["start_date"],
                    baseline["end_date"],
                    baseline.get("notes", ""),
                    baseline.get("rater_id", "default"),
                    baseline.get("labeled_at", datetime.now().isoformat()),
                    baseline["duration_days"],
                ),
            )

    def clear_and_save_labeler(self, labeler: EpisodeLabeler) -> None:
        """
        Clear all existing data and save the labeler's data.
        
        This is used when we want to completely replace the data,
        such as after deleting an episode via the API.
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                # Clear existing data first
                cursor.execute("DELETE FROM episodes")
                cursor.execute("DELETE FROM baseline_periods")
                
                # Then save the new data
                self._save_labeler_data(cursor, labeler)
                
                conn.commit()
                logger.info(
                    f"Saved labeler data",
                    episodes=len(labeler.episodes),
                    baselines=len(labeler.baseline_periods),
                )
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to save labeler: {e}")
                raise
            finally:
                conn.close()

    def save_labeler(self, labeler: EpisodeLabeler) -> None:
        """Save all episodes and baselines from a labeler.

        Args:
            labeler: EpisodeLabeler containing episodes to save
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            try:
                self._save_labeler_data(cursor, labeler)
                
                conn.commit()
                logger.info(
                    "Saved labeler data",
                    episodes=len(labeler.episodes),
                    baselines=len(labeler.baseline_periods),
                )

            except Exception as e:
                conn.rollback()
                logger.error("Failed to save labeler data", error=str(e))
                raise
            finally:
                conn.close()

    def load_into_labeler(self, labeler: EpisodeLabeler) -> None:
        """Load all episodes and baselines into a labeler.

        Args:
            labeler: EpisodeLabeler to load data into
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Load episodes
            cursor.execute("SELECT * FROM episodes ORDER BY start_date")
            for row in cursor.fetchall():
                episode = dict(row)

                # Convert to expected format
                if episode["start_date"] == episode["end_date"]:
                    # Single-day episode
                    episode["date"] = episode["start_date"]

                # Remove database-specific fields
                episode.pop("id", None)

                labeler.episodes.append(episode)

            # Load baseline periods
            cursor.execute("SELECT * FROM baseline_periods ORDER BY start_date")
            for row in cursor.fetchall():
                baseline = dict(row)
                baseline.pop("id", None)
                labeler.baseline_periods.append(baseline)

            conn.close()

            logger.info(
                "Loaded labeler data",
                episodes=len(labeler.episodes),
                baselines=len(labeler.baseline_periods),
            )

    def get_episodes_by_rater(self, rater_id: str) -> list[dict]:
        """Get all episodes for a specific rater.

        Args:
            rater_id: ID of the rater

        Returns:
            List of episode dictionaries
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM episodes WHERE rater_id = ? ORDER BY start_date",
                (rater_id,),
            )

            episodes = []
            for row in cursor.fetchall():
                episode = dict(row)
                if episode["start_date"] == episode["end_date"]:
                    episode["date"] = episode["start_date"]
                episode.pop("id", None)
                episodes.append(episode)

            conn.close()
            return episodes

    def get_baselines_by_rater(self, rater_id: str) -> list[dict]:
        """Get all baseline periods for a specific rater.

        Args:
            rater_id: ID of the rater

        Returns:
            List of baseline dictionaries
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM baseline_periods WHERE rater_id = ? ORDER BY start_date",
                (rater_id,),
            )

            baselines = []
            for row in cursor.fetchall():
                baseline = dict(row)
                baseline.pop("id", None)
                baselines.append(baseline)

            conn.close()
            return baselines

    def clear_all(self) -> None:
        """Clear all data from the repository."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM episodes")
            cursor.execute("DELETE FROM baseline_periods")

            conn.commit()
            conn.close()

            logger.info("Cleared all label data")