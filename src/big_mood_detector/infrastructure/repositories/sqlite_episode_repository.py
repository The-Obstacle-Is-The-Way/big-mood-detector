"""
SQLite Episode Repository

Provides SQLite-based persistence for episode labels.
"""

import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
from big_mood_detector.infrastructure.logging import get_module_logger

logger = get_module_logger(__name__)


class SQLiteEpisodeRepository:
    """SQLite repository for persisting episode labels."""

    def __init__(self, db_path: Path | str):
        """Initialize repository with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self) -> None:
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
                    severity INTEGER NOT NULL,
                    notes TEXT,
                    rater_id TEXT NOT NULL,
                    labeled_at TEXT NOT NULL,
                    duration_days INTEGER NOT NULL,
                    UNIQUE(start_date, end_date, rater_id)
                )
            """
            )

            # Create baseline_periods table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS baseline_periods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    notes TEXT,
                    rater_id TEXT NOT NULL,
                    labeled_at TEXT NOT NULL,
                    duration_days INTEGER NOT NULL,
                    UNIQUE(start_date, end_date, rater_id)
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_date ON episodes(start_date, end_date)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_rater ON episodes(rater_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_baseline_date ON baseline_periods(start_date, end_date)"
            )

            conn.commit()
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
                # Clear existing data first
                cursor.execute("DELETE FROM episodes")
                cursor.execute("DELETE FROM baseline_periods")
                
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

    def get_episodes_by_date_range(
        self, start_date: date, end_date: date
    ) -> list[dict[str, Any]]:
        """Get episodes within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of episodes in the range
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM episodes
                WHERE start_date >= ? AND end_date <= ?
                ORDER BY start_date
            """,
                (start_date.isoformat(), end_date.isoformat()),
            )

            episodes = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return episodes

    def get_episodes_by_rater(self, rater_id: str) -> list[dict[str, Any]]:
        """Get all episodes by a specific rater.

        Args:
            rater_id: ID of the rater

        Returns:
            List of episodes by the rater
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM episodes
                WHERE rater_id = ?
                ORDER BY start_date
            """,
                (rater_id,),
            )

            episodes = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return episodes
