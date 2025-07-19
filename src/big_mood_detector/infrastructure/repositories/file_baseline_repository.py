"""
File-based Baseline Repository

Simple implementation using JSON files for baseline storage.
Following KISS principle - perfect for MVP with hundreds of users.
"""

import json
from datetime import date, datetime
from pathlib import Path

from big_mood_detector.domain.repositories.baseline_repository_interface import (
    BaselineRepositoryInterface,
    UserBaseline,
)
from big_mood_detector.infrastructure.logging import get_logger

logger = get_logger()


class FileBaselineRepository(BaselineRepositoryInterface):
    """
    File-based implementation of baseline repository.

    Stores baselines as JSON files: baselines/{user_id}/baseline_history.json
    Simple, reliable, and sufficient for MVP scale.
    """

    def __init__(self, base_path: Path):
        """
        Initialize repository with base storage path.

        Args:
            base_path: Directory where baseline files will be stored
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("file_baseline_repository_initialized", path=str(self.base_path))

    def save_baseline(self, baseline: UserBaseline) -> None:
        """Save or update a user's baseline."""
        user_dir = self.base_path / baseline.user_id
        user_dir.mkdir(exist_ok=True)

        history_file = user_dir / "baseline_history.json"

        # Load existing history
        history = self._load_history(history_file)

        # Add new baseline
        baseline_dict = self._baseline_to_dict(baseline)
        history.append(baseline_dict)

        # Keep only last 10 baselines
        history = history[-10:]

        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(
            "baseline_saved",
            user_id=baseline.user_id,
            date=str(baseline.baseline_date),
            data_points=baseline.data_points,
        )

    def get_baseline(self, user_id: str) -> UserBaseline | None:
        """Retrieve the most recent baseline for a user."""
        history_file = self.base_path / user_id / "baseline_history.json"

        if not history_file.exists():
            logger.debug("baseline_not_found", user_id=user_id)
            return None

        history = self._load_history(history_file)
        if not history:
            return None

        # Return most recent baseline
        latest_dict = history[-1]
        baseline = self._dict_to_baseline(latest_dict)

        logger.debug(
            "baseline_retrieved",
            user_id=user_id,
            date=str(baseline.baseline_date),
        )
        return baseline

    def get_baseline_history(
        self, user_id: str, limit: int = 10
    ) -> list[UserBaseline]:
        """Get historical baselines for trend analysis."""
        history_file = self.base_path / user_id / "baseline_history.json"

        if not history_file.exists():
            logger.debug("baseline_history_not_found", user_id=user_id)
            return []

        history = self._load_history(history_file)

        # Convert to UserBaseline objects
        baselines = [self._dict_to_baseline(d) for d in history]
        
        # Sort by baseline_date chronologically (oldest first) and limit
        baselines.sort(key=lambda b: b.baseline_date)
        baselines = baselines[-limit:] if limit else baselines

        logger.debug(
            "baseline_history_retrieved",
            user_id=user_id,
            count=len(baselines),
        )
        return baselines

    def _load_history(self, history_file: Path) -> list[dict]:
        """Load baseline history from file."""
        if not history_file.exists():
            return []

        with open(history_file) as f:
            data: list[dict] = json.load(f)
            return data

    def _baseline_to_dict(self, baseline: UserBaseline) -> dict:
        """Convert UserBaseline to dictionary for JSON storage."""
        return {
            "user_id": baseline.user_id,
            "baseline_date": baseline.baseline_date.isoformat(),
            "sleep_mean": baseline.sleep_mean,
            "sleep_std": baseline.sleep_std,
            "activity_mean": baseline.activity_mean,
            "activity_std": baseline.activity_std,
            "circadian_phase": baseline.circadian_phase,
            "last_updated": baseline.last_updated.isoformat(),
            "data_points": baseline.data_points,
        }

    def _dict_to_baseline(self, data: dict) -> UserBaseline:
        """Convert dictionary to UserBaseline object."""
        return UserBaseline(
            user_id=data["user_id"],
            baseline_date=date.fromisoformat(data["baseline_date"]),
            sleep_mean=data["sleep_mean"],
            sleep_std=data["sleep_std"],
            activity_mean=data["activity_mean"],
            activity_std=data["activity_std"],
            circadian_phase=data["circadian_phase"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            data_points=data["data_points"],
        )
