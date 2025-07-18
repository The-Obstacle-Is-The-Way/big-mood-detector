"""
Episode Labeler Domain Service

Manages mood episode labels for ground truth collection.
"""

from datetime import date as date_type
from datetime import datetime
from typing import Any

import pandas as pd


class EpisodeLabeler:
    """Label episodes and baseline periods for training."""

    def __init__(self) -> None:
        """Initialize episode labeler."""
        self.episodes: list[dict[str, Any]] = []
        self.baseline_periods: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []  # For undo functionality

    def add_episode(
        self,
        *args: Any,
        date: str | date_type | None = None,
        start_date: str | date_type | None = None,
        end_date: str | date_type | None = None,
        episode_type: str = "",
        severity: int = 3,
        notes: str = "",
        rater_id: str = "default",
    ) -> None:
        """Add an episode label.

        Args:
            *args: Positional arguments for compatibility
            date: Single date for episode
            start_date: Start date for multi-day episode
            end_date: End date for multi-day episode
            episode_type: Type of episode (hypomanic, depressive, manic, mixed)
            severity: Severity rating (1-5)
            notes: Additional notes about the episode
            rater_id: Identifier for the rater
        """
        # Convert date objects to strings
        if date and isinstance(date, date_type):
            date = date.isoformat()
        if start_date and isinstance(start_date, date_type):
            start_date = start_date.isoformat()
        if end_date and isinstance(end_date, date_type):
            end_date = end_date.isoformat()

        # Handle positional arguments for backward compatibility
        if args:
            if len(args) == 3:
                # Single date: add_episode(date, episode_type, severity)
                date = args[0]
                episode_type = args[1]
                severity = args[2]
            elif len(args) == 4:
                # Date range: add_episode(start_date, end_date, episode_type, severity)
                start_date = args[0]
                end_date = args[1]
                episode_type = args[2]
                severity = args[3]

        episode: dict[str, Any] = {
            "episode_type": episode_type,
            "severity": severity,
            "notes": notes,
            "rater_id": rater_id,
            "labeled_at": datetime.now().isoformat(),
        }

        if date:
            # Single day episode
            episode["date"] = date
            episode["start_date"] = date
            episode["end_date"] = date
            episode["duration_days"] = 1
        elif start_date and end_date:
            # Date range episode
            if isinstance(start_date, str) and isinstance(end_date, str):
                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                end = datetime.strptime(end_date, "%Y-%m-%d").date()
            else:
                start = (
                    start_date
                    if isinstance(start_date, date_type)
                    else datetime.strptime(str(start_date), "%Y-%m-%d").date()
                )
                end = (
                    end_date
                    if isinstance(end_date, date_type)
                    else datetime.strptime(str(end_date), "%Y-%m-%d").date()
                )
            duration = (end - start).days + 1

            episode["start_date"] = (
                start_date if isinstance(start_date, str) else start_date.isoformat()
            )
            episode["end_date"] = (
                end_date if isinstance(end_date, str) else end_date.isoformat()
            )
            episode["duration_days"] = duration

        self.episodes.append(episode)
        self._history.append({"action": "add_episode", "data": episode})

    def add_baseline(
        self,
        start_date: str | date_type,
        end_date: str | date_type,
        notes: str = "",
        rater_id: str = "default",
    ) -> None:
        """Add a baseline (stable) period.

        Args:
            start_date: Start date of baseline period
            end_date: End date of baseline period
            notes: Additional notes about the baseline period
            rater_id: Identifier for the rater
        """
        # Convert date objects to strings
        if isinstance(start_date, date_type):
            start_date = start_date.isoformat()
        if isinstance(end_date, date_type):
            end_date = end_date.isoformat()

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        duration = (end - start).days + 1

        baseline_period = {
            "start_date": start_date,
            "end_date": end_date,
            "duration_days": duration,
            "notes": notes,
            "rater_id": rater_id,
            "labeled_at": datetime.now().isoformat(),
        }

        self.baseline_periods.append(baseline_period)
        self._history.append({"action": "add_baseline", "data": baseline_period})

    def check_overlap(self, start_date: date_type, end_date: date_type) -> bool:
        """Check if dates overlap with existing episodes.

        Args:
            start_date: Start date to check
            end_date: End date to check

        Returns:
            True if overlap exists
        """
        for episode in self.episodes:
            ep_start = datetime.strptime(episode["start_date"], "%Y-%m-%d").date()
            ep_end = datetime.strptime(episode["end_date"], "%Y-%m-%d").date()

            # Check for overlap
            if not (end_date < ep_start or start_date > ep_end):
                return True

        return False

    def undo_last(self) -> bool:
        """Undo the last labeling action.

        Returns:
            True if undo was successful
        """
        if not self._history:
            return False

        last_action = self._history.pop()

        if last_action["action"] == "add_episode":
            # Remove the last episode that matches
            self.episodes.pop()
        elif last_action["action"] == "add_baseline":
            # Remove the last baseline
            self.baseline_periods.pop()

        return True

    def to_dataframe(self) -> pd.DataFrame:
        """Export all labels to a DataFrame.

        Returns:
            DataFrame with columns: date, label, severity, confidence, model_agreed
        """
        rows = []

        # Add episode days
        for episode in self.episodes:
            if "date" in episode:
                # Single day episode
                rows.append(
                    {
                        "date": episode["date"],
                        "label": episode["episode_type"],
                        "severity": episode["severity"],
                        "confidence": 1.0,  # Default confidence
                        "model_agreed": False,  # Will be set later if model prediction matches
                        "rater_id": episode.get("rater_id", "default"),
                    }
                )
            else:
                # Multi-day episode: create a row for each day
                start = datetime.strptime(episode["start_date"], "%Y-%m-%d")
                end = datetime.strptime(episode["end_date"], "%Y-%m-%d")
                current = start

                while current <= end:
                    rows.append(
                        {
                            "date": current.strftime("%Y-%m-%d"),
                            "label": episode["episode_type"],
                            "severity": episode["severity"],
                            "confidence": 1.0,
                            "model_agreed": False,
                            "rater_id": episode.get("rater_id", "default"),
                        }
                    )
                    current = pd.Timestamp(current) + pd.Timedelta(days=1)

        # Add baseline days
        for baseline in self.baseline_periods:
            start = datetime.strptime(baseline["start_date"], "%Y-%m-%d")
            end = datetime.strptime(baseline["end_date"], "%Y-%m-%d")
            current = start

            while current <= end:
                rows.append(
                    {
                        "date": current.strftime("%Y-%m-%d"),
                        "label": "baseline",
                        "severity": 0,
                        "confidence": 1.0,
                        "model_agreed": True,  # Baseline usually agrees with low risk
                        "rater_id": baseline.get("rater_id", "default"),
                    }
                )
                current = pd.Timestamp(current) + pd.Timedelta(days=1)

        if not rows:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(
                columns=[
                    "date",
                    "label",
                    "severity",
                    "confidence",
                    "model_agreed",
                    "rater_id",
                ]
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df
