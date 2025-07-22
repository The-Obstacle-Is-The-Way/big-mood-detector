"""
Sleep Window Analyzer Service

Implements clinical sleep window analysis for bipolar disorder detection.
Merges sleep episodes within 3.75-hour windows following research guidelines.

Design Patterns:
- Strategy Pattern: Configurable merge threshold
- Value Objects: Immutable SleepWindow and WindowAnalysisResult
- Single Responsibility: Only analyzes windows, not sleep quality
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from big_mood_detector.domain.entities.sleep_record import SleepRecord


@dataclass(frozen=True)
class SleepWindow:
    """
    Immutable value object representing a merged sleep window.

    A window contains one or more sleep episodes that are
    close enough in time to be considered a single sleep period.
    """

    start_time: datetime
    end_time: datetime
    episode_count: int
    total_duration_hours: float
    gap_hours: list[float] = field(default_factory=list)

    @property
    def window_span_hours(self) -> float:
        """Total time span from first sleep to last wake."""
        return (self.end_time - self.start_time).total_seconds() / 3600

    @property
    def fragmentation_index(self) -> float:
        """Ratio of gaps to total window span (0-1)."""
        if self.window_span_hours == 0:
            return 0.0
        total_gap_hours = sum(self.gap_hours)
        return total_gap_hours / self.window_span_hours

    @property
    def is_fragmented(self) -> bool:
        """Window is fragmented if it has multiple episodes."""
        return self.episode_count > 1


@dataclass(frozen=True)
class WindowAnalysisResult:
    """
    Immutable value object containing sleep window analysis metrics.

    These metrics are critical for bipolar disorder detection,
    particularly the percentage of short/long windows.
    """

    total_windows: int
    short_windows: int  # < 6 hours
    long_windows: int  # > 10 hours
    fragmented_windows: int
    average_window_duration_hours: float
    short_window_percentage: float
    long_window_percentage: float
    fragmentation_percentage: float


class SleepWindowAnalyzer:
    """
    Analyzes sleep episodes to identify sleep windows.

    Following the Interface Segregation Principle (ISP),
    this service has a single, focused interface for window analysis.
    """

    # Clinical thresholds from research
    DEFAULT_MERGE_THRESHOLD_HOURS = 3.75
    SHORT_WINDOW_THRESHOLD_HOURS = 6.0
    LONG_WINDOW_THRESHOLD_HOURS = 10.0

    def __init__(self, merge_threshold_hours: float = DEFAULT_MERGE_THRESHOLD_HOURS):
        """
        Initialize analyzer with configurable merge threshold.

        Args:
            merge_threshold_hours: Maximum gap between episodes to merge (default: 3.75)
        """
        self.merge_threshold_hours = merge_threshold_hours

    def analyze_sleep_episodes(
        self, episodes: list[SleepRecord], target_date: date | None = None
    ) -> list[SleepWindow]:
        """
        Analyze sleep episodes and merge into windows.

        Args:
            episodes: List of sleep records to analyze
            target_date: Optional date to filter episodes

        Returns:
            List of merged sleep windows, sorted by start time

        Raises:
            ValueError: If episodes contains None or invalid types
        """
        # Since episodes is typed as list[SleepRecord], we can trust the type system
        # But we'll validate if passed from untrusted sources
        if not all(isinstance(e, SleepRecord) for e in episodes):
            raise ValueError("All episodes must be SleepRecord instances")

        # Filter episodes for target date if specified
        if target_date:
            if not isinstance(target_date, date):
                raise ValueError(f"Expected date, got {type(target_date).__name__}")
            # Use Seoul paper rule: assign based on nearest midnight of midpoint
            filtered_episodes = []
            for e in episodes:
                midpoint = e.start_date + (e.end_date - e.start_date) / 2
                # Find nearest midnight
                midnight_today = midpoint.replace(hour=0, minute=0, second=0, microsecond=0)
                midnight_tomorrow = midnight_today + timedelta(days=1)
                
                # Check which midnight is closer
                time_to_today_midnight = abs((midpoint - midnight_today).total_seconds())
                time_to_tomorrow_midnight = abs((midpoint - midnight_tomorrow).total_seconds())
                
                if time_to_today_midnight <= time_to_tomorrow_midnight:
                    assigned_date = midnight_today.date()
                else:
                    assigned_date = midnight_tomorrow.date()
                
                if assigned_date == target_date:
                    filtered_episodes.append(e)
            episodes = filtered_episodes

        if not episodes:
            return []

        # Sort episodes by start time
        sorted_episodes = sorted(episodes, key=lambda e: e.start_date)

        # Initialize with first episode
        windows: list[SleepWindow] = []
        current_window_episodes = [sorted_episodes[0]]
        current_gaps: list[float] = []

        # Process remaining episodes
        for episode in sorted_episodes[1:]:
            # Calculate gap from last episode in current window
            last_episode = current_window_episodes[-1]
            gap_hours = self._calculate_gap_hours(last_episode, episode)

            if gap_hours <= self.merge_threshold_hours:
                # Merge into current window
                current_window_episodes.append(episode)
                if gap_hours > 0:  # Only add positive gaps (not overlaps)
                    current_gaps.append(gap_hours)
            else:
                # Finalize current window and start new one
                windows.append(
                    self._create_window(current_window_episodes, current_gaps)
                )
                current_window_episodes = [episode]
                current_gaps = []

        # Don't forget the last window
        windows.append(self._create_window(current_window_episodes, current_gaps))

        return windows

    def get_analysis_summary(
        self, episodes: list[SleepRecord], days: int
    ) -> WindowAnalysisResult:
        """
        Get comprehensive analysis summary for a period.

        Args:
            episodes: Sleep records to analyze
            days: Number of days in the analysis period

        Returns:
            Analysis result with key metrics

        Raises:
            ValueError: If days is not positive
        """
        # Validate days
        if days is None or days <= 0:
            raise ValueError(f"Days must be positive, got {days}")

        try:
            windows = self.analyze_sleep_episodes(episodes)
        except Exception as e:
            # Log error and return empty result
            print(f"Warning: Sleep analysis failed: {e}")
            return WindowAnalysisResult(
                total_windows=0,
                short_windows=0,
                long_windows=0,
                fragmented_windows=0,
                average_window_duration_hours=0.0,
                short_window_percentage=0.0,
                long_window_percentage=0.0,
                fragmentation_percentage=0.0,
            )

        if not windows:
            return WindowAnalysisResult(
                total_windows=0,
                short_windows=0,
                long_windows=0,
                fragmented_windows=0,
                average_window_duration_hours=0.0,
                short_window_percentage=0.0,
                long_window_percentage=0.0,
                fragmentation_percentage=0.0,
            )

        # Count window types
        short_windows = sum(
            1
            for w in windows
            if w.total_duration_hours < self.SHORT_WINDOW_THRESHOLD_HOURS
        )
        long_windows = sum(
            1
            for w in windows
            if w.total_duration_hours > self.LONG_WINDOW_THRESHOLD_HOURS
        )
        fragmented_windows = sum(1 for w in windows if w.is_fragmented)

        # Calculate averages
        total_duration = sum(w.total_duration_hours for w in windows)
        avg_duration = total_duration / len(windows)

        # Calculate percentages
        total = len(windows)
        short_pct = (short_windows / total * 100) if total > 0 else 0.0
        long_pct = (long_windows / total * 100) if total > 0 else 0.0
        frag_pct = (fragmented_windows / total * 100) if total > 0 else 0.0

        return WindowAnalysisResult(
            total_windows=total,
            short_windows=short_windows,
            long_windows=long_windows,
            fragmented_windows=fragmented_windows,
            average_window_duration_hours=avg_duration,
            short_window_percentage=short_pct,
            long_window_percentage=long_pct,
            fragmentation_percentage=frag_pct,
        )

    def _calculate_gap_hours(
        self, episode1: SleepRecord, episode2: SleepRecord
    ) -> float:
        """
        Calculate gap between two episodes in hours.

        Returns negative value if episodes overlap.
        """
        if not episode1 or not episode2:
            return 0.0

        try:
            gap = episode2.start_date - episode1.end_date
            return gap.total_seconds() / 3600
        except Exception:
            return 0.0

    def _create_window(
        self, episodes: list[SleepRecord], gaps: list[float]
    ) -> SleepWindow:
        """
        Create a sleep window from episodes.

        Handles overlapping episodes by taking earliest start
        and latest end time.
        """
        # Get window boundaries
        start_time = min(e.start_date for e in episodes)
        end_time = max(e.end_date for e in episodes)

        # Calculate total sleep duration (handling overlaps)
        total_duration_hours = self._calculate_total_sleep_hours(episodes)

        return SleepWindow(
            start_time=start_time,
            end_time=end_time,
            episode_count=len(episodes),
            total_duration_hours=total_duration_hours,
            gap_hours=gaps,
        )

    def _calculate_total_sleep_hours(self, episodes: list[SleepRecord]) -> float:
        """
        Calculate total sleep hours, handling overlapping episodes.

        Uses interval merging algorithm to avoid double-counting overlaps.
        """
        if not episodes:
            return 0.0

        # Convert to intervals
        intervals = [(e.start_date, e.end_date) for e in episodes]
        intervals.sort()

        # Merge overlapping intervals
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_end = merged[-1][1]
            if start <= last_end:
                # Overlapping - extend the last interval
                merged[-1] = (merged[-1][0], max(last_end, end))
            else:
                # Non-overlapping - add new interval
                merged.append((start, end))

        # Sum durations
        total_seconds = sum((end - start).total_seconds() for start, end in merged)

        return total_seconds / 3600
