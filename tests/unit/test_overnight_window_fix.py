"""
Test to ensure overnight sleep windows are correctly assigned to dates.

The Seoul paper assigns sleep windows based on the wake day, not the start day.
A sleep period from 22:00 Jan 1 to 06:00 Jan 2 should count for Jan 2.
"""

from datetime import date, datetime

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.sleep_window_analyzer import SleepWindowAnalyzer


class TestOvernightWindowFix:
    """Test that overnight sleep windows are assigned to the correct date."""

    def test_overnight_window_counts_for_wake_day(self):
        """Test that sleep from 22:00-06:00 counts for the wake day (Jan 2)."""
        analyzer = SleepWindowAnalyzer()

        # Create overnight sleep record
        sleep_record = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 1, 22, 0),  # 10 PM Jan 1
            end_date=datetime(2025, 1, 2, 6, 0),     # 6 AM Jan 2
            state=SleepState.ASLEEP,
        )

        # Analyze for Jan 2 (wake day)
        windows = analyzer.analyze_sleep_episodes([sleep_record], date(2025, 1, 2))

        # Should find the window for Jan 2
        assert len(windows) == 1
        assert windows[0].total_duration_hours == 8.0

        # Analyze for Jan 1 (sleep start day)
        windows_jan1 = analyzer.analyze_sleep_episodes([sleep_record], date(2025, 1, 1))

        # Should NOT find the window for Jan 1 (current bug: it does)
        assert len(windows_jan1) == 0

    def test_very_long_sleep_uses_midpoint_rule(self):
        """Test that even very long sleep sessions use the midpoint rule."""
        analyzer = SleepWindowAnalyzer()

        # Create a 16-hour depression sleep (8 PM to noon next day)
        sleep_record = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 1, 20, 0),  # 8 PM Jan 1
            end_date=datetime(2025, 1, 2, 12, 0),     # Noon Jan 2
            state=SleepState.ASLEEP,
        )

        # Midpoint is 4 AM Jan 2, so it should count for Jan 2
        windows_jan1 = analyzer.analyze_sleep_episodes([sleep_record], date(2025, 1, 1))
        assert len(windows_jan1) == 0

        windows_jan2 = analyzer.analyze_sleep_episodes([sleep_record], date(2025, 1, 2))
        assert len(windows_jan2) == 1
        assert windows_jan2[0].total_duration_hours == 16.0

    def test_nap_follows_midpoint_rule(self):
        """Test that naps follow the midpoint rule correctly."""
        analyzer = SleepWindowAnalyzer()

        # Morning nap (midpoint 10:45 AM - closer to today's midnight)
        morning_nap = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 2, 10, 0),  # 10 AM
            end_date=datetime(2025, 1, 2, 11, 30),   # 11:30 AM
            state=SleepState.ASLEEP,
        )

        # Afternoon nap (midpoint 2:45 PM - closer to tomorrow's midnight)
        afternoon_nap = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 2, 14, 0),  # 2 PM
            end_date=datetime(2025, 1, 2, 15, 30),   # 3:30 PM
            state=SleepState.ASLEEP,
        )

        # Morning nap should count for Jan 2
        windows = analyzer.analyze_sleep_episodes([morning_nap], date(2025, 1, 2))
        assert len(windows) == 1

        # Afternoon nap should count for Jan 3 (closer to next midnight)
        windows = analyzer.analyze_sleep_episodes([afternoon_nap], date(2025, 1, 3))
        assert len(windows) == 1

    def test_seoul_paper_midpoint_rule(self):
        """Test the Seoul paper's rule: assign to nearest midnight of midpoint."""
        analyzer = SleepWindowAnalyzer()

        # Sleep with midpoint before noon (should go to current day)
        early_sleep = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 1, 6, 0),   # 6 AM
            end_date=datetime(2025, 1, 1, 10, 0),    # 10 AM (midpoint 8 AM)
            state=SleepState.ASLEEP,
        )

        # Sleep with midpoint after midnight (should go to Jan 2)
        late_sleep = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 1, 22, 0),  # 10 PM
            end_date=datetime(2025, 1, 2, 6, 0),     # 6 AM (midpoint 2 AM)
            state=SleepState.ASLEEP,
        )

        # Check early sleep goes to Jan 1
        windows_jan1 = analyzer.analyze_sleep_episodes([early_sleep], date(2025, 1, 1))
        assert len(windows_jan1) == 1

        # Check late sleep goes to Jan 2
        windows_jan2 = analyzer.analyze_sleep_episodes([late_sleep], date(2025, 1, 2))
        assert len(windows_jan2) == 1
