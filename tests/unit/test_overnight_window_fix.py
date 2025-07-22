"""
Test to ensure overnight sleep windows are correctly assigned to dates.

The Seoul paper assigns sleep windows based on the wake day, not the start day.
A sleep period from 22:00 Jan 1 to 06:00 Jan 2 should count for Jan 2.
"""

import pytest
from datetime import datetime, date
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

    def test_multi_day_sleep_counts_for_all_overlapping_days(self):
        """Test that very long sleep sessions count for all days they overlap."""
        analyzer = SleepWindowAnalyzer()
        
        # Create a 36-hour sleep session (unusual but possible in depression)
        sleep_record = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 1, 20, 0),  # 8 PM Jan 1
            end_date=datetime(2025, 1, 3, 8, 0),     # 8 AM Jan 3
            state=SleepState.ASLEEP,
        )
        
        # Should count for Jan 1, 2, and 3
        for day in [1, 2, 3]:
            windows = analyzer.analyze_sleep_episodes(
                [sleep_record], 
                date(2025, 1, day)
            )
            assert len(windows) > 0, f"Sleep should count for Jan {day}"

    def test_nap_counts_for_its_day(self):
        """Test that daytime naps count for the day they occur."""
        analyzer = SleepWindowAnalyzer()
        
        # Afternoon nap
        nap = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 2, 14, 0),  # 2 PM
            end_date=datetime(2025, 1, 2, 15, 30),   # 3:30 PM
            state=SleepState.ASLEEP,
        )
        
        # Debug: check what date it's assigned to
        midpoint = nap.start_date + (nap.end_date - nap.start_date) / 2
        print(f"Nap midpoint: {midpoint}")
        
        # Should count for Jan 2
        windows = analyzer.analyze_sleep_episodes([nap], date(2025, 1, 2))
        assert len(windows) == 1
        assert windows[0].total_duration_hours == 1.5

    def test_seoul_paper_midpoint_rule(self):
        """Test the Seoul paper's rule: assign to nearest midnight of midpoint."""
        analyzer = SleepWindowAnalyzer()
        
        # Sleep with midpoint before midnight (should go to Jan 1)
        early_sleep = SleepRecord(
            source_name="test",
            start_date=datetime(2025, 1, 1, 20, 0),  # 8 PM
            end_date=datetime(2025, 1, 1, 23, 0),    # 11 PM (midpoint 9:30 PM)
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