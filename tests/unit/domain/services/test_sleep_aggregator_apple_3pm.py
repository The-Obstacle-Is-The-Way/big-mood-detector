"""
Test for Apple Health 3pm cutoff date assignment.

Apple Health assigns sleep to the date you wake up, with a 3pm cutoff.
Sleep that ends before 3pm is assigned to that day.
Sleep that ends after 3pm is assigned to the next day.
"""

from datetime import date, datetime

class TestSleepAggregatorApple3PM:
    """Test Apple Health 3pm cutoff convention for sleep date assignment."""

    def setup_method(self):
        """Setup test fixtures."""
        self.aggregator = SleepAggregator()

    def test_normal_night_sleep_assigned_to_wake_date(self):
        """
        Normal night sleep (10pm-6am) should be assigned to wake date.

        This is the most common case.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        sleep = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 22, 0),  # 10pm Jan 1
            end_date=datetime(2024, 1, 2, 6, 0),  # 6am Jan 2
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([sleep])

        # Should be assigned to Jan 2 (wake date)
        assert date(2024, 1, 2) in summaries
        assert date(2024, 1, 1) not in summaries

    def test_morning_nap_before_3pm_same_day(self):
        """
        Morning nap ending before 3pm stays on same day.

        Example: 10am-1pm nap on Jan 2.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        nap = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 10, 0),  # 10am
            end_date=datetime(2024, 1, 2, 13, 0),  # 1pm (before 3pm)
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([nap])

        # Should stay on Jan 2
        assert date(2024, 1, 2) in summaries
        assert summaries[date(2024, 1, 2)].total_sleep_hours == 3.0

    def test_afternoon_nap_after_3pm_next_day(self):
        """
        Afternoon nap ending after 3pm goes to next day.

        Example: 2pm-4pm nap on Jan 2.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        nap = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 14, 0),  # 2pm
            end_date=datetime(2024, 1, 2, 16, 0),  # 4pm (after 3pm)
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([nap])

        # Should be assigned to Jan 3 (next day)
        assert date(2024, 1, 3) in summaries
        assert date(2024, 1, 2) not in summaries

    def test_exactly_3pm_boundary(self):
        """
        Test sleep ending exactly at 3pm (edge case).

        Convention: exactly 3pm should stay on same day.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        nap = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 13, 0),  # 1pm
            end_date=datetime(2024, 1, 2, 15, 0),  # Exactly 3pm
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([nap])

        # Should stay on Jan 2 (3pm is inclusive for same day)
        assert date(2024, 1, 2) in summaries

    def test_shift_worker_pattern(self):
        """
        Test shift worker sleeping during day.

        Example: Night shift worker sleeps 8am-4pm.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        shift_sleep = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 8, 0),  # 8am
            end_date=datetime(2024, 1, 2, 16, 0),  # 4pm (after 3pm)
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([shift_sleep])

        # Should be assigned to Jan 3 (ends after 3pm)
        assert date(2024, 1, 3) in summaries
        assert summaries[date(2024, 1, 3)].total_sleep_hours == 8.0

    def test_multiple_sleep_sessions_same_day(self):
        """
        Test multiple sleep sessions with 3pm rule.

        Example: Night sleep + morning nap + afternoon nap.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        # Night sleep: 11pm Jan 1 to 7am Jan 2
        night = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 23, 0),
            end_date=datetime(2024, 1, 2, 7, 0),
            state=SleepState.ASLEEP,
        )

        # Morning nap: 10am-11am Jan 2 (before 3pm)
        morning_nap = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 10, 0),
            end_date=datetime(2024, 1, 2, 11, 0),
            state=SleepState.ASLEEP,
        )

        # Afternoon nap: 4pm-5pm Jan 2 (after 3pm)
        afternoon_nap = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 16, 0),
            end_date=datetime(2024, 1, 2, 17, 0),
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([night, morning_nap, afternoon_nap])

        # Night sleep + morning nap -> Jan 2
        assert date(2024, 1, 2) in summaries
        assert summaries[date(2024, 1, 2)].total_sleep_hours == 9.0  # 8 + 1

        # Afternoon nap -> Jan 3
        assert date(2024, 1, 3) in summaries
        assert summaries[date(2024, 1, 3)].total_sleep_hours == 1.0

    def test_very_long_sleep_crossing_3pm(self):
        """
        Test very long sleep that crosses 3pm boundary.

        Example: Sick day sleeping 10pm to 5pm next day.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        long_sleep = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 22, 0),  # 10pm Jan 1
            end_date=datetime(2024, 1, 2, 17, 0),  # 5pm Jan 2 (after 3pm)
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([long_sleep])

        # Should be assigned to Jan 3 (ends after 3pm on Jan 2)
        assert date(2024, 1, 3) in summaries
        assert summaries[date(2024, 1, 3)].total_sleep_hours == 19.0
