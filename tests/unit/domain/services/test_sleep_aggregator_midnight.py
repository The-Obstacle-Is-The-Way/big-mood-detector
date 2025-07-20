"""
Unit test for SleepAggregator midnight boundary handling.

This test ensures sleep calculations work correctly when sleep spans midnight,
which is the most common case for normal sleep patterns.
"""

from datetime import date, datetime, timedelta

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator


class TestSleepAggregatorMidnight:
    """Test sleep aggregation across midnight boundaries."""

    def setup_method(self):
        """Setup test fixtures."""
        self.aggregator = SleepAggregator()

    def test_normal_sleep_across_midnight(self):
        """
        Test normal sleep pattern: 10pm to 6am.

        This is the most common case and MUST work correctly.
        """
        # Create sleep record from 10pm to 6am next day
        sleep_start = datetime(2024, 1, 1, 22, 0)  # 10pm
        sleep_end = datetime(2024, 1, 2, 6, 0)  # 6am next day

        sleep_record = SleepRecord(
            source_name="Apple Watch",
            start_date=sleep_start,
            end_date=sleep_end,
            state=SleepState.ASLEEP,
        )

        # Aggregate
        summaries = self.aggregator.aggregate_daily([sleep_record])

        # Should be assigned to Jan 2 (the date of waking)
        assert date(2024, 1, 2) in summaries
        assert date(2024, 1, 1) not in summaries  # Not Jan 1

        summary = summaries[date(2024, 1, 2)]
        assert summary.total_sleep_hours == 8.0, "Should be exactly 8 hours"
        assert summary.date == date(2024, 1, 2)

    def test_late_night_sleep_after_midnight(self):
        """
        Test late night sleep: 2am to 10am.

        Some people sleep very late (after midnight).
        """
        # Sleep from 2am to 10am same day
        sleep_start = datetime(2024, 1, 2, 2, 0)  # 2am
        sleep_end = datetime(2024, 1, 2, 10, 0)  # 10am same day

        sleep_record = SleepRecord(
            source_name="Apple Watch",
            start_date=sleep_start,
            end_date=sleep_end,
            state=SleepState.ASLEEP,
        )

        # Aggregate
        summaries = self.aggregator.aggregate_daily([sleep_record])

        # Should be assigned to Jan 2
        assert date(2024, 1, 2) in summaries
        summary = summaries[date(2024, 1, 2)]
        assert summary.total_sleep_hours == 8.0

    def test_split_sleep_across_midnight(self):
        """
        Test split sleep with nap before midnight and main sleep after.

        Example: Evening nap 8-9pm, then main sleep 11pm-7am.
        """
        # Evening nap
        nap = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 20, 0),  # 8pm
            end_date=datetime(2024, 1, 1, 21, 0),  # 9pm
            state=SleepState.ASLEEP,
        )

        # Main sleep
        main_sleep = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 23, 0),  # 11pm
            end_date=datetime(2024, 1, 2, 7, 0),  # 7am next day
            state=SleepState.ASLEEP,
        )

        # Aggregate
        summaries = self.aggregator.aggregate_daily([nap, main_sleep])

        # Both should be assigned to Jan 2 (wake date of main sleep)
        assert date(2024, 1, 2) in summaries
        summary = summaries[date(2024, 1, 2)]

        # Total should be 1 hour (nap) + 8 hours (main) = 9 hours
        assert summary.total_sleep_hours == 9.0
        assert summary.sleep_sessions == 2

    def test_exactly_midnight_boundary(self):
        """
        Test sleep that starts or ends exactly at midnight.
        """
        # Sleep from exactly midnight to 8am
        sleep1 = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 2, 0, 0),  # Exactly midnight
            end_date=datetime(2024, 1, 2, 8, 0),  # 8am
            state=SleepState.ASLEEP,
        )

        # Sleep from 10pm to exactly midnight
        sleep2 = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 22, 0),  # 10pm
            end_date=datetime(2024, 1, 2, 0, 0),  # Exactly midnight
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([sleep1, sleep2])

        # Both sleep records should be on Jan 2 (both end before 3pm on Jan 2)
        assert date(2024, 1, 2) in summaries

        # Total for Jan 2 should be 8 + 2 = 10 hours
        assert summaries[date(2024, 1, 2)].total_sleep_hours == 10.0
        assert summaries[date(2024, 1, 2)].sleep_sessions == 2

    def test_multi_day_sleep_marathon(self):
        """
        Test extremely long sleep spanning multiple days.

        This is rare but can happen with illness or medication.
        """
        # Sleep for 36 hours straight (illness/recovery)
        marathon_sleep = SleepRecord(
            source_name="Apple Watch",
            start_date=datetime(2024, 1, 1, 20, 0),  # 8pm Jan 1
            end_date=datetime(2024, 1, 3, 8, 0),  # 8am Jan 3
            state=SleepState.ASLEEP,
        )

        summaries = self.aggregator.aggregate_daily([marathon_sleep])

        # Should be assigned to wake date (Jan 3)
        assert date(2024, 1, 3) in summaries
        assert summaries[date(2024, 1, 3)].total_sleep_hours == 36.0

    def test_week_of_normal_sleep_patterns(self):
        """
        Test a realistic week of sleep with midnight crossings.

        This ensures our aggregation works for real-world patterns.
        """
        sleep_records = []

        # Create a week of sleep, each night 10:30pm to 6:30am (8 hours)
        for day in range(7):
            base_date = date(2024, 1, 1) + timedelta(days=day)

            # Sleep from 10:30pm to 6:30am next day
            sleep_start = datetime.combine(base_date, datetime.min.time()).replace(
                hour=22, minute=30
            )
            sleep_end = sleep_start + timedelta(hours=8)  # 6:30am next day

            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=sleep_start,
                    end_date=sleep_end,
                    state=SleepState.ASLEEP,
                )
            )

        summaries = self.aggregator.aggregate_daily(sleep_records)

        # Should have 7 days of sleep (Jan 2-8, since sleep is assigned to wake date)
        assert len(summaries) == 7

        # Each day should have exactly 8 hours
        for day in range(2, 9):  # Jan 2-8
            target_date = date(2024, 1, day)
            assert target_date in summaries
            assert summaries[target_date].total_sleep_hours == 8.0

    def test_regression_reasonable_sleep_duration(self):
        """
        REGRESSION TEST: Ensure sleep durations are reasonable.

        This guards against the sleep_percentage * 24 bug.
        """
        # Create fragmented sleep that would trigger the bug
        sleep_records = []
        base_date = date(2024, 1, 1)

        # Night sleep: 11pm-3am (4 hours)
        sleep_records.append(
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime.combine(base_date, datetime.min.time()).replace(
                    hour=23
                ),
                end_date=datetime.combine(
                    base_date + timedelta(days=1), datetime.min.time()
                ).replace(hour=3),
                state=SleepState.ASLEEP,
            )
        )

        # Wake period 3am-4am

        # Back to sleep: 4am-7am (3 hours)
        sleep_records.append(
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime.combine(
                    base_date + timedelta(days=1), datetime.min.time()
                ).replace(hour=4),
                end_date=datetime.combine(
                    base_date + timedelta(days=1), datetime.min.time()
                ).replace(hour=7),
                state=SleepState.ASLEEP,
            )
        )

        summaries = self.aggregator.aggregate_daily(sleep_records)

        # Should have total of 7 hours (4 + 3), not some bogus calculation
        assert date(2024, 1, 2) in summaries
        total_hours = summaries[date(2024, 1, 2)].total_sleep_hours

        assert total_hours == 7.0, f"Expected 7 hours, got {total_hours}"
        assert 4.0 <= total_hours <= 12.0, (
            f"Sleep duration {total_hours}h outside reasonable range [4,12]. "
            "This may indicate a calculation bug!"
        )
