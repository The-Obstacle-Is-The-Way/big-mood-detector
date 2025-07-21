"""
Regression test for sleep aggregator to ensure sleep durations stay reasonable.

This guards against the sleep_percentage * 24 bug at the domain level.
"""

from datetime import date, datetime, timedelta

class TestSleepAggregatorRegression:
    """Regression tests to ensure sleep calculations remain reasonable."""

    def setup_method(self):
        """Setup test fixtures."""
        self.aggregator = SleepAggregator()

    def test_various_sleep_patterns_reasonable_duration(self):
        """
        Test that various sleep patterns produce reasonable durations.

        This guards against calculation bugs that would produce
        unrealistic sleep durations.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        test_cases = [
            # (start_time, end_time, expected_hours, description)
            (
                datetime(2024, 1, 1, 22, 0),  # 10pm
                datetime(2024, 1, 2, 6, 0),  # 6am next day
                8.0,
                "Normal 8h night sleep",
            ),
            (
                datetime(2024, 1, 1, 23, 0),  # 11pm
                datetime(2024, 1, 2, 5, 0),  # 5am next day
                6.0,
                "Short 6h night sleep",
            ),
            (
                datetime(2024, 1, 1, 21, 0),  # 9pm
                datetime(2024, 1, 2, 7, 0),  # 7am next day
                10.0,
                "Long 10h night sleep",
            ),
            (
                datetime(2024, 1, 2, 13, 0),  # 1pm
                datetime(2024, 1, 2, 14, 30),  # 2:30pm
                1.5,
                "Afternoon nap",
            ),
            (
                datetime(2024, 1, 2, 2, 0),  # 2am
                datetime(2024, 1, 2, 9, 0),  # 9am
                7.0,
                "Late night sleep",
            ),
        ]

        for start, end, expected_hours, description in test_cases:
            sleep_record = SleepRecord(
                source_name="Apple Watch",
                start_date=start,
                end_date=end,
                state=SleepState.ASLEEP,
            )

            summaries = self.aggregator.aggregate_daily([sleep_record])

            # Should have exactly one summary
            assert (
                len(summaries) == 1
            ), f"{description}: Expected 1 summary, got {len(summaries)}"

            # Get the summary (date depends on Apple 3pm rule)
            summary = list(summaries.values())[0]

            # Check duration matches expected
            assert (
                summary.total_sleep_hours == expected_hours
            ), f"{description}: Expected {expected_hours}h, got {summary.total_sleep_hours}h"

            # REGRESSION CHECK: Duration must be reasonable
            assert (
                0 < summary.total_sleep_hours <= 24
            ), f"{description}: Sleep duration {summary.total_sleep_hours}h is impossible!"

            # For typical sleep, check it's in normal range
            if "night sleep" in description:
                assert (
                    4 <= summary.total_sleep_hours <= 12
                ), f"{description}: Night sleep {summary.total_sleep_hours}h outside normal range [4,12]"

    def test_fragmented_sleep_total_correct(self):
        """
        Test fragmented sleep calculates total correctly.

        This was the core of the sleep_percentage * 24 bug.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        # Create fragmented sleep pattern
        sleep_records = [
            # First sleep segment: 11pm-3am (4 hours)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 1, 23, 0),
                end_date=datetime(2024, 1, 2, 3, 0),
                state=SleepState.ASLEEP,
            ),
            # Wake period: 3am-4am
            # Second sleep segment: 4am-7am (3 hours)
            SleepRecord(
                source_name="Apple Watch",
                start_date=datetime(2024, 1, 2, 4, 0),
                end_date=datetime(2024, 1, 2, 7, 0),
                state=SleepState.ASLEEP,
            ),
        ]

        summaries = self.aggregator.aggregate_daily(sleep_records)

        # Both segments should be on Jan 2 (wake before 3pm)
        assert date(2024, 1, 2) in summaries
        summary = summaries[date(2024, 1, 2)]

        # Total should be 4 + 3 = 7 hours
        assert (
            summary.total_sleep_hours == 7.0
        ), f"Fragmented sleep should total 7h, got {summary.total_sleep_hours}h"

        # Should have 2 sleep sessions
        assert summary.sleep_sessions == 2

        # REGRESSION: Must not calculate as sleep_percentage * 24
        # If it did, it would show something like 2-5 hours
        assert (
            summary.total_sleep_hours >= 6.0
        ), "Fragmented sleep calculation may have regressed!"

    def test_week_average_reasonable(self):
        """
        Test that a week of sleep averages to reasonable values.
        """
        from big_mood_detector.domain.entities.sleep_record import (
            SleepRecord,
            SleepState,
        )

        sleep_records = []

        # Create a week of normal sleep (7.5 hours average)
        for day in range(7):
            base_date = date(2024, 1, 1) + timedelta(days=day)

            # Vary sleep 7-8 hours
            sleep_hours = 7.5 + (0.5 if day % 2 == 0 else -0.5)

            start_time = datetime.combine(
                base_date - timedelta(days=1), datetime.min.time()
            ).replace(
                hour=22, minute=30
            )  # 10:30pm

            end_time = start_time + timedelta(hours=sleep_hours)

            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=start_time,
                    end_date=end_time,
                    state=SleepState.ASLEEP,
                )
            )

        summaries = self.aggregator.aggregate_daily(sleep_records)

        # Calculate average
        total_hours = sum(s.total_sleep_hours for s in summaries.values())
        avg_hours = total_hours / len(summaries)

        # Average should be around 7.5 hours
        assert (
            7.0 <= avg_hours <= 8.0
        ), f"Weekly average {avg_hours}h outside expected range [7,8]"

        # Each day should be reasonable
        for sleep_date, summary in summaries.items():
            assert (
                6.5 <= summary.total_sleep_hours <= 8.5
            ), f"Day {sleep_date} has unusual sleep: {summary.total_sleep_hours}h"
