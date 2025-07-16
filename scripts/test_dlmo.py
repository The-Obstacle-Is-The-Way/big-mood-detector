#!/usr/bin/env python3
"""
Test DLMO calculation to debug constant values.
"""

import sys
from datetime import date, datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator


def create_test_sleep_patterns():
    """Create different sleep patterns to test DLMO variation."""
    patterns = {
        "normal": [
            # Normal sleep pattern (11pm - 7am)
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 10, 23, 0),
                end_date=datetime(2025, 5, 11, 7, 0),
                state=SleepState.ASLEEP_CORE,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 11, 23, 30),
                end_date=datetime(2025, 5, 12, 7, 30),
                state=SleepState.ASLEEP_CORE,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 12, 22, 45),
                end_date=datetime(2025, 5, 13, 6, 45),
                state=SleepState.ASLEEP_CORE,
            ),
        ],
        "late": [
            # Late sleep pattern (2am - 10am)
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 11, 2, 0),
                end_date=datetime(2025, 5, 11, 10, 0),
                state=SleepState.ASLEEP_CORE,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 12, 2, 30),
                end_date=datetime(2025, 5, 12, 10, 30),
                state=SleepState.ASLEEP_CORE,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 13, 1, 45),
                end_date=datetime(2025, 5, 13, 9, 45),
                state=SleepState.ASLEEP_CORE,
            ),
        ],
        "early": [
            # Early sleep pattern (8pm - 4am)
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 10, 20, 0),
                end_date=datetime(2025, 5, 11, 4, 0),
                state=SleepState.ASLEEP_CORE,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 11, 20, 30),
                end_date=datetime(2025, 5, 12, 4, 30),
                state=SleepState.ASLEEP_CORE,
            ),
            SleepRecord(
                source_name="test",
                start_date=datetime(2025, 5, 12, 19, 45),
                end_date=datetime(2025, 5, 13, 3, 45),
                state=SleepState.ASLEEP_CORE,
            ),
        ],
    }
    return patterns


def main():
    """Test DLMO calculation with different sleep patterns."""
    calculator = DLMOCalculator()
    patterns = create_test_sleep_patterns()
    target_date = date(2025, 5, 13)

    print("DLMO Calculation Test")
    print("=" * 50)
    print(f"Target date: {target_date}")
    print()

    for pattern_name, sleep_records in patterns.items():
        print(f"\n{pattern_name.upper()} Sleep Pattern:")

        # Show sleep times
        for i, record in enumerate(sleep_records[-3:], 1):
            print(
                f"  Day {i}: {record.start_date.strftime('%H:%M')} - {record.end_date.strftime('%H:%M')}"
            )

        # Calculate DLMO
        result = calculator.calculate_dlmo(sleep_records, target_date, days_to_model=3)

        if result:
            print("\nResults:")
            print(f"  DLMO Time: {result.dlmo_time} ({result.dlmo_hour:.2f} hours)")
            print(f"  CBT Minimum: {result.cbt_min_hour:.2f} hours")
            print(f"  CBT Amplitude: {result.cbt_amplitude:.4f}")
            print(f"  Expected DLMO range for {pattern_name}: ", end="")

            # Expected DLMO times based on sleep patterns
            if pattern_name == "normal":
                print("20:00-22:00 (8-10pm)")
            elif pattern_name == "late":
                print("23:00-01:00 (11pm-1am)")
            elif pattern_name == "early":
                print("17:00-19:00 (5-7pm)")
        else:
            print("  ERROR: No DLMO calculated")

    print("\n" + "=" * 50)
    print("Analysis:")
    print("- DLMO should occur ~2-3 hours before habitual sleep onset")
    print("- Different sleep patterns should produce different DLMO times")
    print("- If all DLMO times are similar, the model isn't responding to sleep timing")


if __name__ == "__main__":
    main()
