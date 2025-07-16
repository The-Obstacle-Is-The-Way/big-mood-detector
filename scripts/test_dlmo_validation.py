#!/usr/bin/env python3
"""
Validate DLMO calculation against known test cases.

For a person sleeping 11pm-7am, DLMO should be around 9-10pm.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator


def create_regular_sleeper_data():
    """Create data for a regular sleeper (11pm-7am)."""
    sleep_records = []
    activity_records = []

    # 14 days of regular sleep
    for day in range(14):
        date_obj = datetime(2025, 5, 1) + timedelta(days=day)

        # Consistent sleep 23:00-07:00
        sleep_records.append(
            SleepRecord(
                source_name="test",
                start_date=date_obj.replace(hour=23, minute=0),
                end_date=(date_obj + timedelta(days=1)).replace(hour=7, minute=0),
                state=SleepState.ASLEEP_CORE,
            )
        )

        # Regular daytime activity pattern
        # Morning (7-9am): moderate activity
        for hour in [7, 8]:
            activity_records.append(
                ActivityRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=hour, minute=0),
                    end_date=date_obj.replace(hour=hour, minute=59),
                    activity_type=ActivityType.STEP_COUNT,
                    value=800,
                    unit="count",
                )
            )

        # Day (9am-5pm): high activity
        for hour in range(9, 17):
            activity_records.append(
                ActivityRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=hour, minute=0),
                    end_date=date_obj.replace(hour=hour, minute=59),
                    activity_type=ActivityType.STEP_COUNT,
                    value=1200,
                    unit="count",
                )
            )

        # Evening (5-10pm): moderate activity
        for hour in range(17, 22):
            activity_records.append(
                ActivityRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=hour, minute=0),
                    end_date=date_obj.replace(hour=hour, minute=59),
                    activity_type=ActivityType.STEP_COUNT,
                    value=600,
                    unit="count",
                )
            )

        # Late evening (10-11pm): low activity
        activity_records.append(
            ActivityRecord(
                source_name="test",
                start_date=date_obj.replace(hour=22, minute=0),
                end_date=date_obj.replace(hour=22, minute=59),
                activity_type=ActivityType.STEP_COUNT,
                value=200,
                unit="count",
            )
        )

    return sleep_records, activity_records


def create_shift_worker_data():
    """Create data for a night shift worker."""
    sleep_records = []
    activity_records = []

    # 14 days of shift work pattern
    for day in range(14):
        date_obj = datetime(2025, 5, 1) + timedelta(days=day)

        if day % 7 < 5:  # Work days (sleep 8am-4pm)
            sleep_records.append(
                SleepRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=8, minute=0),
                    end_date=date_obj.replace(hour=16, minute=0),
                    state=SleepState.ASLEEP_CORE,
                )
            )

            # Night activity (work)
            for hour in range(0, 7):
                activity_records.append(
                    ActivityRecord(
                        source_name="test",
                        start_date=date_obj.replace(hour=hour, minute=0),
                        end_date=date_obj.replace(hour=hour, minute=59),
                        activity_type=ActivityType.STEP_COUNT,
                        value=1000,
                        unit="count",
                    )
                )

            # Evening activity (preparing for work)
            for hour in range(17, 24):
                activity_records.append(
                    ActivityRecord(
                        source_name="test",
                        start_date=date_obj.replace(hour=hour, minute=0),
                        end_date=date_obj.replace(hour=hour, minute=59),
                        activity_type=ActivityType.STEP_COUNT,
                        value=800,
                        unit="count",
                    )
                )

        else:  # Off days (try to sleep at night)
            sleep_records.append(
                SleepRecord(
                    source_name="test",
                    start_date=date_obj.replace(hour=23, minute=0),
                    end_date=(date_obj + timedelta(days=1)).replace(hour=7, minute=0),
                    state=SleepState.ASLEEP_CORE,
                )
            )

            # Daytime activity
            for hour in range(8, 22):
                activity_records.append(
                    ActivityRecord(
                        source_name="test",
                        start_date=date_obj.replace(hour=hour, minute=0),
                        end_date=date_obj.replace(hour=hour, minute=59),
                        activity_type=ActivityType.STEP_COUNT,
                        value=900,
                        unit="count",
                    )
                )

    return sleep_records, activity_records


def debug_circadian_model(calculator, activity_records, target_date):
    """Debug the circadian model to understand the calculation."""
    print("\nDEBUG: Circadian Model Details")
    print("-" * 40)

    # Create activity profiles
    profiles = calculator._create_activity_profiles(
        activity_records, target_date, days=7
    )

    # Debug: Show raw activity values
    hourly_activity = calculator._calculate_hourly_activity(
        activity_records, target_date
    )
    max_activity = max([r.value for r in activity_records if r.value > 0])

    print(f"Max activity value: {max_activity}")
    print(f"Half max (threshold base): {max_activity/2}")
    print("\nRaw hourly activity:")
    for hour, activity in enumerate(hourly_activity):
        print(f"  Hour {hour:2d}: {activity:6.0f} steps")

    # Show thresholds
    print("\nActivity thresholds:")
    half_max = max_activity / 2.0
    for i, mult in enumerate(calculator.ACTIVITY_THRESHOLD_MULTIPLIERS):
        threshold = mult * half_max
        lux = calculator.ACTIVITY_LUX_LEVELS[i]
        print(f"  Threshold {i}: {threshold:6.0f} steps -> {lux:4.0f} lux")

    # Show activity to lux conversion
    print("\nActivity to Lux Conversion (last day):")
    last_profile = profiles[-1]
    for hour, lux in enumerate(last_profile.hourly_values):
        print(f"  Hour {hour:2d}: {lux:6.0f} lux")

    # Run model and get detailed output
    cbt_rhythm = calculator._run_circadian_model(profiles)

    print("\nCBT Rhythm:")
    for hour, cbt in cbt_rhythm:
        print(f"  Hour {hour:2d}: CBT = {cbt:6.3f}")

    # Find minima manually
    cbt_values = np.array([cbt for _, cbt in cbt_rhythm])
    min_idx = np.argmin(cbt_values)
    print(f"\nSimple minimum at hour {min_idx} (CBT = {cbt_values[min_idx]:.3f})")

    # Calculate DLMO
    dlmo_hour = (min_idx - 7.1) % 24
    print(
        f"DLMO = CBT_min - 7.1h = {dlmo_hour:.1f}h ({int(dlmo_hour)}:{int((dlmo_hour % 1) * 60):02d})"
    )


def main():
    """Test DLMO calculation with known patterns."""
    calc = DLMOCalculator()

    print("DLMO VALIDATION TEST")
    print("=" * 60)

    # Test 1: Regular sleeper
    print("\n1. REGULAR SLEEPER (11pm-7am)")
    print("-" * 60)

    sleep_records, activity_records = create_regular_sleeper_data()
    target_date = date(2025, 5, 14)

    result = calc.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=activity_records,
        target_date=target_date,
        days_to_model=14,  # More days for steady state
        use_activity=True,
    )

    if result:
        print(f"DLMO: {result.dlmo_time} ({result.dlmo_hour:.2f}h)")
        print(f"CBT Min: {result.cbt_min_hour:.2f}h")
        print(f"Confidence: {result.confidence:.2f}")

        # Check if in expected range
        expected_start = 20.0  # 8pm
        expected_end = 22.0  # 10pm

        if expected_start <= result.dlmo_hour <= expected_end:
            print(f"✓ DLMO within expected range ({expected_start}-{expected_end}h)")
        else:
            print(f"✗ DLMO outside expected range ({expected_start}-{expected_end}h)")
            debug_circadian_model(calc, activity_records, target_date)

    # Test 2: Shift worker
    print("\n\n2. SHIFT WORKER (Variable schedule)")
    print("-" * 60)

    sleep_records, activity_records = create_shift_worker_data()

    result = calc.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=activity_records,
        target_date=target_date,
        days_to_model=14,  # More days for steady state
        use_activity=True,
    )

    if result:
        print(f"DLMO: {result.dlmo_time} ({result.dlmo_hour:.2f}h)")
        print(f"CBT Min: {result.cbt_min_hour:.2f}h")
        print(f"Confidence: {result.confidence:.2f}")
        print("Note: Shift workers can have DLMO at any time of day")

    # Test 3: Compare methods
    print("\n\n3. METHOD COMPARISON")
    print("-" * 60)

    sleep_records, activity_records = create_regular_sleeper_data()

    # Activity-based
    result_activity = calc.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=activity_records,
        target_date=target_date,
        days_to_model=7,
        use_activity=True,
    )

    # Sleep-based
    result_sleep = calc.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=None,
        target_date=target_date,
        days_to_model=7,
        use_activity=False,
    )

    print(
        f"Activity-based DLMO: {result_activity.dlmo_time if result_activity else 'N/A'}"
    )
    print(f"Sleep-based DLMO: {result_sleep.dlmo_time if result_sleep else 'N/A'}")

    if result_activity and result_sleep:
        diff = abs(result_activity.dlmo_hour - result_sleep.dlmo_hour)
        if diff > 12:
            diff = 24 - diff
        print(f"Difference: {diff:.2f} hours")


if __name__ == "__main__":
    main()
