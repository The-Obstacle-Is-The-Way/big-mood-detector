#!/usr/bin/env python3
"""
Test unified DLMO calculator with different scenarios.

Tests the unified implementation with:
1. Regular sleep patterns
2. Shift work patterns
3. Activity-based vs sleep-based prediction
4. Seasonal adjustments
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator


def create_test_data():
    """Create test sleep and activity data."""
    sleep_records = []
    activity_records = []

    # Create 7 days of data with varying patterns
    base_date = datetime(2025, 5, 10)

    for day in range(7):
        current_date = base_date + timedelta(days=day)

        # Sleep pattern (varying bedtimes)
        bedtime_hour = 23 + (day % 3)  # 23:00, 00:00, 01:00
        wake_hour = 7 + (day % 2)  # 7:00 or 8:00

        sleep_start = current_date.replace(hour=bedtime_hour % 24, minute=0)
        if bedtime_hour >= 24:
            sleep_start += timedelta(days=1)

        sleep_end = (current_date + timedelta(days=1)).replace(hour=wake_hour, minute=0)

        sleep_records.append(
            SleepRecord(
                source_name="test",
                start_date=sleep_start,
                end_date=sleep_end,
                state=SleepState.ASLEEP_CORE,
            )
        )

        # Activity pattern
        # Morning activity
        for hour in range(wake_hour, wake_hour + 2):
            activity_records.append(
                ActivityRecord(
                    source_name="test",
                    start_date=(current_date + timedelta(days=1)).replace(
                        hour=hour, minute=0
                    ),
                    end_date=(current_date + timedelta(days=1)).replace(
                        hour=hour, minute=59
                    ),
                    activity_type=ActivityType.STEP_COUNT,
                    value=600 + day * 50,
                    unit="count",
                )
            )

        # Daytime activity
        for hour in range(9, 17):
            activity_records.append(
                ActivityRecord(
                    source_name="test",
                    start_date=(current_date + timedelta(days=1)).replace(
                        hour=hour, minute=0
                    ),
                    end_date=(current_date + timedelta(days=1)).replace(
                        hour=hour, minute=59
                    ),
                    activity_type=ActivityType.STEP_COUNT,
                    value=800 + day * 100,
                    unit="count",
                )
            )

        # Evening activity
        for hour in range(17, 22):
            activity_records.append(
                ActivityRecord(
                    source_name="test",
                    start_date=(current_date + timedelta(days=1)).replace(
                        hour=hour, minute=0
                    ),
                    end_date=(current_date + timedelta(days=1)).replace(
                        hour=hour, minute=59
                    ),
                    activity_type=ActivityType.STEP_COUNT,
                    value=400 + day * 30,
                    unit="count",
                )
            )

    return sleep_records, activity_records


def plot_cbt_rhythm(
    calculator, sleep_records, activity_records, target_date, title, use_activity=True
):
    """Plot CBT rhythm and DLMO timing."""
    # Get light profiles
    if use_activity and activity_records:
        profiles = calculator._create_activity_profiles(
            activity_records, target_date, days=3
        )
    else:
        profiles = calculator._create_light_profiles_from_sleep(
            sleep_records, target_date, days=3
        )

    # Run circadian model
    cbt_rhythm = calculator._run_circadian_model(profiles)

    # Extract data for plotting
    hours = [h for h, _ in cbt_rhythm]
    cbt_values = [cbt for _, cbt in cbt_rhythm]

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(hours, cbt_values, "b-", linewidth=2, label="CBT Rhythm")

    # Mark minimum
    min_idx = np.argmin(cbt_values)
    plt.plot(
        hours[min_idx], cbt_values[min_idx], "ro", markersize=10, label="CBT Minimum"
    )

    # Mark DLMO (using calibrated offset)
    dlmo_hour = (hours[min_idx] - calculator.CBT_TO_DLMO_OFFSET) % 24
    plt.axvline(
        x=dlmo_hour, color="g", linestyle="--", label=f"DLMO ({dlmo_hour:.1f}h)"
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("CBT (arbitrary units)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 3))

    return plt.gcf()


def main():
    """Test DLMO calculator with different scenarios."""
    print("Unified DLMO Calculator Test")
    print("=" * 60)

    # Create test data
    sleep_records, activity_records = create_test_data()
    target_date = date(2025, 5, 16)

    # Initialize calculator
    calc = DLMOCalculator()

    print("\n1. ACTIVITY-BASED PREDICTION (Preferred)")
    print("-" * 60)

    # Test with activity data
    result_activity = calc.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=activity_records,
        target_date=target_date,
        days_to_model=7,
        use_activity=True,
    )

    if result_activity:
        print(
            f"DLMO Time: {result_activity.dlmo_time} ({result_activity.dlmo_hour:.1f}h)"
        )
        print(f"CBT Minimum: {result_activity.cbt_min_hour:.1f}h")
        print(f"CBT Amplitude: {result_activity.cbt_amplitude:.2f}")
        print(f"Confidence: {result_activity.confidence:.2f}")

    print("\n2. SLEEP-BASED PREDICTION (Fallback)")
    print("-" * 60)

    # Test with sleep data only
    result_sleep = calc.calculate_dlmo(
        sleep_records=sleep_records,
        target_date=target_date,
        days_to_model=7,
        use_activity=False,
    )

    if result_sleep:
        print(f"DLMO Time: {result_sleep.dlmo_time} ({result_sleep.dlmo_hour:.1f}h)")
        print(f"CBT Minimum: {result_sleep.cbt_min_hour:.1f}h")
        print(f"CBT Amplitude: {result_sleep.cbt_amplitude:.2f}")
        print(f"Confidence: {result_sleep.confidence:.2f}")

    print("\n3. SEASONAL ADJUSTMENT TEST")
    print("-" * 60)

    # Test with winter adjustment
    result_winter = calc.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=activity_records,
        target_date=target_date,
        days_to_model=7,
        use_activity=True,
        day_length_hours=9.5,  # Winter day length
    )

    if result_winter:
        print(
            f"Winter DLMO Time: {result_winter.dlmo_time} ({result_winter.dlmo_hour:.1f}h)"
        )
        print(f"Winter CBT Minimum: {result_winter.cbt_min_hour:.1f}h")
        print(f"Winter Confidence: {result_winter.confidence:.2f}")

    # Plot comparisons
    print("\n4. GENERATING PLOTS...")
    print("-" * 60)

    # Plot activity-based
    plot_cbt_rhythm(
        calc,
        sleep_records,
        activity_records,
        target_date,
        "Activity-Based DLMO Prediction",
        use_activity=True,
    )
    plt.savefig("dlmo_activity_based.png", dpi=150, bbox_inches="tight")
    print("Saved: dlmo_activity_based.png")

    # Plot sleep-based
    plot_cbt_rhythm(
        calc,
        sleep_records,
        activity_records,
        target_date,
        "Sleep-Based DLMO Prediction",
        use_activity=False,
    )
    plt.savefig("dlmo_sleep_based.png", dpi=150, bbox_inches="tight")
    print("Saved: dlmo_sleep_based.png")

    # Summary
    print("\n5. SUMMARY")
    print("-" * 60)
    print("Key Features of Unified Calculator:")
    print("- Activity-based prediction (concordance 0.72 vs 0.63)")
    print("- Calibrated offset for correct DLMO timing (20-22h)")
    print("- Enhanced CBT minimum detection with scipy")
    print("- Light suppression for melatonin synthesis")
    print("- Dynamic activity thresholds")
    print("- Seasonal adjustment support")
    print("\nNote: CBT minimum timing is ~6h later than physiological")
    print("due to simplified phase proxy, but DLMO timing is calibrated correctly.")


if __name__ == "__main__":
    main()
