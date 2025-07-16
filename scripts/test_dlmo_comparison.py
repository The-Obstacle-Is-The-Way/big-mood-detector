#!/usr/bin/env python3
"""
Compare original and enhanced DLMO calculations.

Tests both implementations to show improvements in:
1. CBT minimum detection
2. Activity-based prediction
3. Seasonal adjustments
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator
from big_mood_detector.domain.services.dlmo_calculator_v2 import DLMOCalculatorV2


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
        wake_hour = 7 + (day % 2)       # 7:00 or 8:00
        
        sleep_start = current_date.replace(hour=bedtime_hour % 24, minute=0)
        if bedtime_hour >= 24:
            sleep_start += timedelta(days=1)
        
        sleep_end = (current_date + timedelta(days=1)).replace(hour=wake_hour, minute=0)
        
        sleep_records.append(SleepRecord(
            source_name="test",
            start_date=sleep_start,
            end_date=sleep_end,
            state=SleepState.ASLEEP_CORE
        ))
        
        # Activity pattern (realistic daily activity)
        # Morning activity
        for hour in range(7, 12):
            activity_records.append(ActivityRecord(
                source_name="test",
                start_date=current_date.replace(hour=hour, minute=0),
                end_date=current_date.replace(hour=hour, minute=59),
                activity_type=ActivityType.STEP_COUNT,
                value=np.random.randint(500, 1500),
                unit="count"
            ))
        
        # Afternoon activity (higher)
        for hour in range(12, 18):
            activity_records.append(ActivityRecord(
                source_name="test",
                start_date=current_date.replace(hour=hour, minute=0),
                end_date=current_date.replace(hour=hour, minute=59),
                activity_type=ActivityType.STEP_COUNT,
                value=np.random.randint(800, 2000),
                unit="count"
            ))
        
        # Evening activity (lower)
        for hour in range(18, 22):
            activity_records.append(ActivityRecord(
                source_name="test",
                start_date=current_date.replace(hour=hour, minute=0),
                end_date=current_date.replace(hour=hour, minute=59),
                activity_type=ActivityType.STEP_COUNT,
                value=np.random.randint(200, 800),
                unit="count"
            ))
    
    return sleep_records, activity_records


def visualize_cbt_rhythm(calculator, sleep_records, activity_records, target_date, title):
    """Visualize the CBT rhythm to show minimum detection."""
    # For V1, we need to access internal methods
    if isinstance(calculator, DLMOCalculator):
        # Create light profiles
        profiles = calculator._create_light_profiles(
            sleep_records, target_date, days_to_model=3
        )
        # Run model
        cbt_rhythm = calculator._run_circadian_model(profiles)
    else:
        # For V2, we can get more detailed output
        profiles = calculator._create_activity_profiles(
            activity_records, target_date, days=3
        )
        cbt_rhythm = calculator._run_circadian_model(profiles)
    
    # Extract data for plotting
    hours = [h for h, _ in cbt_rhythm]
    cbt_values = [cbt for _, cbt in cbt_rhythm]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(hours, cbt_values, 'b-', linewidth=2, label='CBT Rhythm')
    
    # Mark minimum
    min_idx = np.argmin(cbt_values)
    plt.plot(hours[min_idx], cbt_values[min_idx], 'ro', markersize=10, label='CBT Minimum')
    
    # Mark DLMO (7.1 hours before minimum)
    dlmo_hour = (hours[min_idx] - 7.1) % 24
    plt.axvline(x=dlmo_hour, color='g', linestyle='--', label=f'DLMO ({dlmo_hour:.1f}h)')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('CBT (arbitrary units)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 3))
    
    return plt.gcf()


def main():
    """Compare DLMO calculations."""
    print("DLMO Calculator Comparison")
    print("=" * 60)
    
    # Create test data
    sleep_records, activity_records = create_test_data()
    target_date = date(2025, 5, 16)
    
    # Initialize calculators
    calc_v1 = DLMOCalculator()
    calc_v2 = DLMOCalculatorV2()
    
    print("\n1. ORIGINAL CALCULATOR (Sleep-based, Simple Minimum)")
    print("-" * 60)
    
    result_v1 = calc_v1.calculate_dlmo(
        sleep_records,
        target_date,
        days_to_model=7
    )
    
    if result_v1:
        print(f"DLMO Time: {result_v1.dlmo_time} ({result_v1.dlmo_hour:.2f} hours)")
        print(f"CBT Minimum: {result_v1.cbt_min_hour:.2f} hours")
        print(f"CBT Amplitude: {result_v1.cbt_amplitude:.4f}")
    
    print("\n2. ENHANCED CALCULATOR V2 (Activity-based, Local Minima)")
    print("-" * 60)
    
    # Test with activity data
    result_v2_activity = calc_v2.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=activity_records,
        target_date=target_date,
        days_to_model=7,
        use_activity=True
    )
    
    if result_v2_activity:
        print(f"DLMO Time: {result_v2_activity.dlmo_time} ({result_v2_activity.dlmo_hour:.2f} hours)")
        print(f"CBT Minimum: {result_v2_activity.cbt_min_hour:.2f} hours")
        print(f"CBT Amplitude: {result_v2_activity.cbt_amplitude:.4f}")
        print(f"Confidence: {result_v2_activity.confidence:.2f}")
    
    print("\n3. ENHANCED CALCULATOR V2 (Sleep-based for comparison)")
    print("-" * 60)
    
    # Test without activity data (fallback to sleep)
    result_v2_sleep = calc_v2.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=None,
        target_date=target_date,
        days_to_model=7,
        use_activity=False
    )
    
    if result_v2_sleep:
        print(f"DLMO Time: {result_v2_sleep.dlmo_time} ({result_v2_sleep.dlmo_hour:.2f} hours)")
        print(f"CBT Minimum: {result_v2_sleep.cbt_min_hour:.2f} hours")
        print(f"CBT Amplitude: {result_v2_sleep.cbt_amplitude:.4f}")
        print(f"Confidence: {result_v2_sleep.confidence:.2f}")
    
    print("\n4. SEASONAL ADJUSTMENT TEST (Winter)")
    print("-" * 60)
    
    # Test with short day length (winter)
    result_v2_winter = calc_v2.calculate_dlmo(
        sleep_records=sleep_records,
        activity_records=activity_records,
        target_date=target_date,
        days_to_model=7,
        use_activity=True,
        day_length_hours=9.5  # Winter day
    )
    
    if result_v2_winter:
        print(f"DLMO Time: {result_v2_winter.dlmo_time} ({result_v2_winter.dlmo_hour:.2f} hours)")
        print(f"CBT Minimum: {result_v2_winter.cbt_min_hour:.2f} hours")
        print(f"CBT Amplitude: {result_v2_winter.cbt_amplitude:.4f}")
        print(f"Note: Light sensitivity doubled for winter adjustment")
    
    print("\n5. COMPARISON SUMMARY")
    print("-" * 60)
    
    if result_v1 and result_v2_activity:
        diff_hours = abs(result_v1.dlmo_hour - result_v2_activity.dlmo_hour)
        if diff_hours > 12:
            diff_hours = 24 - diff_hours
        
        print(f"DLMO Difference: {diff_hours:.2f} hours")
        print(f"Amplitude Ratio: {result_v2_activity.cbt_amplitude / result_v1.cbt_amplitude:.2f}x")
        
        # Expected DLMO range for normal sleepers
        print("\nExpected DLMO range for regular sleep pattern: 20:00-22:00 (8-10pm)")
        print("V1 accuracy: ", end="")
        if 20 <= result_v1.dlmo_hour <= 22:
            print("✓ Within expected range")
        else:
            print("✗ Outside expected range")
        
        print("V2 accuracy: ", end="")
        if 20 <= result_v2_activity.dlmo_hour <= 22:
            print("✓ Within expected range")
        else:
            print("✗ Outside expected range")
    
    # Visualize CBT rhythms
    print("\n6. GENERATING CBT RHYTHM PLOTS...")
    
    try:
        fig1 = visualize_cbt_rhythm(
            calc_v1, sleep_records, activity_records, target_date,
            "Original Calculator - CBT Rhythm"
        )
        fig1.savefig('output/cbt_rhythm_v1.png', dpi=150, bbox_inches='tight')
        print("Saved: output/cbt_rhythm_v1.png")
        
        fig2 = visualize_cbt_rhythm(
            calc_v2, sleep_records, activity_records, target_date,
            "Enhanced Calculator V2 - CBT Rhythm (Activity-based)"
        )
        fig2.savefig('output/cbt_rhythm_v2.png', dpi=150, bbox_inches='tight')
        print("Saved: output/cbt_rhythm_v2.png")
        
        plt.close('all')
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS IN V2:")
    print("1. Activity-based prediction (0.72 concordance vs 0.63)")
    print("2. Local minima detection with prominence threshold")
    print("3. Seasonal adjustment for winter months")
    print("4. Confidence scoring based on CBT prominence")
    print("5. Perfect for Apple Watch (no light sensor needed)")


if __name__ == '__main__':
    main()