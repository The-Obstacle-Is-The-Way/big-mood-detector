#!/usr/bin/env python3
"""Profile aggregation pipeline performance to identify bottlenecks."""

import cProfile
import pstats
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationConfig,
    AggregationPipeline,
)
from big_mood_detector.application.services.optimized_aggregation_pipeline import (
    OptimizedAggregationPipeline,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord, HeartMetricType
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


def create_test_data(num_days: int = 60) -> tuple[
    list[SleepRecord],
    list[ActivityRecord],
    list[HeartRateRecord],
]:
    """Create realistic test data for profiling."""
    sleep_records = []
    activity_records = []
    heart_records = []
    
    start_date = date(2024, 1, 1)
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # Create sleep records (2-3 per night)
        for i in range(2):
            # Sleep starts at 10 PM and midnight
            start_hour = 22 + (i * 2)
            if start_hour >= 24:
                start_hour -= 24
                sleep_date = current_date
            else:
                sleep_date = current_date - timedelta(days=1)
                
            start_time = datetime.combine(
                sleep_date,
                datetime.min.time()
            ).replace(hour=start_hour, tzinfo=timezone.utc)
            
            end_time = start_time + timedelta(hours=3.5)
            
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=start_time,
                    end_date=end_time,
                    state=SleepState.ASLEEP,
                )
            )
        
        # Create activity records (one per hour)
        for hour in range(24):
            start_time = datetime.combine(
                current_date,
                datetime.min.time()
            ).replace(hour=hour, tzinfo=timezone.utc)
            
            # Simulate step counts
            steps = 0 if 23 <= hour or hour <= 6 else (hour * 100 + day * 10) % 2000
            
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    activity_type=ActivityType.STEP_COUNT,
                    start_date=start_time,
                    end_date=start_time + timedelta(hours=1),
                    value=steps,
                    unit="count",
                )
            )
        
        # Create heart rate records (every 5 minutes)
        for minute in range(0, 1440, 5):
            timestamp = datetime.combine(
                current_date,
                datetime.min.time()
            ).replace(tzinfo=timezone.utc) + timedelta(minutes=minute)
            
            # Simulate heart rate
            base_hr = 60 + (minute // 60) % 20
            hr_value = base_hr + (day % 10)
            
            heart_records.append(
                HeartRateRecord(
                    source_name="Apple Watch",
                    timestamp=timestamp,
                    metric_type=HeartMetricType.HEART_RATE,
                    value=hr_value,
                    unit="bpm",
                )
            )
    
    return sleep_records, activity_records, heart_records


def profile_pipeline(
    pipeline: Any,
    sleep_records: list[SleepRecord],
    activity_records: list[ActivityRecord],
    heart_records: list[HeartRateRecord],
    name: str,
) -> float:
    """Profile a pipeline and return execution time."""
    print(f"\n{'=' * 60}")
    print(f"Profiling {name}")
    print(f"{'=' * 60}")
    
    profiler = cProfile.Profile()
    
    start_time = time.time()
    profiler.enable()
    
    features = pipeline.aggregate_daily_features(
        sleep_records=sleep_records,
        activity_records=activity_records,
        heart_records=heart_records,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 29),  # 60 days
    )
    
    profiler.disable()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Features generated: {len(features)}")
    
    # Print top 20 functions by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nTop 20 functions by cumulative time:")
    stats.print_stats(20)
    
    # Print functions with highest self time
    stats.sort_stats('time')
    print("\nTop 10 functions by self time:")
    stats.print_stats(10)
    
    return execution_time


def main():
    """Main profiling function."""
    print("Creating test data...")
    sleep_records, activity_records, heart_records = create_test_data(60)
    
    print(f"Created:")
    print(f"  - {len(sleep_records)} sleep records")
    print(f"  - {len(activity_records)} activity records")
    print(f"  - {len(heart_records)} heart rate records")
    
    # Test with different configurations
    configs = [
        ("No expensive ops", AggregationConfig(
            enable_dlmo_calculation=False,
            enable_circadian_analysis=False,
        )),
        ("With circadian", AggregationConfig(
            enable_dlmo_calculation=False,
            enable_circadian_analysis=True,
        )),
        ("With DLMO", AggregationConfig(
            enable_dlmo_calculation=True,
            enable_circadian_analysis=False,
        )),
        ("All features", AggregationConfig(
            enable_dlmo_calculation=True,
            enable_circadian_analysis=True,
        )),
    ]
    
    for config_name, config in configs:
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config_name}")
        print(f"{'=' * 80}")
    
    # Profile original pipeline
    original_pipeline = AggregationPipeline(config=config)
    original_time = profile_pipeline(
        original_pipeline,
        sleep_records,
        activity_records,
        heart_records,
        "Original AggregationPipeline"
    )
    
    # Profile optimized pipeline
    optimized_pipeline = OptimizedAggregationPipeline(config=config)
    optimized_time = profile_pipeline(
        optimized_pipeline,
        sleep_records,
        activity_records,
        heart_records,
        "Optimized AggregationPipeline"
    )
    
    # Compare results
    print(f"\n{'=' * 60}")
    print("Performance Comparison")
    print(f"{'=' * 60}")
    print(f"Original time: {original_time:.2f}s")
    print(f"Optimized time: {optimized_time:.2f}s")
    print(f"Speedup: {original_time / optimized_time:.2f}x")
    print(f"Time saved: {original_time - optimized_time:.2f}s ({((original_time - optimized_time) / original_time * 100):.1f}%)")


if __name__ == "__main__":
    main()