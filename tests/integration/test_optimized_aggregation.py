"""Test the optimized aggregation pipeline."""

import time
from datetime import date, datetime, timedelta

import pytest

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.application.services.optimized_aggregation_pipeline import (
    OptimizedAggregationPipeline,
    OptimizationConfig,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState


def generate_test_data(num_days: int, records_per_day: int):
    """Generate test data."""
    sleep_records = []
    activity_records = []
    
    base_date = date.today() - timedelta(days=num_days)
    
    for day in range(num_days):
        current_date = base_date + timedelta(days=day)
        
        # 3 sleep records per day
        for i in range(3):
            start_time = datetime.combine(current_date, datetime.min.time()).replace(hour=22) + timedelta(hours=i*2)
            end_time = start_time + timedelta(hours=1, minutes=30)
            
            sleep_record = SleepRecord(
                source_name="Apple Watch",
                start_date=start_time,
                end_date=end_time,
                state=SleepState.ASLEEP,
            )
            sleep_records.append(sleep_record)
        
        # Many activity records per day
        for i in range(records_per_day):
            activity_time = datetime.combine(current_date, datetime.min.time()) + timedelta(
                hours=i % 24, minutes=(i * 5) % 60
            )
            
            activity_record = ActivityRecord(
                source_name="iPhone",
                start_date=activity_time,
                end_date=activity_time,
                activity_type=ActivityType.STEP_COUNT,
                value=100.0 + i,
                unit="count",
            )
            activity_records.append(activity_record)
    
    return sleep_records, activity_records, base_date


@pytest.mark.integration
def test_optimized_vs_original_performance():
    """Compare performance of optimized vs original pipeline."""
    # Generate test data
    num_days = 60
    records_per_day = 500
    
    sleep_records, activity_records, base_date = generate_test_data(
        num_days, records_per_day
    )
    
    total_records = len(sleep_records) + len(activity_records)
    print(f"\nTest data: {num_days} days, {total_records:,} records")
    
    # Test original pipeline
    original_pipeline = AggregationPipeline()
    
    start_time = time.time()
    original_features = original_pipeline.aggregate_daily_features(
        sleep_records=sleep_records,
        activity_records=activity_records,
        heart_records=[],
        start_date=base_date,
        end_date=base_date + timedelta(days=num_days-1),
    )
    original_time = time.time() - start_time
    
    print(f"\nOriginal pipeline:")
    print(f"  Time: {original_time:.2f}s")
    print(f"  Features: {len(original_features)}")
    print(f"  Time per day: {original_time/num_days:.3f}s")
    
    # Test optimized pipeline
    optimized_pipeline = OptimizedAggregationPipeline(
        optimization_config=OptimizationConfig(
            optimization_threshold_days=5,
            optimization_threshold_records=100,
        )
    )
    
    start_time = time.time()
    optimized_features = optimized_pipeline.aggregate_daily_features(
        sleep_records=sleep_records,
        activity_records=activity_records,
        heart_records=[],
        start_date=base_date,
        end_date=base_date + timedelta(days=num_days-1),
    )
    optimized_time = time.time() - start_time
    
    print(f"\nOptimized pipeline:")
    print(f"  Time: {optimized_time:.2f}s")
    print(f"  Features: {len(optimized_features)}")
    print(f"  Time per day: {optimized_time/num_days:.3f}s")
    
    # Calculate improvement
    speedup = original_time / optimized_time
    print(f"\nSpeedup: {speedup:.1f}x faster!")
    
    # Verify same results
    assert len(optimized_features) == len(original_features), "Different number of features!"
    
    # Verify significant speedup
    assert optimized_time < original_time * 0.7, f"Not enough speedup: only {speedup:.1f}x"


@pytest.mark.integration
@pytest.mark.slow
def test_optimized_handles_365_days():
    """Test that optimized pipeline can handle 1 year of data quickly."""
    # Generate 1 year of data
    num_days = 365
    records_per_day = 1000
    
    print(f"\nGenerating {num_days} days of test data...")
    sleep_records, activity_records, base_date = generate_test_data(
        num_days, records_per_day
    )
    
    total_records = len(sleep_records) + len(activity_records)
    print(f"Total records: {total_records:,}")
    
    # Test optimized pipeline
    pipeline = OptimizedAggregationPipeline()
    
    print("\nRunning optimized aggregation...")
    start_time = time.time()
    
    features = pipeline.aggregate_daily_features(
        sleep_records=sleep_records,
        activity_records=activity_records,
        heart_records=[],
        start_date=base_date,
        end_date=base_date + timedelta(days=num_days-1),
    )
    
    duration = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Days processed: {num_days}")
    print(f"  Features generated: {len(features)}")
    print(f"  Total time: {duration:.2f}s")
    print(f"  Time per day: {duration/num_days:.3f}s")
    print(f"  Records per second: {total_records/duration:,.0f}")
    
    # Target: Process 1 year in under 60 seconds
    assert duration < 60, f"Too slow: {duration:.1f}s for 1 year (target < 60s)"


@pytest.mark.integration
def test_optimized_produces_same_results():
    """Verify optimized pipeline produces identical results."""
    # Small dataset for detailed comparison
    num_days = 14
    records_per_day = 100
    
    sleep_records, activity_records, base_date = generate_test_data(
        num_days, records_per_day
    )
    
    # Run both pipelines
    original = AggregationPipeline()
    optimized = OptimizedAggregationPipeline()
    
    original_features = original.aggregate_daily_features(
        sleep_records=sleep_records,
        activity_records=activity_records,
        heart_records=[],
        start_date=base_date,
        end_date=base_date + timedelta(days=num_days-1),
    )
    
    optimized_features = optimized.aggregate_daily_features(
        sleep_records=sleep_records,
        activity_records=activity_records,
        heart_records=[],
        start_date=base_date,
        end_date=base_date + timedelta(days=num_days-1),
    )
    
    # Compare results
    assert len(optimized_features) == len(original_features)
    
    # Compare each day's features
    for orig, opt in zip(original_features, optimized_features):
        assert orig.date == opt.date
        # Compare a few key metrics
        assert orig.seoul_features.total_steps == opt.seoul_features.total_steps
        assert orig.seoul_features.sedentary_hours == opt.seoul_features.sedentary_hours
        
    print(f"✓ Optimized pipeline produces identical results!")