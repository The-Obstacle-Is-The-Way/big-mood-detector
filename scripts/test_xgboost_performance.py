#!/usr/bin/env python3
"""Test XGBoost pipeline performance with large dataset."""

import logging
import time
from datetime import UTC, datetime, timedelta, date
from pathlib import Path

from big_mood_detector.application.pipelines.xgboost_pipeline import XGBoostPipeline
from big_mood_detector.application.services.seoul_feature_extractor import SeoulFeatureExtractor
from big_mood_detector.application.validators.pipeline_validators import XGBoostValidator
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord, HeartMetricType
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostMoodPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_large_dataset(days: int = 60, records_per_day: int = 1000):
    """Create a large synthetic dataset to test performance."""
    logger.info(f"Creating {days} days with {records_per_day} HR records/day = {days * records_per_day} total")
    
    sleep_records = []
    activity_records = []
    heart_records = []
    
    base_date = datetime(2025, 6, 1, tzinfo=UTC)
    
    for day in range(days):
        current_date = base_date + timedelta(days=day)
        
        # Sleep record (one per night)
        sleep_records.append(
            SleepRecord(
                source_name="Apple Watch",
                start_date=current_date.replace(hour=22),
                end_date=(current_date + timedelta(days=1)).replace(hour=6),
                state=SleepState.ASLEEP,
            )
        )
        
        # Activity record (one per day)
        activity_records.append(
            ActivityRecord(
                source_name="iPhone",
                start_date=current_date.replace(hour=0),
                end_date=current_date.replace(hour=23, minute=59),
                activity_type=ActivityType.STEP_COUNT,
                value=8000.0,
                unit="count",
            )
        )
        
        # Many heart rate records throughout the day
        for hour in range(24):
            for minute in range(0, 60, 5):  # Every 5 minutes
                heart_records.append(
                    HeartRateRecord(
                        source_name="Apple Watch",
                        timestamp=current_date.replace(hour=hour, minute=minute),
                        metric_type=HeartMetricType.HEART_RATE,
                        value=70.0 + (hour * 0.5),  # Vary by hour
                        unit="count/min",
                    )
                )
    
    return sleep_records, activity_records, heart_records


def main():
    """Test XGBoost pipeline performance."""
    # Create pipeline
    try:
        predictor = XGBoostMoodPredictor()
        predictor.load_models(Path("model_weights/xgboost/converted"))
    except Exception as e:
        logger.warning(f"Could not load XGBoost models: {e}")
        logger.info("Creating mock predictor for performance testing")
        predictor = None
    
    feature_extractor = SeoulFeatureExtractor()
    validator = XGBoostValidator()
    
    pipeline = XGBoostPipeline(
        feature_extractor=feature_extractor,
        predictor=predictor,
        validator=validator,
    )
    
    # Test with different dataset sizes
    for days in [30, 60, 90]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with {days} days of data")
        logger.info(f"{'='*60}")
        
        sleep_records, activity_records, heart_records = create_large_dataset(days=days)
        
        total_records = len(sleep_records) + len(activity_records) + len(heart_records)
        logger.info(f"Total records: {total_records:,}")
        
        target_date = date(2025, 6, 1) + timedelta(days=days-1)
        
        # Time the feature extraction
        start_time = time.time()
        
        if predictor:
            result = pipeline.process(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                target_date=target_date,
            )
        else:
            # Just test feature extraction
            seoul_features = feature_extractor.extract_seoul_features(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                target_date=target_date,
            )
            result = seoul_features
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Processing time: {duration:.2f} seconds")
        logger.info(f"Records/second: {total_records/duration:,.0f}")
        
        if result:
            logger.info("✓ Feature extraction successful")
        else:
            logger.error("✗ Feature extraction failed")
        
        # Warn if approaching timeout
        if duration > 30:
            logger.warning(f"⚠️  Processing took {duration:.1f}s - approaching 60s timeout!")


if __name__ == "__main__":
    main()