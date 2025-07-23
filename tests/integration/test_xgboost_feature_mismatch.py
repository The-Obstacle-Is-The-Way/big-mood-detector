"""
Test to expose the feature mismatch between our feature generator and XGBoost models.
This test should FAIL until we fix the pipeline to use AggregationPipeline.
"""
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.heart_rate_record import (
    HeartMetricType,
    HeartRateRecord,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
)
from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader


def create_sample_health_data(start_date, days=30):
    """Create sample health data for testing."""
    sleep_records = []
    activity_records = []
    heart_records = []

    for i in range(days):
        current_date = start_date + timedelta(days=i)

        # Create sleep records (night sleep)
        sleep_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=23)
        sleep_end = sleep_start + timedelta(hours=7.5)
        sleep_records.append(SleepRecord(
            source_name="test",
            start_date=sleep_start,
            end_date=sleep_end,
            state=SleepState.ASLEEP
        ))

        # Create activity records (hourly steps)
        for hour in range(8, 22):  # 8 AM to 10 PM
            activity_start = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour)
            activity_end = activity_start + timedelta(hours=1)
            steps = 500 + (hour - 8) * 100  # Varying step counts
            activity_records.append(ActivityRecord(
                source_name="test",
                start_date=activity_start,
                end_date=activity_end,
                activity_type=ActivityType.STEP_COUNT,
                value=float(steps),
                unit="count"
            ))

        # Create heart rate records
        for hour in range(24):
            hr_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour)
            hr_value = 60 + (10 if 8 <= hour <= 20 else -10)  # Higher during day
            heart_records.append(HeartRateRecord(
                source_name="test",
                timestamp=hr_time,
                value=float(hr_value),
                unit="bpm",
                metric_type=HeartMetricType.RESTING_HEART_RATE
            ))

    return sleep_records, activity_records, heart_records


class TestXGBoostFeatureMismatch:
    """Tests to expose and fix the feature mismatch bug."""

    @pytest.mark.integration
    def test_clinical_features_fail_with_xgboost(self):
        """
        This test demonstrates the current bug - XGBoost fails with clinical features.
        This should FAIL with 'missing fields' error.
        """
        # Arrange: Create sample health data
        sleep_records, activity_records, heart_records = create_sample_health_data(
            start_date=date(2025, 6, 1),
            days=30
        )

        # Create clinical feature extractor (WRONG approach)
        clinical_extractor = ClinicalFeatureExtractor()

        # Extract features using clinical extractor
        clinical_features = clinical_extractor.extract_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            target_date=date(2025, 6, 30)
        )

        # Load XGBoost models
        xgboost_loader = XGBoostModelLoader()
        model_path = Path("model_weights/xgboost")

        if model_path.exists():
            xgboost_loader.load_all_models(model_path)

            # Act & Assert: This should fail
            feature_array = np.array(clinical_features.to_xgboost_features())

            # This will fail because features don't match expected names
            with pytest.raises(Exception) as exc_info:
                xgboost_loader.predict(feature_array)

            assert "missing fields" in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_aggregation_features_work_with_xgboost(self):
        """
        This test shows the correct approach - using AggregationPipeline.
        This should PASS once we fix the pipeline.
        """
        # Arrange: Create sample health data
        sleep_records, activity_records, heart_records = create_sample_health_data(
            start_date=date(2025, 6, 1),
            days=30
        )

        # Create aggregation pipeline (CORRECT approach)
        aggregation_pipeline = AggregationPipeline()

        # Generate daily features for the past 30 days
        daily_features_list = aggregation_pipeline.aggregate_daily_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
            start_date=date(2025, 6, 1),
            end_date=date(2025, 6, 30)
        )

        # Skip if no features generated
        if not daily_features_list:
            pytest.skip("No features generated from test data")

        # Get the latest day's features
        latest_features = daily_features_list[-1]
        feature_dict = latest_features.to_dict()

        # Load XGBoost models
        xgboost_loader = XGBoostModelLoader()
        model_path = Path("model_weights/xgboost")

        if not model_path.exists():
            pytest.skip("XGBoost models not found")

        xgboost_loader.load_all_models(model_path)

        # Act: Convert to array and predict
        # Remove 'date' key as it's not a feature
        feature_dict.pop('date', None)

        # Verify we have all expected features
        expected_features = xgboost_loader.FEATURE_NAMES
        missing_features = set(expected_features) - set(feature_dict.keys())

        assert not missing_features, f"Missing features: {missing_features}"

        # Create feature array in correct order
        feature_array = xgboost_loader.dict_to_array(feature_dict)

        # This should work!
        prediction = xgboost_loader.predict(feature_array)

        # Assert: Valid prediction
        assert prediction is not None
        assert 0 <= prediction.depression_risk <= 1
        assert 0 <= prediction.hypomanic_risk <= 1
        assert 0 <= prediction.manic_risk <= 1
        assert 0 <= prediction.confidence <= 1

    @pytest.mark.unit
    def test_feature_name_mapping(self):
        """
        Unit test to verify feature name mapping between pipelines.
        """
        # The features XGBoost expects
        xgboost_expected = XGBoostModelLoader.FEATURE_NAMES

        # Create a mock DailyFeatures to test to_dict mapping
        from big_mood_detector.application.services.aggregation_pipeline import (
            DailyFeatures,
        )

        daily_features = DailyFeatures(
            date=date(2025, 6, 30),
            # Sleep features
            sleep_percentage_mean=0.3,
            sleep_percentage_std=0.1,
            sleep_percentage_zscore=0.5,
            sleep_amplitude_mean=0.2,
            sleep_amplitude_std=0.05,
            sleep_amplitude_zscore=-0.3,
            # Long sleep features
            long_sleep_num_mean=1.5,
            long_sleep_num_std=0.5,
            long_sleep_num_zscore=0.2,
            long_sleep_len_mean=7.5,
            long_sleep_len_std=1.0,
            long_sleep_len_zscore=0.1,
            long_sleep_st_mean=6.5,
            long_sleep_st_std=0.8,
            long_sleep_st_zscore=0.0,
            long_sleep_wt_mean=1.0,
            long_sleep_wt_std=0.2,
            long_sleep_wt_zscore=-0.1,
            # Short sleep features
            short_sleep_num_mean=0.5,
            short_sleep_num_std=0.3,
            short_sleep_num_zscore=-0.5,
            short_sleep_len_mean=2.0,
            short_sleep_len_std=0.5,
            short_sleep_len_zscore=-0.3,
            short_sleep_st_mean=1.8,
            short_sleep_st_std=0.4,
            short_sleep_st_zscore=-0.2,
            short_sleep_wt_mean=0.2,
            short_sleep_wt_std=0.1,
            short_sleep_wt_zscore=-0.4,
            # Circadian features
            circadian_amplitude_mean=0.7,
            circadian_amplitude_std=0.1,
            circadian_amplitude_zscore=0.3,
            circadian_phase_mean=22.0,
            circadian_phase_std=1.5,
            circadian_phase_zscore=0.0
        )

        # Get the dictionary representation
        feature_dict = daily_features.to_dict()
        feature_dict.pop('date', None)

        # Verify all expected features are present
        for expected_name in xgboost_expected:
            assert expected_name in feature_dict, f"Missing XGBoost feature: {expected_name}"

        # Verify the mapping is correct
        assert feature_dict["sleep_percentage_MN"] == 0.3
        assert feature_dict["sleep_percentage_SD"] == 0.1
        assert feature_dict["sleep_percentage_Z"] == 0.5
