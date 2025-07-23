"""
Integration test to verify XGBoost models work with Seoul features from AggregationPipeline.
This test demonstrates the correct approach vs the current bug.
"""
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from big_mood_detector.application.services.aggregation_pipeline import (
    AggregationPipeline,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState
from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostModelLoader


class TestXGBoostSeoulFeatures:
    """Test that XGBoost models work correctly with Seoul features."""

    @pytest.fixture
    def sample_records(self):
        """Create 30 days of sample health records."""
        sleep_records = []
        activity_records = []

        base_date = date(2025, 6, 1)

        for i in range(30):
            current_date = base_date + timedelta(days=i)

            # Sleep record (7-8 hours per night)
            sleep_start = datetime.combine(current_date - timedelta(days=1),
                                         datetime.min.time()) + timedelta(hours=23)
            sleep_hours = 7.0 + (i % 3) * 0.5  # Vary between 7-8 hours
            sleep_end = sleep_start + timedelta(hours=sleep_hours)

            sleep_records.append(SleepRecord(
                source_name="test",
                start_date=sleep_start,
                end_date=sleep_end,
                state=SleepState.ASLEEP
            ))

            # Activity records (simulate daily steps)
            for hour in range(8, 20):  # 8 AM to 8 PM
                activity_start = datetime.combine(current_date,
                                                datetime.min.time()) + timedelta(hours=hour)
                activity_end = activity_start + timedelta(hours=1)

                # More steps during midday
                if 12 <= hour <= 14:
                    steps = 800
                else:
                    steps = 400

                activity_records.append(ActivityRecord(
                    source_name="test",
                    start_date=activity_start,
                    end_date=activity_end,
                    activity_type=ActivityType.STEP_COUNT,
                    value=float(steps),
                    unit="count"
                ))

        return sleep_records, activity_records

    @pytest.mark.integration
    def test_aggregation_pipeline_generates_seoul_features(self, sample_records):
        """Verify AggregationPipeline generates all 36 Seoul features correctly."""
        sleep_records, activity_records = sample_records

        # Create aggregation pipeline
        pipeline = AggregationPipeline()

        # Generate Seoul features specifically for XGBoost
        daily_features = pipeline.aggregate_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],  # Empty for this test
            start_date=date(2025, 6, 1),
            end_date=date(2025, 6, 30)
        )

        # Should have features for multiple days
        print(f"Generated {len(daily_features)} days of features")
        assert len(daily_features) > 0

        # Get the last day's features
        latest_features = daily_features[-1]
        # Use to_xgboost_dict() to get only the 36 Seoul features
        feature_dict = latest_features.to_xgboost_dict()
        print(f"Feature dict keys: {list(feature_dict.keys())}")

        # Verify all Seoul features are present
        expected_features = XGBoostModelLoader.FEATURE_NAMES

        missing_features = set(expected_features) - set(feature_dict.keys())
        assert not missing_features, f"Missing Seoul features: {missing_features}"

        # Verify we have exactly 36 features
        assert len(feature_dict) == 36, f"Expected 36 features, got {len(feature_dict)}"

        # Spot check some values
        assert 0 <= feature_dict["sleep_percentage_MN"] <= 1
        assert feature_dict["sleep_percentage_SD"] >= 0
        assert isinstance(feature_dict["long_num_MN"], int | float)

    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("model_weights/xgboost").exists(),
        reason="XGBoost models not available"
    )
    def test_xgboost_prediction_with_seoul_features(self, sample_records):
        """Test that XGBoost models can make predictions with Seoul features."""
        sleep_records, activity_records = sample_records

        # Generate Seoul features via AggregationPipeline
        pipeline = AggregationPipeline()
        daily_features = pipeline.aggregate_seoul_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=[],
            start_date=date(2025, 6, 1),
            end_date=date(2025, 6, 30)
        )

        # Get latest features
        latest_features = daily_features[-1]
        feature_dict = latest_features.to_xgboost_dict()

        # Load XGBoost models
        xgboost_loader = XGBoostModelLoader()
        xgboost_loader.load_all_models(Path("model_weights/xgboost"))

        # Convert to array using model's expected order
        feature_array = xgboost_loader.dict_to_array(feature_dict)

        # Make prediction
        prediction = xgboost_loader.predict(feature_array)

        # Verify prediction is valid
        assert prediction is not None
        assert 0 <= prediction.depression_risk <= 1
        assert 0 <= prediction.hypomanic_risk <= 1
        assert 0 <= prediction.manic_risk <= 1
        assert 0 <= prediction.confidence <= 1

        # Should not be all default values
        assert not (prediction.depression_risk == 0.5 and
                   prediction.hypomanic_risk == 0.5 and
                   prediction.manic_risk == 0.5)

    @pytest.mark.unit
    def test_feature_name_mapping_is_correct(self):
        """Verify the feature name mapping between pipelines."""
        from big_mood_detector.application.services.aggregation_pipeline import (
            DailyFeatures,
        )

        # Create a DailyFeatures object with known values
        daily_features = DailyFeatures(
            date=date(2025, 6, 30),
            # Sleep percentage features
            sleep_percentage_mean=0.35,  # 35% of day sleeping
            sleep_percentage_std=0.05,
            sleep_percentage_zscore=0.2,
            # Sleep amplitude features
            sleep_amplitude_mean=0.15,
            sleep_amplitude_std=0.03,
            sleep_amplitude_zscore=-0.1,
            # Long sleep features
            long_sleep_num_mean=1.2,
            long_sleep_num_std=0.3,
            long_sleep_num_zscore=0.1,
            long_sleep_len_mean=7.8,
            long_sleep_len_std=0.9,
            long_sleep_len_zscore=0.3,
            long_sleep_st_mean=7.2,
            long_sleep_st_std=0.7,
            long_sleep_st_zscore=0.2,
            long_sleep_wt_mean=0.6,
            long_sleep_wt_std=0.2,
            long_sleep_wt_zscore=-0.1,
            # Short sleep features
            short_sleep_num_mean=0.3,
            short_sleep_num_std=0.2,
            short_sleep_num_zscore=-0.4,
            short_sleep_len_mean=1.8,
            short_sleep_len_std=0.6,
            short_sleep_len_zscore=-0.3,
            short_sleep_st_mean=1.5,
            short_sleep_st_std=0.5,
            short_sleep_st_zscore=-0.2,
            short_sleep_wt_mean=0.3,
            short_sleep_wt_std=0.15,
            short_sleep_wt_zscore=-0.5,
            # Circadian features
            circadian_amplitude_mean=0.65,
            circadian_amplitude_std=0.12,
            circadian_amplitude_zscore=0.4,
            circadian_phase_mean=22.5,
            circadian_phase_std=1.2,
            circadian_phase_zscore=0.1
        )

        # Convert to dict using XGBoost-specific method
        feature_dict = daily_features.to_xgboost_dict()

        # Verify the mapping produces XGBoost expected names
        assert feature_dict["sleep_percentage_MN"] == 0.35
        assert feature_dict["sleep_percentage_SD"] == 0.05
        assert feature_dict["sleep_percentage_Z"] == 0.2

        assert feature_dict["long_num_MN"] == 1.2
        assert feature_dict["long_ST_MN"] == 7.2  # Note uppercase ST
        assert feature_dict["circadian_phase_MN"] == 22.5

        # Verify all expected XGBoost features are present
        expected_features = XGBoostModelLoader.FEATURE_NAMES
        for feature in expected_features:
            assert feature in feature_dict, f"Missing mapped feature: {feature}"
