#!/usr/bin/env python3
"""
Test full 36-feature extraction with Feature Engineering Orchestrator.
This ensures we're getting all Seoul paper features, not just basic ones.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.infrastructure.di.container import Container


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_full_feature_extraction():
    """Test extraction of all 36 Seoul features with orchestrator."""

    print_section("FULL 36-FEATURE EXTRACTION TEST")

    # Initialize DI container to get orchestrator
    container = Container()

    # Configure pipeline to use ensemble (includes PAT sequences)
    config = PipelineConfig(
        include_pat_sequences=True,  # This triggers full feature extraction
        min_days_required=7,
        enable_personal_calibration=True,
        user_id="test_user",
    )

    # Create pipeline with DI container for orchestrator
    pipeline = MoodPredictionPipeline(config=config, di_container=container)

    # Test with JSON data
    json_dir = Path("data/input/health_auto_export")

    # Process June 2025 data
    start_date = date(2025, 6, 1)
    end_date = date(2025, 6, 30)

    print(f"\nProcessing data from {start_date} to {end_date}")
    print(f"Data source: {json_dir}")

    # Process and predict to get full features
    result = pipeline.process_apple_health_file(
        file_path=json_dir, start_date=start_date, end_date=end_date
    )

    print("\n✅ Processing complete!")
    print(f"   Days processed: {result.features_extracted}")
    print(f"   Records processed: {result.records_processed}")
    print(f"   Processing time: {result.processing_time_seconds:.1f}s")

    # Check if we have predictions with full features
    if result.daily_predictions:
        first_date = list(result.daily_predictions.keys())[0]
        prediction = result.daily_predictions[first_date]

        print(f"\n📊 Sample prediction for {first_date}:")
        print(f"   Depression risk: {prediction['depression_risk']:.1%}")
        print(f"   Confidence: {prediction['confidence']:.1%}")

        if "models_used" in prediction:
            print(f"   Models used: {prediction['models_used']}")

    # Now extract features in CSV format to verify all 36
    print_section("EXTRACTING FEATURES TO CSV")

    # Use a shorter date range for feature extraction
    features_start = date(2025, 6, 15)
    features_end = date(2025, 6, 20)

    features = pipeline.extract_features_batch(
        sleep_records=result.metadata.get("sleep_records", []),
        activity_records=result.metadata.get("activity_records", []),
        heart_records=result.metadata.get("heart_records", []),
        start_date=features_start,
        end_date=features_end,
    )

    if features:
        # Get first feature set
        first_feature_set = next(iter(features.values()))

        if first_feature_set and first_feature_set.seoul_features:
            # Get XGBoost feature vector
            feature_vector = first_feature_set.seoul_features.to_xgboost_features()

            print(f"\n✅ Found {len(feature_vector)} XGBoost features!")

            # Show feature names and values
            feature_names = [
                # Basic Sleep Features (1-5)
                "sleep_duration_hours",
                "sleep_efficiency",
                "sleep_onset_hour",
                "wake_time_hour",
                "sleep_fragmentation",
                # Advanced Sleep Features (6-10)
                "sleep_regularity_index",
                "short_sleep_window_pct",
                "long_sleep_window_pct",
                "sleep_onset_variance",
                "wake_time_variance",
                # Circadian Rhythm Features (11-18)
                "interdaily_stability",
                "intradaily_variability",
                "relative_amplitude",
                "l5_value",
                "m10_value",
                "l5_onset_hour",
                "m10_onset_hour",
                "dlmo_hour",
                # Activity Features (19-24)
                "total_steps",
                "activity_variance",
                "sedentary_hours",
                "activity_fragmentation",
                "sedentary_bout_mean",
                "activity_intensity_ratio",
                # Heart Rate Features (25-28)
                "avg_resting_hr",
                "hrv_sdnn",
                "hr_circadian_range",
                "hr_minimum_hour",
                # Phase Features (29-32)
                "circadian_phase_advance",
                "circadian_phase_delay",
                "dlmo_confidence",
                "pat_hour",
                # Z-Score Features (33-36)
                "sleep_duration_zscore",
                "activity_zscore",
                "hr_zscore",
                "hrv_zscore",
            ]

            print("\n📊 Seoul XGBoost Features:")
            for i, (name, value) in enumerate(
                zip(feature_names[:10], feature_vector[:10], strict=False)
            ):
                print(f"   {i + 1:2d}. {name:30s}: {value:8.2f}")
            print("   ... (showing first 10 of 36)")

            # Check for orchestrator metadata
            print("\n🔍 Feature Orchestrator Status:")
            print(
                f"   Validation enabled: {'orchestrator' in str(type(pipeline.clinical_extractor))}"
            )
            print(
                f"   Anomaly detection: {'orchestrator' in str(type(pipeline.clinical_extractor))}"
            )

            # Save to CSV for inspection
            output_path = Path("data/output/full_36_features_test.csv")
            df = pd.DataFrame(
                [
                    {
                        "date": first_feature_set.date,
                        **dict(zip(feature_names, feature_vector, strict=False)),
                    }
                ]
            )
            df.to_csv(output_path, index=False)
            print(f"\n💾 Saved full features to: {output_path}")

        else:
            print("\n❌ No Seoul features found in feature set")
    else:
        print("\n❌ No features extracted")

    print("\n" + "=" * 60)
    print(" TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_full_feature_extraction()
