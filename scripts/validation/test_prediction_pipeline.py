#!/usr/bin/env python3
"""
Test the actual prediction pipeline to ensure all features are used correctly.
This validates the REAL workflow, not just CSV export.
"""

import json
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def validate_prediction_pipeline():
    """Test the prediction pipeline with real data."""

    print_section("PREDICTION PIPELINE VALIDATION")

    # Test data location
    json_dir = Path("data/input/health_auto_export")

    # Test different configurations
    test_configs = [
        {
            "name": "XGBoost Only (Single Model)",
            "config": PipelineConfig(
                include_pat_sequences=False,
                min_days_required=1,
                enable_personal_calibration=False,
            ),
        },
        {
            "name": "Ensemble (XGBoost + PAT)",
            "config": PipelineConfig(
                include_pat_sequences=True,
                min_days_required=7,
                enable_personal_calibration=False,
            ),
        },
        {
            "name": "Ensemble + Personal Calibration",
            "config": PipelineConfig(
                include_pat_sequences=True,
                min_days_required=7,
                enable_personal_calibration=True,
                user_id="test_user",
            ),
        },
    ]

    # Test with June 2025 data
    start_date = date(2025, 6, 1)
    end_date = date(2025, 6, 30)

    results = {}

    for test in test_configs:
        print_section(test["name"])

        try:
            # Create pipeline with config
            pipeline = MoodPredictionPipeline(config=test["config"])

            # Process and predict
            result = pipeline.process_apple_health_file(
                file_path=json_dir, start_date=start_date, end_date=end_date
            )

            print("\n✅ Processing complete!")
            print(f"   Days processed: {result.features_extracted}")
            print(f"   Records processed: {result.records_processed:,}")
            print(f"   Processing time: {result.processing_time_seconds:.1f}s")
            print(f"   Confidence score: {result.confidence_score:.1%}")

            # Check predictions
            if result.daily_predictions:
                print(
                    f"\n📊 Predictions generated: {len(result.daily_predictions)} days"
                )

                # Sample first prediction
                first_date = sorted(result.daily_predictions.keys())[0]
                pred = result.daily_predictions[first_date]

                print(f"\n   Sample prediction for {first_date}:")
                print(f"   - Depression risk: {pred['depression_risk']:.1%}")
                print(f"   - Hypomanic risk: {pred['hypomanic_risk']:.1%}")
                print(f"   - Manic risk: {pred['manic_risk']:.1%}")
                print(f"   - Confidence: {pred['confidence']:.1%}")

                if "models_used" in pred:
                    print(f"   - Models used: {', '.join(pred['models_used'])}")

                # Check feature validation
                if "feature_validation" in result.metadata:
                    validation = result.metadata["feature_validation"]
                    print("\n🔍 Feature Validation:")
                    print(
                        f"   - Features computed: {validation.get('features_computed', 'N/A')}"
                    )
                    print(
                        f"   - Validation passed: {validation.get('all_valid', 'N/A')}"
                    )
                    if "warnings" in validation:
                        print(f"   - Warnings: {len(validation['warnings'])}")

            results[test["name"]] = {
                "success": True,
                "days_processed": result.features_extracted,
                "predictions": len(result.daily_predictions),
                "confidence": result.confidence_score,
            }

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback

            traceback.print_exc()
            results[test["name"]] = {"success": False, "error": str(e)}

    # Summary
    print_section("VALIDATION SUMMARY")

    for config_name, result in results.items():
        if result["success"]:
            print(f"\n✅ {config_name}:")
            print(f"   - Days processed: {result['days_processed']}")
            print(f"   - Predictions: {result['predictions']}")
            print(f"   - Confidence: {result['confidence']:.1%}")
        else:
            print(f"\n❌ {config_name}: FAILED")
            print(f"   - Error: {result['error']}")

    # Save results
    output_path = Path("data/output/prediction_validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print(" VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_prediction_pipeline()
