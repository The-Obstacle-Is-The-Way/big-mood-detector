#!/usr/bin/env python3
"""
Test the full pipeline including mood predictions.

This demonstrates the complete flow from raw Apple Health data
to mood episode risk predictions.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.mood_prediction_pipeline import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.services.activity_aggregator import ActivityAggregator
from big_mood_detector.domain.services.advanced_feature_engineering import (
    AdvancedFeatureEngineer,
)
from big_mood_detector.domain.services.heart_rate_aggregator import HeartRateAggregator
from big_mood_detector.domain.services.mood_predictor import MoodPredictor
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator


def main():
    """Run full pipeline test with mood predictions."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "health_auto_export"
    output_path = project_root / "output" / "full_pipeline_results.csv"
    output_path.parent.mkdir(exist_ok=True)

    print("=" * 60)
    print("FULL MOOD PREDICTION PIPELINE TEST")
    print("=" * 60)

    # 1. Create pipeline
    print("\n1. Creating mood prediction pipeline...")
    pipeline = MoodPredictionPipeline()

    # 2. Process health data
    print(f"\n2. Processing data from: {data_path}")
    try:
        df = pipeline.process_health_export(
            data_path,
            output_path,
            start_date=date(2025, 5, 10),
            end_date=date(2025, 5, 20),
        )
        print(f"   ✓ Processed {len(df)} days of features")
    except Exception as e:
        print(f"   ✗ Error processing data: {e}")
        return

    # 3. Load mood predictor
    print("\n3. Loading XGBoost mood prediction models...")
    predictor = MoodPredictor()
    if predictor.is_loaded:
        model_info = predictor.get_model_info()
        for mood_type, info in model_info.items():
            print(f"   ✓ {mood_type}: {info['type']}")
    else:
        print("   ✗ Failed to load models")
        return

    # 4. Make predictions for each day
    print("\n4. Making mood predictions...")
    predictions = []

    for idx, row in df.iterrows():
        # Extract 36 features (excluding date column)
        features = row.drop("date" if "date" in row else []).values

        try:
            # Make prediction
            pred = predictor.predict(features)

            # Add to results
            result = {
                "date": idx,
                "depression_risk": pred.depression_risk,
                "hypomanic_risk": pred.hypomanic_risk,
                "manic_risk": pred.manic_risk,
                "highest_risk": pred.highest_risk_type,
                "risk_value": pred.highest_risk_value,
                "confidence": pred.confidence,
            }
            predictions.append(result)

        except Exception as e:
            print(f"   ✗ Error predicting for {idx}: {e}")

    # 5. Display results
    print(f"\n5. Results Summary:")
    print("-" * 60)

    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.set_index("date", inplace=True)

        print("\nMood Risk Predictions by Day:")
        print(pred_df.round(3))

        # Calculate average risks
        print("\nAverage Risk Levels:")
        print(f"  Depression: {pred_df['depression_risk'].mean():.1%}")
        print(f"  Hypomanic:  {pred_df['hypomanic_risk'].mean():.1%}")
        print(f"  Manic:      {pred_df['manic_risk'].mean():.1%}")

        # Check for high-risk days
        high_risk_days = pred_df[pred_df["risk_value"] > 0.5]
        if not high_risk_days.empty:
            print("\n⚠️  HIGH RISK DAYS (>50%):")
            for idx, row in high_risk_days.iterrows():
                print(f"  {idx}: {row['highest_risk']} ({row['risk_value']:.1%})")
        else:
            print("\n✓ No high-risk days detected")

        # Save combined results
        combined_output = output_path.parent / "mood_predictions.csv"
        pred_df.to_csv(combined_output)
        print(f"\n6. Saved predictions to: {combined_output}")

    else:
        print("No predictions generated")

    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
