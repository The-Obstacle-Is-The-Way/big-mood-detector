#!/usr/bin/env python3
"""
Test the complete ML pipeline: XML → Features → XGBoost → Predictions

This validates the entire workflow with real data.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.mood_prediction_pipeline import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.services.mood_predictor import MoodPredictor


def test_full_pipeline():
    """Test complete ML pipeline with XML data."""
    print("=" * 70)
    print("FULL ML PIPELINE TEST")
    print("=" * 70)

    # 1. Create pipeline and predictor
    pipeline = MoodPredictionPipeline()
    predictor = MoodPredictor()

    # Check models loaded
    if not predictor.is_loaded:
        print("❌ Failed to load XGBoost models")
        return

    model_info = predictor.get_model_info()
    print("\n1. Loaded Models:")
    for mood_type, info in model_info.items():
        print(f"   ✓ {mood_type}: {info['type']}")

    # 2. Process XML data for a recent period with data
    print("\n2. Processing XML data...")
    xml_path = Path("apple_export/export.xml")
    output_path = Path("output/ml_pipeline_features.csv")

    # Use May 2025 since we know it has data
    start_date = date(2025, 5, 1)
    end_date = date(2025, 5, 31)

    try:
        df = pipeline.process_health_export(
            xml_path, output_path, start_date=start_date, end_date=end_date
        )
        print(f"   ✓ Extracted features for {len(df)} days")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # 3. Make predictions for each day
    print("\n3. Making mood predictions...")
    predictions = []

    for idx, row in df.iterrows():
        # Get features (exclude any non-feature columns)
        feature_values = row.values

        try:
            # Make prediction
            result = predictor.predict(feature_values)

            predictions.append(
                {
                    "date": idx,
                    "depression_risk": result.depression_risk,
                    "hypomanic_risk": result.hypomanic_risk,
                    "manic_risk": result.manic_risk,
                    "highest_risk": result.highest_risk_type,
                    "confidence": result.confidence,
                }
            )

        except Exception as e:
            print(f"   ⚠️  Error predicting for {idx}: {e}")

    # 4. Analyze results
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.set_index("date", inplace=True)

        print("\n4. RESULTS SUMMARY")
        print("-" * 70)
        print(pred_df.round(3))

        # Statistics
        print("\n5. RISK STATISTICS")
        print("-" * 70)
        print(f"Average depression risk: {pred_df['depression_risk'].mean():.1%}")
        print(f"Average hypomanic risk:  {pred_df['hypomanic_risk'].mean():.1%}")
        print(f"Average manic risk:      {pred_df['manic_risk'].mean():.1%}")

        # High risk days
        high_risk = pred_df[
            pred_df[["depression_risk", "hypomanic_risk", "manic_risk"]].max(axis=1)
            > 0.5
        ]
        if not high_risk.empty:
            print("\n⚠️  HIGH RISK DAYS (>50%):")
            for idx, row in high_risk.iterrows():
                risk_type = row["highest_risk"]
                risk_value = row[f"{risk_type}_risk"]
                print(f"   {idx}: {risk_type} ({risk_value:.1%})")
        else:
            print("\n✅ No high-risk days detected")

        # Save predictions
        pred_output = Path("output/ml_pipeline_predictions.csv")
        pred_df.to_csv(pred_output)
        print(f"\n6. Predictions saved to: {pred_output}")

        # Feature importance analysis
        print("\n7. FEATURE ANALYSIS")
        print("-" * 70)

        # Check which features have high variance (might be important)
        feature_std = df.std()
        top_varying = feature_std.nlargest(10)
        print("\nMost varying features:")
        for feat, std in top_varying.items():
            print(f"   {feat}: std={std:.3f}")

        # Check for features with extreme values on high-risk days
        if not high_risk.empty:
            print("\nFeatures on high-risk days:")
            high_risk_dates = high_risk.index
            for feat in ["circadian_phase_Z", "sleep_percentage_Z", "long_num_Z"]:
                if feat in df.columns:
                    values = df.loc[high_risk_dates, feat]
                    print(f"   {feat}: {values.mean():.3f} (avg)")

    print("\n" + "=" * 70)
    print("✅ FULL ML PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_full_pipeline()
