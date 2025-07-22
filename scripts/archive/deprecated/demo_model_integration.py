#!/usr/bin/env python3
"""
Demo Model Integration Script

Demonstrates how to use PAT and XGBoost models together for mood prediction.
This shows the complete pipeline from raw activity data to mood predictions.

Usage:
    python scripts/demo_model_integration.py
"""

import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.mood_prediction_pipeline import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.entities.activity_record import (
    ActivityRecord,
    ActivityType,
)
from big_mood_detector.domain.services.pat_sequence_builder import (
    PATSequenceBuilder,
)
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
from big_mood_detector.infrastructure.ml_models.xgboost_models import (
    XGBoostMoodPredictor,
)


def create_demo_activity_data():
    """Create synthetic activity data for demonstration."""
    print("Creating synthetic activity data...")

    records = []
    base_date = datetime(2025, 5, 9, tzinfo=UTC)

    # Create 7 days of activity data with realistic patterns
    for day in range(7):
        day_start = base_date + timedelta(days=day)

        # Create activity throughout the day
        for hour in range(24):
            # Base activity level varies by hour
            if 0 <= hour <= 6:
                base_activity = 5  # Low activity at night
            elif 7 <= hour <= 9:
                base_activity = 50  # Morning activity
            elif 10 <= hour <= 17:
                base_activity = 70  # Daytime activity
            elif 18 <= hour <= 22:
                base_activity = 40  # Evening activity
            else:
                base_activity = 10  # Late night

            # Add some randomness
            for minute in range(0, 60, 5):
                start = day_start + timedelta(hours=hour, minutes=minute)
                end = start + timedelta(minutes=5)

                # Add random variation
                value = base_activity + np.random.normal(0, 10)
                value = max(0, value)  # Ensure non-negative

                records.append(
                    ActivityRecord(
                        source_name="Demo",
                        start_date=start,
                        end_date=end,
                        activity_type=ActivityType.STEP_COUNT,
                        value=value,
                        unit="count",
                    )
                )

    print(f"Created {len(records)} activity records for 7 days")
    return records


def demo_pat_model():
    """Demonstrate PAT model usage."""
    print("\n" + "=" * 70)
    print("DEMO: PAT Model Integration")
    print("=" * 70)

    # Check if model weights exist
    weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
    if not weights_path.exists():
        print(f"❌ PAT model weights not found at: {weights_path}")
        print("   Please run: python scripts/download_model_weights.py --model pat")
        return None

    # Initialize PAT model
    print("\n1. Initializing PAT-Medium model...")
    pat_model = PATModel(model_size="medium")

    # Load pretrained weights
    print("2. Loading pretrained weights...")
    if pat_model.load_pretrained_weights(weights_path):
        print("   ✅ Model loaded successfully!")
    else:
        print("   ❌ Failed to load model")
        return None

    # Get model info
    info = pat_model.get_model_info()
    print("\n3. Model Information:")
    print(f"   - Model size: {info['model_size']}")
    print(f"   - Patch size: {info['patch_size']} minutes")
    print(f"   - Number of patches: {info['num_patches']}")
    print(f"   - Embedding dimension: {info['embed_dim']}")
    print(f"   - Parameters: {info['parameters']:,}")

    # Create activity sequence
    print("\n4. Creating 7-day activity sequence...")
    activity_records = create_demo_activity_data()

    # Build PAT sequence
    builder = PATSequenceBuilder()
    sequence = builder.build_sequence(
        activity_records=activity_records, end_date=date(2025, 5, 15)
    )

    print(f"   - Sequence complete: {sequence.is_complete}")
    print(f"   - Data quality score: {sequence.data_quality_score:.2f}")

    # Extract features
    print("\n5. Extracting PAT features...")
    try:
        pat_features = pat_model.extract_features(sequence)
        print(f"   ✅ Extracted {len(pat_features)}-dimensional feature vector")
        print(
            f"   Feature stats: mean={pat_features.mean():.3f}, std={pat_features.std():.3f}"
        )
        return pat_features
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def demo_xgboost_models():
    """Demonstrate XGBoost model usage."""
    print("\n" + "=" * 70)
    print("DEMO: XGBoost Model Integration")
    print("=" * 70)

    # Check if models exist
    model_dir = Path("model_weights/xgboost/pretrained")
    expected_models = ["depression_model.pkl", "hypomanic_model.pkl", "manic_model.pkl"]

    missing = [m for m in expected_models if not (model_dir / m).exists()]
    if missing:
        print(f"❌ Missing XGBoost models: {missing}")
        print("   Please copy from reference_repos/mood_ml/")
        return None

    # Initialize predictor
    print("\n1. Initializing XGBoost mood predictor...")
    predictor = XGBoostMoodPredictor()

    # Load models
    print("2. Loading pretrained models...")
    results = predictor.load_models(model_dir)

    for model_type, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {model_type} model")

    if not predictor.is_loaded:
        print("   ❌ Failed to load all models")
        return None

    # Get model info
    info = predictor.get_model_info()
    print("\n3. Model Information:")
    print(f"   - Number of features: {info['num_features']}")
    print(f"   - Models loaded: {', '.join(info['models_loaded'])}")

    # Create demo features (normally from feature extraction)
    print("\n4. Creating demo features...")
    demo_features = np.random.randn(36)  # Random features for demo

    # Make prediction
    print("\n5. Making mood predictions...")
    try:
        prediction = predictor.predict(demo_features)

        print("\n   Risk Scores:")
        print(f"   - Depression: {prediction.depression_risk:.1%}")
        print(f"   - Hypomanic:  {prediction.hypomanic_risk:.1%}")
        print(f"   - Manic:      {prediction.manic_risk:.1%}")
        print(f"\n   Highest risk: {prediction.highest_risk_type}")
        print(f"   Confidence: {prediction.confidence:.1%}")

        return prediction
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def demo_full_pipeline():
    """Demonstrate the full pipeline using real models."""
    print("\n" + "=" * 70)
    print("DEMO: Full Pipeline Integration")
    print("=" * 70)

    # Check XML data
    xml_path = Path("apple_export/export.xml")
    if not xml_path.exists():
        print(f"❌ No Apple Health export found at: {xml_path}")
        print("   Using synthetic data instead...")

        # For demo, we'll just show the pipeline structure
        print("\nPipeline components:")
        print("1. XML/JSON Parser → Domain Entities")
        print("2. Activity Sequence Extractor → 7-day sequences")
        print("3. PAT Model → Deep features (optional)")
        print("4. Feature Engineering → 36 statistical features")
        print("5. XGBoost Models → Mood predictions")
        return

    # Run actual pipeline
    print("\n1. Initializing mood prediction pipeline...")
    pipeline = MoodPredictionPipeline()

    print("2. Processing health data...")
    try:
        # Process a small date range
        df = pipeline.process_health_export(
            xml_path,
            output_path=Path("output/demo_features.csv"),
            start_date=date(2025, 5, 1),
            end_date=date(2025, 5, 7),
        )

        print(f"   ✅ Extracted features for {len(df)} days")

        # Load XGBoost models
        print("\n3. Loading mood prediction models...")
        predictor = XGBoostMoodPredictor()
        predictor.load_models(Path("model_weights/xgboost/pretrained"))

        if predictor.is_loaded:
            print("\n4. Making predictions...")
            for idx, row in df.iterrows():
                features = row.values
                prediction = predictor.predict(features)

                print(f"\n   Date: {idx}")
                print(f"   Depression risk: {prediction.depression_risk:.1%}")
                print(f"   Hypomanic risk:  {prediction.hypomanic_risk:.1%}")
                print(f"   Manic risk:      {prediction.manic_risk:.1%}")

    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    """Run all demos."""
    print("BIG MOOD DETECTOR - Model Integration Demo")
    print("=" * 70)

    # Demo PAT model
    demo_pat_model()

    # Demo XGBoost models
    demo_xgboost_models()

    # Demo full pipeline
    demo_full_pipeline()

    print("\n" + "=" * 70)
    print("Demo complete!")

    # Summary
    print("\nNext steps:")
    print("1. Ensure all model weights are in place")
    print("2. Process your health data through the pipeline")
    print("3. Consider ensemble methods combining PAT + XGBoost")
    print("4. Fine-tune models on your specific use case")


if __name__ == "__main__":
    main()
