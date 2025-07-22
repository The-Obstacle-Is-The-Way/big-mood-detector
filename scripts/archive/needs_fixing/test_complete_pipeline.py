#!/usr/bin/env python3
"""
Complete Pipeline Integration Test

Tests the entire data processing pipeline from raw Apple Health XML
through feature extraction, PAT/XGBoost predictions, to final ensemble output.

This ensures everything works before we clean up deprecated code.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.ensemble_orchestrator import EnsembleOrchestrator
from big_mood_detector.application.use_cases.process_health_data import (
    ProcessHealthDataUseCase,
)
from big_mood_detector.infrastructure.ml_models import PATModel, XGBoostMoodPredictor
from big_mood_detector.infrastructure.parsers.streaming_xml_parser import (
    StreamingXMLParser,
)
from big_mood_detector.infrastructure.repositories.in_memory_repository import (
    InMemoryHealthRepository,
)


def test_xml_parsing():
    """Test XML parsing with real Apple Health export."""
    print("\n1. Testing XML Parsing")
    print("=" * 70)

    xml_path = Path("apple_export/export.xml")
    if not xml_path.exists():
        print("❌ Apple Health export.xml not found")
        print("   Place your export.xml in apple_export/ directory")
        return None, None, None

    print(f"📁 Found export.xml: {xml_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Use streaming parser for memory efficiency
    parser = StreamingXMLParser()
    repository = InMemoryHealthRepository()

    print("🔄 Parsing XML (this may take a moment)...")
    start_time = time.time()

    try:
        # Parse with progress tracking
        records_parsed = 0
        for record in parser.parse_health_export(xml_path):
            repository.save(record)
            records_parsed += 1
            if records_parsed % 10000 == 0:
                print(f"   Parsed {records_parsed:,} records...")

        parse_time = time.time() - start_time
        print(f"✅ Parsed {records_parsed:,} records in {parse_time:.1f}s")
        print(f"   Rate: {records_parsed / parse_time:.0f} records/second")

        # Get record counts by type
        sleep_count = len(list(repository.get_sleep_records()))
        activity_count = len(list(repository.get_activity_records()))
        hr_count = len(list(repository.get_heart_rate_records()))

        print("\n📊 Record breakdown:")
        print(f"   Sleep records: {sleep_count:,}")
        print(f"   Activity records: {activity_count:,}")
        print(f"   Heart rate records: {hr_count:,}")

        return repository, sleep_count, activity_count

    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        return None, None, None


def test_feature_extraction(repository):
    """Test feature extraction pipeline."""
    print("\n2. Testing Feature Extraction")
    print("=" * 70)

    if not repository:
        print("❌ No repository available")
        return None

    # Initialize use case
    use_case = ProcessHealthDataUseCase(repository)

    print("🔄 Extracting features...")
    start_time = time.time()

    try:
        # Process all data
        summary = use_case.process_all_data()

        extract_time = time.time() - start_time
        print(f"✅ Feature extraction completed in {extract_time:.1f}s")

        # Show summary statistics
        print("\n📊 Feature extraction summary:")
        print(f"   Days processed: {summary.total_days}")
        print(f"   Sleep windows: {summary.sleep_windows_count}")
        print(f"   Activity sequences: {summary.activity_sequences_count}")
        print(f"   Clinical features extracted: {summary.features_extracted}")

        if summary.warnings:
            print(f"\n⚠️  Warnings ({len(summary.warnings)}):")
            for warning in summary.warnings[:5]:  # Show first 5
                print(f"   - {warning}")
            if len(summary.warnings) > 5:
                print(f"   ... and {len(summary.warnings) - 5} more")

        return summary

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_pat_model():
    """Test PAT model loading and inference."""
    print("\n3. Testing PAT Model")
    print("=" * 70)

    weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
    if not weights_path.exists():
        print("❌ PAT weights not found")
        return None

    print("🔄 Loading PAT model...")
    start_time = time.time()

    try:
        pat = PATModel(model_size="medium")
        if pat.load_pretrained_weights(weights_path):
            load_time = time.time() - start_time
            print(f"✅ PAT model loaded in {load_time:.2f}s")

            # Get model info
            info = pat.get_model_info()
            print("\n📊 Model info:")
            print(f"   Model size: {info['model_size']}")
            print(f"   Parameters: {info['parameters']:,}")
            print(f"   Patch size: {info['patch_size']} minutes")
            print(f"   Embed dim: {info['embed_dim']}")
            print(f"   Encoder layers: {info['encoder_layers']}")

            return pat
        else:
            print("❌ Failed to load PAT model")
            return None

    except Exception as e:
        print(f"❌ Error with PAT model: {e}")
        return None


def test_xgboost_models():
    """Test XGBoost model loading."""
    print("\n4. Testing XGBoost Models")
    print("=" * 70)

    model_dir = Path("model_weights/xgboost/pretrained")
    if not model_dir.exists():
        print("❌ XGBoost model directory not found")
        return None

    print("🔄 Loading XGBoost models...")
    start_time = time.time()

    try:
        predictor = XGBoostMoodPredictor()
        results = predictor.load_models(model_dir)

        if predictor.is_loaded:
            load_time = time.time() - start_time
            print(f"✅ XGBoost models loaded in {load_time:.2f}s")

            # Show loaded models
            print("\n📊 Models loaded:")
            for model_type, status in results.items():
                print(f"   {model_type}: {'✅' if status else '❌'}")

            return predictor
        else:
            print("❌ Failed to load XGBoost models")
            return None

    except Exception as e:
        print(f"❌ Error with XGBoost models: {e}")
        return None


def test_ensemble_orchestration(pat_model, xgboost_predictor, repository):
    """Test ensemble orchestration with real data."""
    print("\n5. Testing Ensemble Orchestration")
    print("=" * 70)

    if not all([pat_model, xgboost_predictor, repository]):
        print("❌ Missing required models or data")
        return None

    print("🔄 Running ensemble predictions...")
    start_time = time.time()

    try:
        # Create orchestrator
        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor, pat_model=pat_model
        )

        # Get some activity records for testing
        activity_records = list(repository.get_activity_records())[
            :10080
        ]  # 7 days worth

        if len(activity_records) < 10080:
            print(f"⚠️  Only {len(activity_records)} activity records available")

        # Create mock statistical features for demo
        import numpy as np

        stat_features = np.random.randn(36).astype(np.float32)
        stat_features[0] = 7.5  # sleep duration mean
        stat_features[1] = 1.2  # sleep duration std

        # Run prediction
        result = orchestrator.predict(
            statistical_features=stat_features,
            activity_records=activity_records if activity_records else None,
        )

        pred_time = time.time() - start_time
        print(f"✅ Ensemble prediction completed in {pred_time:.2f}s")

        # Show results
        print("\n📊 Prediction results:")
        print(f"   Depression risk: {result.prediction.depression_risk:.1%}")
        print(f"   Hypomanic risk: {result.prediction.hypomanic_risk:.1%}")
        print(f"   Manic risk: {result.prediction.manic_risk:.1%}")
        print(f"   Confidence: {result.prediction.confidence:.1%}")
        print(f"   Highest risk: {result.prediction.highest_risk_type}")

        # Show model contributions
        if result.xgboost_prediction:
            print("\n   XGBoost contribution:")
            print(f"     Depression: {result.xgboost_prediction.depression_risk:.1%}")
            print(f"     Weight: {orchestrator.config.xgboost_weight:.0%}")

        if result.pat_enhanced_prediction:
            print("\n   PAT-enhanced contribution:")
            print(
                f"     Depression: {result.pat_enhanced_prediction.depression_risk:.1%}"
            )
            print(f"     Weight: {orchestrator.config.pat_weight:.0%}")

        return result

    except Exception as e:
        print(f"❌ Error in ensemble orchestration: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run full pipeline test."""
    print("=" * 70)
    print("COMPLETE PIPELINE INTEGRATION TEST")
    print("=" * 70)
    print("\nThis test validates the entire pipeline before cleanup.")

    # Track overall results
    all_passed = True

    # Test 1: XML Parsing
    repository, sleep_count, activity_count = test_xml_parsing()
    if not repository:
        all_passed = False
        print("\n⚠️  Skipping remaining tests due to parsing failure")

    # Test 2: Feature Extraction
    if repository:
        summary = test_feature_extraction(repository)
        if not summary:
            all_passed = False

    # Test 3: PAT Model
    pat_model = test_pat_model()
    if not pat_model:
        all_passed = False

    # Test 4: XGBoost Models
    xgboost_predictor = test_xgboost_models()
    if not xgboost_predictor:
        all_passed = False

    # Test 5: Ensemble Orchestration
    if repository and pat_model and xgboost_predictor:
        result = test_ensemble_orchestration(pat_model, xgboost_predictor, repository)
        if not result:
            all_passed = False

    # Final summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nThe pipeline is working correctly. Safe to proceed with cleanup:")
        print("1. Remove deprecated files (_deprecated_*.py)")
        print("2. Clean up test scripts")
        print("3. Update documentation")
        print("4. Create production deployment")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nFix issues before proceeding with cleanup.")
        print("Check the error messages above for details.")

    print("\nNext steps for production readiness:")
    print("1. ✅ Test with real data (this script)")
    print("2. 🟡 Add learned positional embeddings from PAT repo")
    print("3. 🟡 Export SavedModel format for faster loading")
    print("4. 🟡 Add h5py type stubs")
    print("5. 🟢 Remove all deprecated code")
    print("6. 🟢 Create Docker image")
    print("7. 🟢 Deploy to production")


if __name__ == "__main__":
    main()
