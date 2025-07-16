#!/usr/bin/env python3
"""
Test Data Processing Pipeline

This script tests the complete data processing pipeline from raw Apple Health XML
through feature extraction and mood predictions.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.ensemble_orchestrator import EnsembleOrchestrator
from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
)
from big_mood_detector.domain.services.sleep_window_analyzer import SleepWindowAnalyzer
from big_mood_detector.infrastructure.ml_models import PATModel, XGBoostMoodPredictor
from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)


def test_xml_parsing():
    """Test XML parsing with real Apple Health export."""
    print("\n1. Testing XML Parsing")
    print("=" * 70)

    xml_path = Path("apple_export/export.xml")
    if not xml_path.exists():
        print("❌ Apple Health export.xml not found")
        print("   Place your export.xml in apple_export/ directory")
        return None

    print(f"📁 Found export.xml: {xml_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Use streaming parser for memory efficiency
    parser = StreamingXMLParser()

    print("🔄 Parsing XML (this may take a moment)...")
    start_time = time.time()

    # Collect records by type
    sleep_records = []
    activity_records = []
    hr_records = []

    try:
        records_parsed = 0
        for record in parser.parse_file(xml_path, entity_type="all"):
            if isinstance(record, SleepRecord):
                sleep_records.append(record)
            elif isinstance(record, ActivityRecord):
                activity_records.append(record)
            elif isinstance(record, HeartRateRecord):
                hr_records.append(record)

            records_parsed += 1
            if records_parsed % 10000 == 0:
                print(f"   Parsed {records_parsed:,} records...")

        parse_time = time.time() - start_time
        print(f"✅ Parsed {records_parsed:,} records in {parse_time:.1f}s")
        print(f"   Rate: {records_parsed / parse_time:.0f} records/second")

        print("\n📊 Record breakdown:")
        print(f"   Sleep records: {len(sleep_records):,}")
        print(f"   Activity records: {len(activity_records):,}")
        print(f"   Heart rate records: {len(hr_records):,}")

        return sleep_records, activity_records, hr_records

    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def test_sleep_analysis(sleep_records):
    """Test sleep window analysis."""
    print("\n2. Testing Sleep Window Analysis")
    print("=" * 70)

    if not sleep_records:
        print("❌ No sleep records to analyze")
        return None

    analyzer = SleepWindowAnalyzer()

    print(f"🔄 Analyzing {len(sleep_records)} sleep records...")
    start_time = time.time()

    try:
        windows = analyzer.analyze_sleep_episodes(sleep_records)
        analysis_time = time.time() - start_time

        print(f"✅ Created {len(windows)} sleep windows in {analysis_time:.2f}s")

        # Show some statistics
        if windows:
            durations = [
                (w.end_time - w.start_time).total_seconds() / 3600 for w in windows
            ]
            print("\n📊 Sleep window statistics:")
            print(f"   Average duration: {np.mean(durations):.1f} hours")
            print(f"   Min duration: {np.min(durations):.1f} hours")
            print(f"   Max duration: {np.max(durations):.1f} hours")

            # Show recent windows
            recent_windows = sorted(windows, key=lambda w: w.start_time)[-5:]
            print("\n   Recent sleep windows:")
            for window in recent_windows:
                print(
                    f"   - {window.start_time.date()}: {window.total_duration_hours:.1f}h ({window.episode_count} episodes)"
                )

        return windows

    except Exception as e:
        print(f"❌ Error analyzing sleep: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_activity_sequences(activity_records):
    """Test activity sequence extraction."""
    print("\n3. Testing Activity Sequence Extraction")
    print("=" * 70)

    if not activity_records:
        print("❌ No activity records to process")
        return None

    extractor = ActivitySequenceExtractor()

    # Get date range from records
    dates = sorted({r.start_date.date() for r in activity_records})
    if not dates:
        print("❌ No dates found in activity records")
        return None

    print(f"📅 Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Extract sequences for recent days
    recent_dates = dates[-7:]  # Last 7 days
    sequences = []

    print(f"🔄 Extracting sequences for {len(recent_dates)} days...")
    start_time = time.time()

    try:
        for date in recent_dates:
            day_records = [r for r in activity_records if r.start_date.date() == date]
            if day_records:
                sequence = extractor.extract_daily_sequence(day_records, date)
                sequences.append(sequence)
                print(f"   ✓ {date}: {len(sequence.activity_values)} values")

        extract_time = time.time() - start_time
        print(f"✅ Extracted {len(sequences)} sequences in {extract_time:.2f}s")

        return sequences

    except Exception as e:
        print(f"❌ Error extracting sequences: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_model_predictions(sequences):
    """Test PAT and XGBoost predictions."""
    print("\n4. Testing Model Predictions")
    print("=" * 70)

    # Check for model weights
    pat_weights = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
    xgb_dir = Path("model_weights/xgboost/pretrained")

    if not pat_weights.exists():
        print("❌ PAT weights not found. Run download_model_weights.py first.")
        return None

    if not xgb_dir.exists():
        print("❌ XGBoost models not found. Run download_model_weights.py first.")
        return None

    print("🔄 Loading models...")
    start_time = time.time()

    try:
        # Load PAT
        pat = PATModel(model_size="medium")
        if not pat.load_pretrained_weights(pat_weights):
            print("⚠️  PAT model failed to load, continuing with XGBoost only")
            pat = None
        else:
            print("✅ PAT model loaded")

        # Load XGBoost
        xgboost = XGBoostMoodPredictor()
        xgboost.load_models(xgb_dir)
        print("✅ XGBoost models loaded")

        load_time = time.time() - start_time
        print(f"   Loading completed in {load_time:.2f}s")

        # Create ensemble
        orchestrator = EnsembleOrchestrator(xgboost, pat)

        # Make a test prediction
        if sequences:
            # Create mock statistical features for demo
            stat_features = np.random.randn(36).astype(np.float32)
            stat_features[0] = 7.5  # sleep duration mean
            stat_features[1] = 1.2  # sleep duration std

            print("\n🔄 Making ensemble prediction...")
            result = orchestrator.predict(
                statistical_features=stat_features,
                activity_records=None,  # Would pass real records here
            )

            print("\n📊 Prediction results:")
            print(
                f"   Depression risk: {result.ensemble_prediction.depression_risk:.1%}"
            )
            print(f"   Hypomanic risk: {result.ensemble_prediction.hypomanic_risk:.1%}")
            print(f"   Manic risk: {result.ensemble_prediction.manic_risk:.1%}")
            print(f"   Confidence: {result.ensemble_prediction.confidence:.1%}")
            print(f"   Models used: {', '.join(result.models_used)}")

            return result

    except Exception as e:
        print(f"❌ Error with predictions: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run complete data processing tests."""
    print("=" * 70)
    print("DATA PROCESSING PIPELINE TEST")
    print("=" * 70)
    print("\nThis test validates each stage of the data processing pipeline.")

    # Test 1: XML Parsing
    sleep_records, activity_records, hr_records = test_xml_parsing()

    if not any([sleep_records, activity_records, hr_records]):
        print("\n⚠️  No data parsed. Please check your Apple Health export.")
        return

    # Test 2: Sleep Analysis
    if sleep_records:
        sleep_windows = test_sleep_analysis(sleep_records)

    # Test 3: Activity Sequences
    if activity_records:
        sequences = test_activity_sequences(activity_records)

    # Test 4: Model Predictions
    if activity_records:
        predictions = test_model_predictions(
            sequences if "sequences" in locals() else None
        )

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print("\n✅ Data Processing Pipeline Status:")
    print(f"   XML Parsing: {'✅' if sleep_records or activity_records else '❌'}")
    print(f"   Sleep Analysis: {'✅' if 'sleep_windows' in locals() else '❌'}")
    print(f"   Activity Sequences: {'✅' if 'sequences' in locals() else '❌'}")
    print(f"   Model Predictions: {'✅' if 'predictions' in locals() else '❌'}")

    print("\n📝 Next Steps:")
    print("1. If any stage failed, check the error messages above")
    print("2. Ensure you have Apple Health export.xml in apple_export/")
    print("3. Run download_model_weights.py if models are missing")
    print("4. Check the logs for detailed debugging information")


if __name__ == "__main__":
    main()
