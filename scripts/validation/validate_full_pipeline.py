#!/usr/bin/env python3
"""
Comprehensive pipeline validation script.

Tests all components:
1. Data parsing (XML and JSON)
2. Sleep window aggregation
3. Activity sequence extraction
4. Feature engineering
5. DLMO calculation
6. Mood predictions
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.entities.activity_record import ActivityType
from big_mood_detector.domain.services.activity_sequence_extractor import (
    ActivitySequenceExtractor,
)
from big_mood_detector.domain.services.dlmo_calculator import DLMOCalculator
from big_mood_detector.domain.services.sleep_window_analyzer import SleepWindowAnalyzer
from big_mood_detector.infrastructure.parsers.json import (
    ActivityJSONParser,
    SleepJSONParser,
)
from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def validate_xml_parsing():
    """Test XML parsing and find data date range."""
    print_section("1. XML PARSING VALIDATION")

    xml_file = Path("data/input/apple_export/export.xml")
    if not xml_file.exists():
        print(f"❌ XML file not found: {xml_file}")
        return None

    print(
        f"✅ Found XML file: {xml_file.name} ({xml_file.stat().st_size / (1024**2):.1f} MB)"
    )

    parser = StreamingXMLParser()
    sleep_dates = set()
    activity_dates = set()

    print("\nSampling first 1000 records to find date range...")

    for i, entity in enumerate(parser.parse_file(str(xml_file))):
        if i >= 1000:
            break

        # Check for sleep records
        if hasattr(entity, "state") and "SLEEP" in str(entity.state):
            if hasattr(entity, "start_date"):
                sleep_dates.add(entity.start_date.date())

        # Check for activity records
        elif hasattr(entity, "step_count"):
            if hasattr(entity, "start_date"):
                activity_dates.add(entity.start_date.date())

    if sleep_dates:
        print(f"\n✅ Sleep data range: {min(sleep_dates)} to {max(sleep_dates)}")
    else:
        print("\n❌ No sleep data found in sample")

    if activity_dates:
        print(f"✅ Activity data range: {min(activity_dates)} to {max(activity_dates)}")
    else:
        print("❌ No activity data found in sample")

    return (
        min(sleep_dates | activity_dates) if (sleep_dates or activity_dates) else None
    )


def validate_json_parsing():
    """Test JSON parsing."""
    print_section("2. JSON PARSING VALIDATION")

    json_dir = Path("data/input/health_auto_export")
    if not json_dir.exists():
        print(f"❌ JSON directory not found: {json_dir}")
        return

    # Check for key files
    sleep_file = json_dir / "Sleep Analysis.json"
    activity_file = json_dir / "Step Count.json"

    if sleep_file.exists():
        sleep_parser = SleepJSONParser()
        sleep_data = sleep_parser.parse_file(str(sleep_file))
        print(f"✅ Parsed {len(sleep_data)} sleep records from JSON")
        if sleep_data:
            dates = [
                r.start_date.date() for r in sleep_data if hasattr(r, "start_date")
            ]
            if dates:
                print(f"   Date range: {min(dates)} to {max(dates)}")
    else:
        print("❌ Sleep Analysis.json not found")

    if activity_file.exists():
        activity_parser = ActivityJSONParser()
        activity_data = activity_parser.parse_file(str(activity_file))
        print(f"✅ Parsed {len(activity_data)} activity records from JSON")
        if activity_data:
            dates = [
                r.start_date.date() for r in activity_data if hasattr(r, "start_date")
            ]
            if dates:
                print(f"   Date range: {min(dates)} to {max(dates)}")
    else:
        print("❌ Step Count.json not found")


def validate_sleep_aggregation():
    """Test sleep window aggregation."""
    print_section("3. SLEEP WINDOW AGGREGATION")

    analyzer = SleepWindowAnalyzer()

    # Create test sleep records
    from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState

    test_records = []
    base_date = datetime(2024, 1, 1, 23, 0)  # 11 PM

    # Create fragmented sleep that should merge
    test_records.append(
        SleepRecord(
            source_name="Test",
            start_date=base_date,
            end_date=base_date + timedelta(hours=3),
            state=SleepState.ASLEEP_CORE,
        )
    )

    # Gap of 2 hours (should merge with 3.75h threshold)
    test_records.append(
        SleepRecord(
            source_name="Test",
            start_date=base_date + timedelta(hours=5),
            end_date=base_date + timedelta(hours=8),
            state=SleepState.ASLEEP_CORE,
        )
    )

    print(f"Created {len(test_records)} test sleep records with 2-hour gap")

    # Aggregate
    windows = analyzer.analyze_sleep_episodes(test_records)

    if len(windows) == 1:
        print("✅ Successfully merged into 1 window (3.75h threshold working)")
        print(f"   Total duration: {windows[0].total_duration_hours:.1f} hours")
    else:
        print(f"❌ Expected 1 merged window, got {len(windows)}")


def validate_activity_extraction():
    """Test activity sequence extraction."""
    print_section("4. ACTIVITY SEQUENCE EXTRACTION")

    extractor = ActivitySequenceExtractor()

    # Create test activity data
    from big_mood_detector.domain.entities.activity_record import ActivityRecord

    test_records = []
    base_date = date(2024, 1, 1)

    # Create activity records for one day
    for hour in range(24):
        timestamp = datetime.combine(base_date, datetime.min.time()) + timedelta(
            hours=hour
        )
        steps = 100 if 8 <= hour <= 20 else 0  # Active during day

        test_records.append(
            ActivityRecord(
                source_name="Test",
                start_date=timestamp,
                end_date=timestamp + timedelta(hours=1),
                activity_type=ActivityType.STEP_COUNT,
                value=float(steps),
                unit="count",
            )
        )

    print(f"Created {len(test_records)} hourly activity records")

    # Extract sequence
    sequence = extractor.extract_daily_sequence(test_records, base_date)

    if sequence and len(sequence.activity_values) == 1440:
        print("✅ Extracted 1440-point activity sequence")
        print(f"   Total steps: {sum(sequence.activity_values)}")
        print(
            f"   Active hours: {sum(1 for v in sequence.activity_values if v > 0) / 60:.1f}"
        )
    else:
        print("❌ Failed to extract proper sequence")


def validate_dlmo_calculation():
    """Test DLMO calculation with real sleep patterns."""
    print_section("5. DLMO CALCULATION")

    calculator = DLMOCalculator()

    # Create normal sleep schedule
    from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState

    sleep_records = []
    base_date = date(2024, 1, 15)

    for day in range(14):
        sleep_start = datetime.combine(
            base_date - timedelta(days=day), datetime.min.time()
        ).replace(hour=23)  # 11 PM

        sleep_records.append(
            SleepRecord(
                source_name="Test",
                start_date=sleep_start,
                end_date=sleep_start + timedelta(hours=8),
                state=SleepState.ASLEEP_CORE,
            )
        )

    print(f"Created {len(sleep_records)} days of normal sleep (11 PM - 7 AM)")

    # Calculate DLMO
    result = calculator.calculate_dlmo(
        sleep_records=sleep_records, target_date=base_date, use_activity=False
    )

    if result:
        print(f"✅ DLMO calculated: {result.dlmo_hour:.1f}h")
        print(f"   CBT minimum: {result.cbt_min_hour:.1f}h")
        print(f"   Phase offset: {(result.cbt_min_hour - result.dlmo_hour) % 24:.1f}h")

        # Check if DLMO is in expected range
        if 20.0 <= result.dlmo_hour <= 22.0:
            print("   ✅ DLMO timing is physiologically plausible (20-22h expected)")
        else:
            print("   ⚠️  DLMO outside expected range")
    else:
        print("❌ Failed to calculate DLMO")


def validate_full_pipeline():
    """Test the complete pipeline."""
    print_section("6. FULL PIPELINE VALIDATION")

    pipeline = MoodPredictionPipeline()

    # Try with JSON data first (smaller, faster)
    json_dir = Path("data/input/health_auto_export")
    output_file = Path("output/validation_features.csv")

    print(f"\nProcessing JSON data from: {json_dir}")

    try:
        # Use recent dates that might have data
        df = pipeline.process_health_export(
            json_dir,
            output_file,
            start_date=date(2025, 5, 1),
            end_date=date(2025, 5, 31),
        )

        if len(df) > 0:
            print(f"✅ Generated {len(df)} days of features")
            print(f"\nFeature columns ({len(df.columns)}):")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i:2d}. {col}")

            # Check for key features
            if "circadian_phase_Z" in df.columns:
                print("\n✅ Circadian phase Z-score present (most important feature)")

            # Check for non-zero values
            non_zero_cols = [col for col in df.columns if df[col].abs().sum() > 0]
            print(
                f"\n✅ {len(non_zero_cols)}/{len(df.columns)} features have non-zero values"
            )

            # Save sample
            print(f"\n✅ Saved features to: {output_file}")
        else:
            print("❌ No features generated - check date range and data availability")

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all validation tests."""
    print("=" * 60)
    print(" BIG MOOD DETECTOR - PIPELINE VALIDATION")
    print("=" * 60)

    # Run validations
    earliest_date = validate_xml_parsing()
    validate_json_parsing()
    validate_sleep_aggregation()
    validate_activity_extraction()
    validate_dlmo_calculation()
    validate_full_pipeline()

    # Summary
    print_section("VALIDATION SUMMARY")
    print("\n✅ All pipeline components tested")
    print("\nRecommendations:")
    if earliest_date:
        print(f"1. For XML processing, use dates after {earliest_date}")
    print("2. JSON data appears to be from 2025-05")
    print("3. Sleep window merging uses 3.75-hour threshold")
    print("4. Activity sequences are 1440 points/day")
    print("5. DLMO calculation expects 20-22h for normal sleepers")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
