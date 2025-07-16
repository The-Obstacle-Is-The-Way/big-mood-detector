#!/usr/bin/env python3
"""Process JSON data from Health Auto Export and extract features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv
import json

from big_mood_detector.domain.services.feature_extraction_service import (
    FeatureExtractionService,
)
from big_mood_detector.infrastructure.parsers.json import (
    ActivityJSONParser,
    HeartRateJSONParser,
    SleepJSONParser,
)


def main():
    # Set up paths
    data_dir = Path("health_auto_export")

    # Initialize parsers
    sleep_parser = SleepJSONParser()
    heart_rate_parser = HeartRateJSONParser()
    activity_parser = ActivityJSONParser()

    # Parse data
    print("Parsing sleep data...")
    sleep_file = data_dir / "Sleep Analysis.json"
    with open(sleep_file) as f:
        sleep_data = json.load(f)
    sleep_records = sleep_parser.parse(sleep_data)
    print(f"  Found {len(sleep_records)} sleep records")

    print("\nParsing heart rate data...")
    heart_rate_file = data_dir / "Heart Rate.json"
    with open(heart_rate_file) as f:
        heart_rate_data = json.load(f)
    heart_rate_records = heart_rate_parser.parse(heart_rate_data)
    print(f"  Found {len(heart_rate_records)} heart rate records")

    print("\nParsing activity data...")
    step_count_file = data_dir / "Step Count.json"
    with open(step_count_file) as f:
        step_data = json.load(f)
    activity_records = activity_parser.parse(step_data)
    print(f"  Found {len(activity_records)} activity records")

    # Extract clinical features
    print("\nExtracting clinical features...")
    feature_service = FeatureExtractionService()
    clinical_features_by_date = feature_service.extract_features(
        sleep_records, activity_records, heart_rate_records
    )

    print(f"\nClinical features extracted for {len(clinical_features_by_date)} days")

    # Convert to list for easier processing
    clinical_data = []
    for date_obj, features in sorted(clinical_features_by_date.items()):
        row = {
            "date": date_obj,
            "sleep_duration_hours": features.sleep_duration_hours,
            "sleep_efficiency": features.sleep_efficiency,
            "sleep_fragmentation": features.sleep_fragmentation,
            "sleep_onset_hour": features.sleep_onset_hour,
            "wake_time_hour": features.wake_time_hour,
            "total_steps": features.total_steps,
            "activity_variance": features.activity_variance,
            "sedentary_hours": features.sedentary_hours,
            "peak_activity_hour": features.peak_activity_hour,
            "avg_resting_hr": features.avg_resting_hr,
            "hrv_sdnn": features.hrv_sdnn,
            "hr_circadian_range": features.hr_circadian_range,
            "circadian_alignment_score": features.circadian_alignment_score,
            "is_clinically_significant": features.is_clinically_significant,
        }
        clinical_data.append(row)

    print("\nSample clinical features (first 5 days):")
    for i, row in enumerate(clinical_data[:5]):
        print(f"  Day {i+1}: {row['date']}")
        print(
            f"    Sleep: {row['sleep_duration_hours']:.1f}h, efficiency: {row['sleep_efficiency']:.1%}"
        )
        print(
            f"    Activity: {row['total_steps']} steps, sedentary: {row['sedentary_hours']:.1f}h"
        )
        print(f"    Heart: {row['avg_resting_hr']:.0f} bpm, HRV: {row['hrv_sdnn']:.1f}")

    # Extract advanced features would require daily summaries
    print("\nNote: Advanced features extraction requires aggregated daily summaries.")
    print("Currently we have extracted clinical features which include:")
    print("  - Sleep metrics (duration, efficiency, timing)")
    print("  - Activity metrics (steps, sedentary time)")
    print("  - Heart rate metrics (resting HR, HRV)")

    # Placeholder for advanced features
    advanced_features = {
        "note": "Advanced features require daily aggregation",
        "features_needed": [
            "sleep_regularity_index",
            "interdaily_stability",
            "intradaily_variability",
            "relative_amplitude",
            "circadian_phase_markers",
        ],
    }

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save clinical features
    if clinical_data:
        with open(output_dir / "clinical_features.csv", "w", newline="") as f:
            fieldnames = list(clinical_data[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clinical_data)
        print(f"\nClinical features saved to: {output_dir / 'clinical_features.csv'}")

    # Save advanced features
    with open(output_dir / "advanced_features.json", "w") as f:
        json.dump(advanced_features, f, indent=2, default=str)
    print(f"Advanced features saved to: {output_dir / 'advanced_features.json'}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    if clinical_data:
        dates = [row["date"] for row in clinical_data]
        print(f"\nDate range: {min(dates)} to {max(dates)}")
        print(f"Total days: {len(dates)}")

        print("\nClinical features summary:")

        # Calculate statistics for numeric features
        numeric_features = [
            "sleep_duration_hours",
            "sleep_efficiency",
            "total_steps",
            "avg_resting_hr",
            "hrv_sdnn",
            "sedentary_hours",
        ]

        for feature in numeric_features:
            values = [row[feature] for row in clinical_data if row[feature] is not None]
            if values:
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)

                # Calculate std manually
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                std_val = variance**0.5

                print(f"\n{feature}:")
                print(f"  Mean: {mean_val:.2f}")
                print(f"  Std:  {std_val:.2f}")
                print(f"  Min:  {min_val:.2f}")
                print(f"  Max:  {max_val:.2f}")


if __name__ == "__main__":
    main()
