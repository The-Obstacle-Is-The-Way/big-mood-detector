#!/usr/bin/env python3
"""
Compare XML and JSON pipeline outputs for the same time period.

This validates that both data sources produce consistent features.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.mood_prediction_pipeline import (
    MoodPredictionPipeline,
)


def compare_pipelines():
    """Compare features extracted from XML vs JSON for the same dates."""
    print("=" * 70)
    print("XML vs JSON PIPELINE COMPARISON")
    print("=" * 70)

    # Create pipeline
    pipeline = MoodPredictionPipeline()

    # Define overlapping date range (May 2025 seems to have data in both)
    start_date = date(2025, 5, 1)
    end_date = date(2025, 5, 31)

    print(f"\nComparing data for: {start_date} to {end_date}")
    print("-" * 70)

    # Process XML data
    print("\n1. Processing XML data...")
    xml_output = Path("output/comparison_xml_features.csv")
    try:
        xml_df = pipeline.process_health_export(
            Path("apple_export/export.xml"),
            xml_output,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"   ✓ XML: Extracted features for {len(xml_df)} days")
    except Exception as e:
        print(f"   ✗ XML Error: {e}")
        return

    # Process JSON data
    print("\n2. Processing JSON data...")
    json_output = Path("output/comparison_json_features.csv")
    try:
        json_df = pipeline.process_health_export(
            Path("health_auto_export"),
            json_output,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"   ✓ JSON: Extracted features for {len(json_df)} days")
    except Exception as e:
        print(f"   ✗ JSON Error: {e}")
        return

    # Find overlapping dates
    print("\n3. Finding overlapping dates...")
    xml_dates = set(xml_df.index)
    json_dates = set(json_df.index)

    common_dates = xml_dates & json_dates
    xml_only = xml_dates - json_dates
    json_only = json_dates - xml_dates

    print(f"   Common dates: {len(common_dates)}")
    print(f"   XML only: {len(xml_only)} days")
    print(f"   JSON only: {len(json_only)} days")

    if not common_dates:
        print("\n⚠️  No overlapping dates to compare!")
        return

    # Compare features for common dates
    print(f"\n4. Comparing features for {len(common_dates)} overlapping days...")
    print("-" * 70)

    # Get feature columns (excluding date index)
    feature_cols = [col for col in xml_df.columns if col != "date"]

    comparison_results = []

    for date_val in sorted(common_dates):
        xml_row = xml_df.loc[date_val]
        json_row = json_df.loc[date_val]

        # Compare each feature
        date_comparison = {"date": date_val}

        for feature in feature_cols:
            if feature in xml_row and feature in json_row:
                xml_val = xml_row[feature]
                json_val = json_row[feature]

                # Calculate difference
                if pd.notna(xml_val) and pd.notna(json_val):
                    if abs(xml_val) > 1e-10 or abs(json_val) > 1e-10:
                        # Relative difference for non-zero values
                        rel_diff = abs(xml_val - json_val) / max(
                            abs(xml_val), abs(json_val)
                        )
                    else:
                        # Both are essentially zero
                        rel_diff = 0.0

                    date_comparison[f"{feature}_diff"] = rel_diff
                    date_comparison[f"{feature}_xml"] = xml_val
                    date_comparison[f"{feature}_json"] = json_val

        comparison_results.append(date_comparison)

    # Analyze differences
    if comparison_results:
        # Show detailed comparison for first date
        first_date = sorted(common_dates)[0]
        print(f"\nDetailed comparison for {first_date}:")
        print("-" * 70)

        xml_row = xml_df.loc[first_date]
        json_row = json_df.loc[first_date]

        significant_diffs = []

        for feature in feature_cols[:10]:  # Show first 10 features
            if feature in xml_row and feature in json_row:
                xml_val = xml_row[feature]
                json_val = json_row[feature]

                if pd.notna(xml_val) and pd.notna(json_val):
                    diff = abs(xml_val - json_val)
                    if diff > 0.01:  # Significant difference threshold
                        significant_diffs.append((feature, xml_val, json_val, diff))

                    print(
                        f"{feature:30s}: XML={xml_val:8.4f}, JSON={json_val:8.4f}, Diff={diff:8.4f}"
                    )

        # Summary statistics
        print("\n5. Summary Statistics")
        print("-" * 70)

        # Calculate average relative differences for each feature type
        feature_types = {
            "sleep": [
                f for f in feature_cols if "sleep" in f or "long" in f or "short" in f
            ],
            "circadian": [f for f in feature_cols if "circadian" in f],
        }

        for feat_type, features in feature_types.items():
            diffs = []
            for result in comparison_results:
                for feat in features:
                    diff_key = f"{feat}_diff"
                    if diff_key in result and pd.notna(result[diff_key]):
                        diffs.append(result[diff_key])

            if diffs:
                print(f"\n{feat_type.upper()} features:")
                print(f"  Average relative difference: {np.mean(diffs):.2%}")
                print(f"  Max relative difference: {np.max(diffs):.2%}")
                print(
                    f"  Features < 5% different: {sum(d < 0.05 for d in diffs) / len(diffs):.1%}"
                )

        # Check if results are reasonably similar
        all_diffs = []
        for result in comparison_results:
            for key, val in result.items():
                if key.endswith("_diff") and pd.notna(val):
                    all_diffs.append(val)

        if all_diffs:
            mean_diff = np.mean(all_diffs)
            print("\nOVERALL COMPARISON:")
            print(f"  Average difference across all features: {mean_diff:.2%}")
            print(
                f"  Features matching within 10%: {sum(d < 0.10 for d in all_diffs) / len(all_diffs):.1%}"
            )

            if mean_diff < 0.10:
                print(
                    "\n✅ VALIDATION PASSED: XML and JSON pipelines produce consistent results!"
                )
            else:
                print(
                    "\n⚠️  WARNING: Significant differences found between XML and JSON processing"
                )
                print(
                    "   This may be due to different data completeness or processing logic"
                )

    # Save detailed comparison
    comparison_df = pd.DataFrame(comparison_results)
    comparison_output = Path("output/xml_json_comparison.csv")
    comparison_df.to_csv(comparison_output, index=False)
    print(f"\nDetailed comparison saved to: {comparison_output}")


if __name__ == "__main__":
    compare_pipelines()
