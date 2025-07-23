#!/usr/bin/env python3
"""Validate golden run outputs for feature completeness and correctness."""

import json
import sys
from pathlib import Path

import pandas as pd

# Expected 36 Seoul features
EXPECTED_FEATURES = [
    # Basic Sleep Features (1-5)
    'sleep_duration_hours',
    'sleep_efficiency',
    'sleep_onset_hour',
    'wake_time_hour',
    'sleep_fragmentation',
    # Advanced Sleep Features (6-10)
    'sleep_regularity_index',
    'short_sleep_window_pct',
    'long_sleep_window_pct',
    'sleep_onset_variance',
    'wake_time_variance',
    # Circadian Rhythm Features (11-18)
    'interdaily_stability',
    'intradaily_variability',
    'relative_amplitude',
    'l5_value',
    'm10_value',
    'l5_onset_hour',
    'm10_onset_hour',
    'dlmo_hour',
    # Activity Features (19-24)
    'total_steps',
    'activity_variance',
    'sedentary_hours',
    'activity_fragmentation',
    'sedentary_bout_mean',
    'activity_intensity_ratio',
    # Heart Rate Features (25-28)
    'avg_resting_hr',
    'hrv_sdnn',
    'hr_circadian_range',
    'hr_minimum_hour',
    # Phase Features (29-32)
    'circadian_phase_advance',
    'circadian_phase_delay',
    'dlmo_confidence',
    'pat_hour',
    # Z-Score Features (33-36)
    'sleep_duration_zscore',
    'activity_zscore',
    'hr_zscore',
    'hrv_zscore'
]

def validate_features(features_path: Path):
    """Validate feature CSV has all expected columns and reasonable values."""
    print("\n📊 Validating Features...")

    df = pd.read_csv(features_path)
    print(f"   Loaded {len(df)} days of features")

    # Check for missing features
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        print(f"   ❌ Missing {len(missing)} features: {sorted(missing)[:5]}...")
    else:
        print("   ✅ All 36 Seoul features present!")

    # Check for NaN values
    nan_counts = df[list(set(EXPECTED_FEATURES) & set(df.columns))].isna().sum()
    if nan_counts.sum() > 0:
        print(f"   ⚠️  Found NaN values in {nan_counts[nan_counts > 0].count()} features")
    else:
        print("   ✅ No NaN values in features")

    # Check value ranges for key features
    if 'sleep_duration_hours' in df.columns:
        sleep_range = df['sleep_duration_hours'].describe()
        print(f"   Sleep duration: {sleep_range['min']:.1f} - {sleep_range['max']:.1f} hours")

    if 'total_steps' in df.columns:
        steps_range = df['total_steps'].describe()
        print(f"   Daily steps: {steps_range['min']:.0f} - {steps_range['max']:.0f}")

    return len(missing) == 0

def validate_predictions(report_path: Path):
    """Validate prediction report exists and contains expected sections."""
    print("\n🎯 Validating Predictions...")

    if not report_path.exists():
        print(f"   ❌ Report not found: {report_path}")
        return False

    content = report_path.read_text()

    # Check for expected sections
    expected_sections = [
        "RISK SUMMARY",
        "Depression Risk",
        "Mania Risk",
        "Hypomania Risk",
        "KEY FINDINGS"
    ]

    found = sum(1 for section in expected_sections if section in content)
    print(f"   ✅ Found {found}/{len(expected_sections)} expected report sections")

    # Extract risk scores
    import re
    depression_match = re.search(r'Depression Risk: \w+ \(([\d.]+)\)', content)
    if depression_match:
        risk = float(depression_match.group(1))
        print(f"   Depression risk: {risk:.2f}")

    return found == len(expected_sections)

def main():
    if len(sys.argv) != 2:
        print("Usage: validate_golden_output.py <output_directory>")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    features_path = out_dir / "features.csv"
    report_path = out_dir / "report.txt"

    # Validate both outputs
    features_ok = validate_features(features_path)
    predictions_ok = validate_predictions(report_path)

    # Summary
    print("\n📋 Validation Summary:")
    print(f"   Features: {'✅ PASS' if features_ok else '❌ FAIL'}")
    print(f"   Predictions: {'✅ PASS' if predictions_ok else '❌ FAIL'}")

    if not (features_ok and predictions_ok):
        sys.exit(1)

    # Save validation results
    results = {
        "features_ok": features_ok,
        "predictions_ok": predictions_ok,
        "feature_count": len(pd.read_csv(features_path).columns),
        "days_processed": len(pd.read_csv(features_path))
    }

    with open(out_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Validation results saved to {out_dir}/validation_results.json")

if __name__ == "__main__":
    main()
