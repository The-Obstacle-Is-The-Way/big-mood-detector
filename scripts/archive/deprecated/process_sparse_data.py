#!/usr/bin/env python3
"""
Process Health Export with Sparse Data Handling

This script demonstrates the full pipeline with sparse data awareness.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from big_mood_detector.application.mood_prediction_pipeline import (
    MoodPredictionPipeline,
)


def main():
    """Process health data with sparse data handling."""

    # Initialize pipeline
    pipeline = MoodPredictionPipeline()

    # Process Health Auto Export data
    input_path = Path("health_auto_export")
    output_path = Path("output/ray_features_with_sparse_handling.csv")

    print("=== Big Mood Detector - Sparse Data Pipeline ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Process all available data
    df = pipeline.process_health_export(
        export_path=input_path,
        output_path=output_path,
        start_date=None,  # Use all available data
        end_date=None,
    )

    if not df.empty:
        print("\n=== Feature Summary ===")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Total days: {len(df)}")

        # Check which features have non-zero values
        non_zero_features = []
        for col in df.columns:
            if df[col].abs().sum() > 0:
                non_zero_features.append(col)

        print(f"Non-zero features: {len(non_zero_features)}/{len(df.columns)}")

        # Check circadian features specifically
        circadian_cols = [col for col in df.columns if "circadian" in col.lower()]
        if circadian_cols:
            print("\nCircadian features:")
            for col in circadian_cols:
                non_zero = (df[col] != 0).sum()
                print(f"  {col}: {non_zero}/{len(df)} days have data")
    else:
        print("\nNo features extracted. Check your data and date ranges.")


if __name__ == "__main__":
    main()
