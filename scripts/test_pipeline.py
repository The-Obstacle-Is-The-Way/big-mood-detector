#!/usr/bin/env python3
"""
Test the end-to-end pipeline with sample data.

This script demonstrates how to use the mood prediction pipeline
with real Apple Health data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from big_mood_detector.application.mood_prediction_pipeline import MoodPredictionPipeline
from datetime import date


def main():
    """Run pipeline test."""
    # Find sample data
    project_root = Path(__file__).parent.parent
    
    # Use the user's real data
    sample_data = project_root / 'health_auto_export'
    
    if not sample_data.exists():
        print(f"Data not found at: {sample_data}")
        return
    
    # Output path
    output_path = project_root / 'output' / 'test_features.csv'
    output_path.parent.mkdir(exist_ok=True)
    
    # Create pipeline
    print("Creating mood prediction pipeline...")
    pipeline = MoodPredictionPipeline()
    
    # Process data
    print(f"Processing data from: {sample_data}")
    print(f"Output will be saved to: {output_path}")
    
    try:
        df = pipeline.process_health_export(
            sample_data,
            output_path,
            start_date=date(2025, 5, 10),  # Use dates with overlap
            end_date=date(2025, 5, 20)
        )
        
        print(f"\nSuccessfully processed {len(df)} days!")
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nFeature statistics:")
        print(df.describe())
        
        # Check for circadian phase z-scores (most important feature)
        if 'circadian_phase_Z' in df.columns:
            print(f"\nCircadian phase Z-scores range: [{df['circadian_phase_Z'].min():.2f}, {df['circadian_phase_Z'].max():.2f}]")
            print("This is the most important feature for mood prediction!")
        
        # Also check if circadian amplitude is non-zero
        if 'circadian_amplitude_MN' in df.columns:
            print(f"\nCircadian amplitude mean: {df['circadian_amplitude_MN'].mean():.4f}")
            print("(Should be non-zero if IS/IV/RA calculations are working)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()