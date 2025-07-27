#!/usr/bin/env python3
"""
Test independent pipelines with real health data.
"""

import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.application.use_cases.process_with_independent_pipelines import (
    ProcessWithIndependentPipelinesUseCase,
)


def main():
    """Run independent pipelines on health data."""
    # Get data path from command line or use default
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        data_path = Path("data/health_auto_export/export.xml")
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print(f"Processing health data from: {data_path}")
    print("=" * 80)
    
    # Create and execute use case
    use_case = ProcessWithIndependentPipelinesUseCase()
    result = use_case.execute(
        file_path=data_path,
        target_date=date.today(),
    )
    
    # Display results
    print("\n📊 DATA SUMMARY")
    print(f"  Sleep days: {result.data_summary['sleep_days']}")
    print(f"  Activity days: {result.data_summary['activity_days']}")
    print(f"  Heart rate days: {result.data_summary['heart_days']}")
    print(f"  Total records: {result.data_summary['total_records']}")
    
    print("\n🧠 PAT (CURRENT STATE ASSESSMENT)")
    print(f"  Available: {result.pat_available}")
    print(f"  Message: {result.pat_message}")
    if result.pat_result:
        print(f"  Depression risk: {result.pat_result.depression_risk_score:.2f}")
        print(f"  Confidence: {result.pat_result.confidence:.2f}")
        print(f"  Model: {result.pat_result.model_version}")
        print(f"  Window: {result.pat_result.window_start_date} to {result.pat_result.window_end_date}")
    
    print("\n🔮 XGBOOST (FUTURE RISK PREDICTION)")
    print(f"  Available: {result.xgboost_available}")
    print(f"  Message: {result.xgboost_message}")
    if result.xgboost_result:
        print(f"  Depression risk: {result.xgboost_result.depression_probability:.2f}")
        print(f"  Mania risk: {result.xgboost_result.mania_probability:.2f}")
        print(f"  Hypomania risk: {result.xgboost_result.hypomania_probability:.2f}")
        print(f"  Highest risk: {result.xgboost_result.highest_risk_episode}")
        print(f"  Confidence: {result.xgboost_result.confidence_level}")
        print(f"  Days used: {result.xgboost_result.data_days_used}")
    
    print("\n🎯 TEMPORAL ENSEMBLE ASSESSMENT")
    ensemble = result.temporal_ensemble
    print(f"  Assessment date: {ensemble['assessment_date']}")
    print(f"\n  Clinical Summary:")
    print(f"  {ensemble['clinical_summary']}")
    
    if ensemble['recommendations']:
        print(f"\n  Recommendations:")
        for i, rec in enumerate(ensemble['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Show temporal windows if available
    if ensemble['temporal_windows']:
        print(f"\n  Temporal Windows:")
        for window_name, window_data in ensemble['temporal_windows'].items():
            print(f"\n  {window_name.upper()}:")
            for key, value in window_data.items():
                print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()