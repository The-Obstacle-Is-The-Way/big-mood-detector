#!/usr/bin/env python3
"""
Quick script to trace the sleep math bug with real data.
"""

from datetime import date, timedelta
from pathlib import Path

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


def main():
    # Use real export
    export_file = Path(
        "/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector/data/input/apple_export/export.xml"
    )

    if not export_file.exists():
        print("No export file found!")
        return

    print(f"🔍 PROCESSING REAL DATA: {export_file}")

    # Create pipeline with baseline tracking
    config = PipelineConfig()
    config.enable_personal_calibration = True
    config.user_id = "trace_test"
    config.min_days_required = 7

    baseline_repo = FileBaselineRepository(Path("./temp_trace_baselines"))

    pipeline = MoodPredictionPipeline(config=config, baseline_repository=baseline_repo)

    # Process the file
    print("\n⏳ Processing Apple Health export...")
    result = pipeline.process_apple_health_file(
        file_path=export_file,
        end_date=date.today(),
        start_date=date.today() - timedelta(days=30),
    )

    if result and result.daily_predictions:
        print(f"\n✅ Processed {len(result.daily_predictions)} days")

        # Show last 7 days of predictions
        dates = sorted(result.daily_predictions.keys())[-7:]
        print("\n📊 LAST 7 DAYS:")
        for d in dates:
            pred = result.daily_predictions[d]
            print(f"   {d}: confidence={pred.get('confidence', 0):.2f}")

    # Check the baseline
    baseline = baseline_repo.get_baseline("trace_test")
    if baseline:
        print("\n🎯 BASELINE CALCULATED:")
        print(f"   Sleep mean: {baseline.sleep_mean:.1f} hours")
        print(f"   Sleep std: {baseline.sleep_std:.1f} hours")
        print(f"   Activity mean: {baseline.activity_mean:.0f} steps")
        print(f"   Data points: {baseline.data_points}")

        if baseline.sleep_mean < 6.0:
            print(
                f"\n🐛 BUG CONFIRMED: Sleep baseline is too low! Should be ~7-8 hours, not {baseline.sleep_mean:.1f}!"
            )
    else:
        print("\n❌ No baseline created!")

    # Cleanup
    import shutil

    if Path("./temp_trace_baselines").exists():
        shutil.rmtree("./temp_trace_baselines")


if __name__ == "__main__":
    main()
