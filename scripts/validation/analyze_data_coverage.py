#!/usr/bin/env python3
"""
Analyze Data Coverage and Quality

This script provides a detailed analysis of data coverage and quality.
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.infrastructure.parsers.json.json_parsers import (
    ActivityJSONParser,
    SleepJSONParser,
)
from big_mood_detector.infrastructure.sparse_data_handler import SparseDataHandler


def analyze_coverage():
    """Analyze data coverage in detail."""

    # Load data
    sleep_parser = SleepJSONParser()
    activity_parser = ActivityJSONParser()

    sleep_records = sleep_parser.parse_file("health_auto_export/Sleep Analysis.json")
    activity_records = activity_parser.parse_file("health_auto_export/Step Count.json")

    print("=== Data Coverage Analysis ===\n")

    # Convert to daily summary
    sleep_by_date = {}
    for record in sleep_records:
        date_key = record.start_date.date()
        if date_key not in sleep_by_date:
            sleep_by_date[date_key] = []
        sleep_by_date[date_key].append(record)

    activity_by_date = {}
    for record in activity_records:
        date_key = record.start_date.date()
        if date_key not in activity_by_date:
            activity_by_date[date_key] = []
        activity_by_date[date_key].append(record)

    # Create daily coverage DataFrame
    all_dates = sorted(set(sleep_by_date.keys()) | set(activity_by_date.keys()))

    coverage_data = []
    for date_key in all_dates:
        coverage_data.append(
            {
                "date": date_key,
                "has_sleep": date_key in sleep_by_date,
                "sleep_hours": sum(
                    r.duration_hours for r in sleep_by_date.get(date_key, [])
                ),
                "has_activity": date_key in activity_by_date,
                "step_count": sum(r.value for r in activity_by_date.get(date_key, [])),
                "has_both": date_key in sleep_by_date and date_key in activity_by_date,
            }
        )

    df = pd.DataFrame(coverage_data)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    # Monthly summary
    print("Monthly Coverage Summary:")
    print("-" * 70)
    monthly = (
        df.groupby("month")
        .agg(
            {
                "has_sleep": "sum",
                "has_activity": "sum",
                "has_both": "sum",
                "date": "count",
            }
        )
        .rename(columns={"date": "total_days"})
    )

    monthly["sleep_coverage"] = monthly["has_sleep"] / monthly["total_days"] * 100
    monthly["activity_coverage"] = monthly["has_activity"] / monthly["total_days"] * 100
    monthly["both_coverage"] = monthly["has_both"] / monthly["total_days"] * 100

    for month, row in monthly.iterrows():
        print(
            f"{month}: Sleep={row['has_sleep']:2.0f}/{row['total_days']:2.0f} ({row['sleep_coverage']:5.1f}%), "
            f"Activity={row['has_activity']:2.0f}/{row['total_days']:2.0f} ({row['activity_coverage']:5.1f}%), "
            f"Both={row['has_both']:2.0f}/{row['total_days']:2.0f} ({row['both_coverage']:5.1f}%)"
        )

    # Find good windows for analysis
    print("\n\nBest Windows for Analysis (7+ consecutive days with both sensors):")
    print("-" * 70)

    consecutive_days = []
    current_run = []

    for _, row in df.iterrows():
        if row["has_both"]:
            current_run.append(row["date"])
        else:
            if len(current_run) >= 3:
                consecutive_days.append(current_run)
            current_run = []

    if len(current_run) >= 3:
        consecutive_days.append(current_run)

    if consecutive_days:
        for i, run in enumerate(consecutive_days):
            print(f"Window {i + 1}: {run[0]} to {run[-1]} ({len(run)} days)")
    else:
        print("No windows with 3+ consecutive days of both sensors found.")

    # Sparse data handler analysis
    print("\n\nSparse Data Handler Analysis:")
    print("-" * 70)

    handler = SparseDataHandler()

    sleep_dates = list(sleep_by_date.keys())
    activity_dates = list(activity_by_date.keys())

    sleep_density = handler.assess_density(sleep_dates)
    activity_density = handler.assess_density(activity_dates)

    print(
        f"Sleep: {sleep_density.density_class.name} "
        f"(coverage: {sleep_density.coverage_ratio:.1%}, "
        f"max gap: {sleep_density.max_gap_days} days, "
        f"longest run: {sleep_density.consecutive_days} days)"
    )

    print(
        f"Activity: {activity_density.density_class.name} "
        f"(coverage: {activity_density.coverage_ratio:.1%}, "
        f"max gap: {activity_density.max_gap_days} days, "
        f"longest run: {activity_density.consecutive_days} days)"
    )

    # Recommendations
    print("\n\nRecommendations:")
    print("-" * 70)

    if len(consecutive_days) == 0:
        print("⚠️  No overlapping periods suitable for full circadian analysis")
        print("   - Consider processing sleep and activity data separately")
        print("   - Use sensor-specific features where available")
    else:
        total_overlap_days = sum(len(run) for run in consecutive_days)
        print(f"✓  Found {total_overlap_days} days with both sensors")
        print(
            f"   - Best window: {consecutive_days[0][0]} to {consecutive_days[0][-1]}"
        )
        print("   - Can perform limited circadian analysis on these windows")

    if sleep_density.density_class.name in ["SPARSE", "VERY_SPARSE"]:
        print(f"\n⚠️  Sleep data is {sleep_density.density_class.name}")
        print("   - Consider using forward-fill interpolation for small gaps")
        print("   - Add confidence scores to predictions")

    if (
        activity_density.density_class.name == "DENSE"
        and sleep_density.density_class.name != "DENSE"
    ):
        print("\n💡 Activity data is dense while sleep is sparse")
        print("   - Can extract robust activity features")
        print("   - Sleep features will have lower confidence")

    # Save detailed coverage report
    coverage_path = Path("output/data_coverage_report.csv")
    df.to_csv(coverage_path, index=False)
    print(f"\n\nDetailed coverage saved to: {coverage_path}")


if __name__ == "__main__":
    analyze_coverage()
