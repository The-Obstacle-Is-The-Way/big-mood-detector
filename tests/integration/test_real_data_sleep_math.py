"""
Test with REAL Apple Health data to trace the sleep duration bug.

This is not mock data - this is YOUR actual sleep data!
"""
from datetime import timedelta
from pathlib import Path

import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
    PipelineConfig,
)
from big_mood_detector.domain.services.sleep_aggregator import SleepAggregator
from big_mood_detector.infrastructure.parsers.parser_factory import ParserFactory
from big_mood_detector.infrastructure.repositories.file_baseline_repository import (
    FileBaselineRepository,
)


class TestRealDataSleepMath:
    """Trace the sleep math bug using REAL Apple Health exports."""

    def test_trace_sleep_math_with_real_data(self, tmp_path):
        """
        Use a real Apple Health export to trace where sleep hours get lost.

        We'll trace the math at every step!
        """
        # Pick a real export file
        export_file = Path("/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector/data/input/apple_export/export.xml")

        if not export_file.exists():
            pytest.skip("No real export file found")

        print(f"\n🔍 TRACING REAL DATA: {export_file}")

        # Step 1: Parse the raw data
        parser = ParserFactory.create_parser(export_file)
        parsed_data = parser.parse_file(str(export_file))

        sleep_records = parsed_data["sleep_records"]
        activity_records = parsed_data["activity_records"]
        heart_records = parsed_data.get("heart_rate_records", [])

        print("\n📊 PARSED DATA:")
        print(f"   Sleep records: {len(sleep_records)}")
        print(f"   Activity records: {len(activity_records)}")
        print(f"   Heart records: {len(heart_records)}")

        # Step 2: Let's manually aggregate sleep for a specific date range
        # Pick the most recent 30 days of data
        if sleep_records:
            latest_date = max(r.end_date.date() for r in sleep_records)
            start_date = latest_date - timedelta(days=30)

            print(f"\n📅 DATE RANGE: {start_date} to {latest_date}")

            # Use SleepAggregator directly
            aggregator = SleepAggregator()
            sleep_summaries = aggregator.aggregate_daily(sleep_records)

            # Look at a few days
            print("\n💤 SLEEP AGGREGATOR OUTPUT (last 7 days):")
            for day in range(7):
                check_date = latest_date - timedelta(days=day)
                if check_date in sleep_summaries:
                    summary = sleep_summaries[check_date]
                    print(f"   {check_date}: {summary.total_sleep_hours:.1f}h sleep")

            # Step 3: Now run through the pipeline
            baseline_repo = FileBaselineRepository(tmp_path / "baselines")

            config = PipelineConfig()
            config.enable_personal_calibration = True
            config.user_id = "real_user_test"
            config.min_days_required = 3

            pipeline = MoodPredictionPipeline(
                config=config,
                baseline_repository=baseline_repo
            )

            # Process just 7 days
            pipeline.process_health_data(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                target_date=latest_date
            )

            # Check baseline
            baseline = baseline_repo.get_baseline("real_user_test")
            if baseline:
                print("\n🎯 BASELINE RESULT:")
                print(f"   Sleep mean: {baseline.sleep_mean:.1f}h")
                print(f"   Sleep std: {baseline.sleep_std:.1f}h")
                print(f"   Data points: {baseline.data_points}")

                # Compare with direct aggregation
                recent_sleep = [
                    sleep_summaries[d].total_sleep_hours
                    for d in sleep_summaries
                    if latest_date - timedelta(days=7) <= d <= latest_date
                ]
                if recent_sleep:
                    manual_mean = sum(recent_sleep) / len(recent_sleep)
                    print("\n⚠️  COMPARISON:")
                    print(f"   Manual calculation mean: {manual_mean:.1f}h")
                    print(f"   Pipeline baseline mean: {baseline.sleep_mean:.1f}h")
                    print(f"   DIFFERENCE: {abs(manual_mean - baseline.sleep_mean):.1f}h")

                    # The bug check
                    if abs(manual_mean - baseline.sleep_mean) > 2.0:
                        print(f"\n🐛 BUG CONFIRMED! Pipeline loses {abs(manual_mean - baseline.sleep_mean):.1f} hours!")

    def test_single_day_detailed_trace(self, tmp_path):
        """Trace a single day's sleep calculation in detail."""
        export_file = Path("/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector/data/input/apple_export/export.xml")

        if not export_file.exists():
            pytest.skip("No real export file found")

        # Parse data
        parser = ParserFactory.create_parser(export_file)
        parsed_data = parser.parse_file(str(export_file))

        sleep_records = parsed_data["sleep_records"]

        if not sleep_records:
            pytest.skip("No sleep records found")

        # Pick a recent date with sleep data
        latest_date = max(r.end_date.date() for r in sleep_records)

        # Get all sleep records for that date
        day_sleep = [
            r for r in sleep_records
            if r.start_date.date() <= latest_date <= r.end_date.date()
        ]

        print(f"\n🔍 DETAILED TRACE FOR {latest_date}:")
        print(f"   Found {len(day_sleep)} sleep records")

        total_hours = 0
        for i, record in enumerate(day_sleep):
            duration = record.duration_hours
            total_hours += duration
            print(f"   Record {i+1}: {record.start_date.strftime('%H:%M')} → {record.end_date.strftime('%H:%M')} = {duration:.1f}h")
            print(f"      State: {record.state.value}")
            print(f"      Source: {record.source_name}")

        print(f"\n   TOTAL RAW SLEEP: {total_hours:.1f} hours")

        # Now aggregate
        aggregator = SleepAggregator()
        summaries = aggregator.aggregate_daily(day_sleep)

        if latest_date in summaries:
            summary = summaries[latest_date]
            print(f"\n   AGGREGATOR RESULT: {summary.total_sleep_hours:.1f} hours")
            print(f"   Sleep efficiency: {summary.sleep_efficiency:.1%}")
            print(f"   Sleep sessions: {summary.sleep_sessions}")

            if abs(total_hours - summary.total_sleep_hours) > 0.1:
                print(f"\n⚠️  MISMATCH: Raw total ({total_hours:.1f}h) != Aggregated ({summary.total_sleep_hours:.1f}h)")
