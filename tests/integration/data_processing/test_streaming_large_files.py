"""
Integration test for streaming large Apple Health XML files.

Tests the streaming parser with real large export files.
"""

import os
from pathlib import Path

import pytest

from big_mood_detector.domain.services.feature_extraction_service import (
    FeatureExtractionService,
)
from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser


@pytest.mark.large
class TestStreamingLargeFiles:
    """Test streaming parser with large real-world files."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        test_file = Path(__file__)
        return test_file.parent.parent.parent

    @pytest.fixture
    def xml_data_path(self, project_root):
        """Path to XML data from Apple Health export."""
        return project_root / "data" / "input" / "apple_export"

    def test_stream_large_export_xml(self, xml_data_path):
        """Test streaming through large export.xml file."""
        export_file = xml_data_path / "export.xml"

        if not export_file.exists():
            pytest.skip("export.xml not found - run this test locally with real data")

        # Get file size
        file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
        print(f"\nProcessing export.xml ({file_size_mb:.1f} MB)")

        # Use streaming parser
        streaming_parser = StreamingXMLParser()

        # Track counts and memory usage
        counts = {"sleep": 0, "activity": 0, "heart": 0, "total": 0}

        print("Streaming through XML file...")

        # Process in batches for efficiency
        for batch in streaming_parser.parse_file_in_batches(
            export_file, batch_size=5000
        ):
            for entity in batch:
                entity_type = type(entity).__name__
                if "Sleep" in entity_type:
                    counts["sleep"] += 1
                elif "Activity" in entity_type:
                    counts["activity"] += 1
                elif "Heart" in entity_type:
                    counts["heart"] += 1
                counts["total"] += 1

            # Progress update
            if counts["total"] % 50000 == 0:
                print(f"  Processed {counts['total']:,} records...")

        # Final report
        print("\nFinal counts:")
        print(f"  Sleep records: {counts['sleep']:,}")
        print(f"  Activity records: {counts['activity']:,}")
        print(f"  Heart rate records: {counts['heart']:,}")
        print(f"  Total records: {counts['total']:,}")

        # Validate we found data
        assert counts["total"] > 0, "Should find at least some records"

        # If file is truly large, should have many records
        if file_size_mb > 100:
            assert counts["total"] > 10000, "Large file should have many records"

    def test_extract_features_from_large_file(self, xml_data_path):
        """Test feature extraction using streaming for large files."""
        export_file = xml_data_path / "export.xml"

        if not export_file.exists():
            pytest.skip("export.xml not found - run this test locally with real data")

        file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
        print(f"\nExtracting features from export.xml ({file_size_mb:.1f} MB)")

        # Use streaming parser
        streaming_parser = StreamingXMLParser()
        feature_service = FeatureExtractionService()

        # Collect records by type
        sleep_records = []
        activity_records = []
        heart_records = []

        print("Collecting records for feature extraction...")

        # Stream and collect records
        for entity in streaming_parser.parse_file(export_file):
            # Check actual entity type
            from big_mood_detector.domain.entities.activity_record import ActivityRecord
            from big_mood_detector.domain.entities.heart_rate_record import (
                HeartRateRecord,
            )
            from big_mood_detector.domain.entities.sleep_record import SleepRecord

            if isinstance(entity, SleepRecord):
                sleep_records.append(entity)
            elif isinstance(entity, ActivityRecord):
                activity_records.append(entity)
            elif isinstance(entity, HeartRateRecord):
                heart_records.append(entity)

            # For large files, limit to recent data for feature extraction
            if len(sleep_records) + len(activity_records) + len(heart_records) > 100000:
                print("  Collected 100k records, proceeding with feature extraction...")
                break

        print(
            f"  Collected: {len(sleep_records)} sleep, {len(activity_records)} activity, {len(heart_records)} heart"
        )

        # Extract features
        print("Extracting clinical features...")
        features = feature_service.extract_features(
            sleep_records=sleep_records,
            activity_records=activity_records,
            heart_records=heart_records,
        )

        # Validate features
        assert len(features) > 0, "Should extract some daily features"

        # Check advanced features if we have enough data AND sleep/activity records
        if len(features) >= 7 and (len(sleep_records) > 0 or len(activity_records) > 0):
            print("Extracting advanced features...")
            advanced_features = feature_service.extract_advanced_features(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                lookback_days=30,
            )

            assert len(advanced_features) > 0, "Should extract advanced features"

            # Check 36-feature requirement
            latest_date = max(advanced_features.keys())
            ml_features = advanced_features[latest_date].to_ml_features()
            assert len(ml_features) == 36, "Should generate 36 features for XGBoost"

            print(f"  Generated {len(ml_features)} ML features for {latest_date}")
        elif len(sleep_records) == 0 and len(activity_records) == 0:
            print("Skipping advanced features test - no sleep/activity data collected")

    def test_memory_efficiency(self, xml_data_path):
        """Test that streaming parser maintains low memory usage."""
        export_file = xml_data_path / "export.xml"

        if not export_file.exists():
            pytest.skip("export.xml not found - run this test locally with real data")

        file_size_mb = os.path.getsize(export_file) / (1024 * 1024)

        # Only run memory test on truly large files
        if file_size_mb < 100:
            pytest.skip(
                f"File too small for memory test ({file_size_mb:.1f} MB < 100 MB)"
            )

        print(f"\nTesting memory efficiency with {file_size_mb:.1f} MB file")

        import tracemalloc

        # Start memory tracking
        tracemalloc.start()
        streaming_parser = StreamingXMLParser()

        # Process file
        record_count = 0
        peak_memory = 0

        for i, _entity in enumerate(streaming_parser.parse_file(export_file)):
            record_count += 1

            # Check memory periodically
            if i % 10000 == 0:
                current, peak = tracemalloc.get_traced_memory()
                peak_memory = max(peak_memory, peak)

                # Memory should stay reasonable even for large files
                assert (
                    peak < 500 * 1024 * 1024
                ), f"Memory usage too high: {peak / 1024 / 1024:.1f} MB"

        tracemalloc.stop()

        peak_mb = peak_memory / 1024 / 1024
        efficiency_ratio = file_size_mb / peak_mb

        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Peak memory: {peak_mb:.1f} MB")
        print(f"  Efficiency ratio: {efficiency_ratio:.1f}x")
        print(f"  Records processed: {record_count:,}")

        # Should use significantly less memory than file size
        assert efficiency_ratio > 2, "Should use less than half the file size in memory"
