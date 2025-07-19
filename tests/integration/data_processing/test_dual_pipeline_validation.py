"""
Dual Pipeline Validation Test
Tests both JSON and XML pipelines with real Apple Health data.
Validates our architecture decisions and data coverage.
"""

import os
from pathlib import Path

import pytest

from big_mood_detector.domain.services.feature_extraction_service import (
    FeatureExtractionService,
)
from big_mood_detector.infrastructure.parsers.json import (
    ActivityJSONParser,
    HeartRateJSONParser,
    SleepJSONParser,
)
from big_mood_detector.infrastructure.parsers.parser_factory import (
    ParserFactory,
    UnifiedHealthDataParser,
)
from big_mood_detector.infrastructure.parsers.xml import (
    ActivityParser,
    HeartRateParser,
    SleepParser,
    StreamingXMLParser,
)


class TestDualPipelineValidation:
    """Validate both JSON and XML pipelines with real data."""

    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path("/Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector")

    @pytest.fixture
    def json_data_path(self, project_root):
        """Path to JSON data from Health Auto Export."""
        return project_root / "data" / "input" / "health_auto_export"

    @pytest.fixture
    def xml_data_path(self, project_root):
        """Path to XML data from Apple Health export."""
        return project_root / "data" / "input" / "apple_export"

    def test_xml_pipeline_smoke_test(self, xml_data_path):
        """Basic test that XML parsing works with real export.xml."""
        export_file = xml_data_path / "export.xml"

        if not export_file.exists():
            pytest.skip("export.xml not found - run this test locally")

        # Get file size for context
        file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
        print(f"\nFound export.xml ({file_size_mb:.1f} MB)")

        # Use streaming parser for large files
        streaming_parser = StreamingXMLParser()

        # Count records by type using streaming
        sleep_count = 0
        activity_count = 0
        heart_count = 0

        print("Streaming through XML file...")

        # Stream through all entities
        for entity in streaming_parser.parse_file(export_file, entity_type="all"):
            entity_type = type(entity).__name__
            if "Sleep" in entity_type:
                sleep_count += 1
            elif "Activity" in entity_type:
                activity_count += 1
            elif "Heart" in entity_type:
                heart_count += 1

            # Progress indicator for large files
            total = sleep_count + activity_count + heart_count
            if total % 10000 == 0:
                print(f"  Processed {total:,} records...")

        print(f"XML Sleep records: {sleep_count:,}")
        print(f"XML Activity records: {activity_count:,}")
        print(f"XML Heart rate records: {heart_count:,}")

        # Basic validation
        assert (
            sleep_count > 0 or activity_count > 0 or heart_count > 0
        ), "Should find at least some records"

    def test_compare_data_coverage(self, json_data_path, xml_data_path):
        """Compare data coverage between JSON and XML formats."""
        results = {
            "json": {"sleep": 0, "activity": 0, "heart_rate": 0, "dates": set()},
            "xml": {"sleep": 0, "activity": 0, "heart_rate": 0, "dates": set()},
        }

        # Parse JSON data
        if (json_data_path / "Sleep Analysis.json").exists():
            parser = SleepJSONParser()
            records = parser.parse_file(json_data_path / "Sleep Analysis.json")
            results["json"]["sleep"] = len(records)
            for r in records:
                results["json"]["dates"].add(r.start_date.date())

        if (json_data_path / "Step Count.json").exists():
            parser = ActivityJSONParser()
            records = parser.parse_file(json_data_path / "Step Count.json")
            results["json"]["activity"] = len(records)
            for r in records:
                results["json"]["dates"].add(r.start_date.date())

        if (json_data_path / "Heart Rate.json").exists():
            parser = HeartRateJSONParser()
            records = parser.parse_file(json_data_path / "Heart Rate.json")
            results["json"]["heart_rate"] = len(records)
            for r in records:
                results["json"]["dates"].add(r.timestamp.date())

        # Parse XML data if available
        export_file = xml_data_path / "export.xml"
        if export_file.exists():
            file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
            if file_size_mb > 100:  # Skip large files in tests
                print(f"Skipping XML parsing - file too large ({file_size_mb:.1f} MB)")
            else:
                with open(export_file) as f:
                    xml_data = f.read()

                # Sleep
                parser = SleepParser()
                records = parser.parse_to_entities(xml_data)
                results["xml"]["sleep"] = len(records)
                for r in records:
                    results["xml"]["dates"].add(r.start_date.date())

                # Activity
                parser = ActivityParser()
                records = parser.parse_to_entities(xml_data)
                results["xml"]["activity"] = len(records)
                for r in records:
                    results["xml"]["dates"].add(r.start_date.date())

                # Heart rate
                parser = HeartRateParser()
                records = parser.parse_to_entities(xml_data)
                results["xml"]["heart_rate"] = len(records)
                for r in records:
                    results["xml"]["dates"].add(r.timestamp.date())

        # Print comparison
        print("\n=== Data Coverage Comparison ===")
        print(f"JSON Sleep records: {results['json']['sleep']}")
        print(f"XML Sleep records: {results['xml']['sleep']}")
        print(f"\nJSON Activity records: {results['json']['activity']}")
        print(f"XML Activity records: {results['xml']['activity']}")
        print(f"\nJSON Heart rate records: {results['json']['heart_rate']}")
        print(f"XML Heart rate records: {results['xml']['heart_rate']}")

        if results["json"]["dates"] and results["xml"]["dates"]:
            json_start = min(results["json"]["dates"])
            json_end = max(results["json"]["dates"])
            xml_start = min(results["xml"]["dates"])
            xml_end = max(results["xml"]["dates"])

            print(f"\nJSON date range: {json_start} to {json_end}")
            print(f"XML date range: {xml_start} to {xml_end}")

            overlap = results["json"]["dates"] & results["xml"]["dates"]
            print(f"\nOverlapping days: {len(overlap)}")

    def test_unified_parser(self, json_data_path, xml_data_path):
        """Test the unified parser with both data sources."""
        parser = UnifiedHealthDataParser()

        # Add JSON sources
        json_files = {
            "sleep": "Sleep Analysis.json",
            "activity": "Step Count.json",
            "heart_rate": "Heart Rate.json",
        }

        for data_type, filename in json_files.items():
            file_path = json_data_path / filename
            if file_path.exists():
                parser.add_json_source(file_path, data_type)
                print(f"Added JSON {data_type} data")

        # Add XML export if available
        export_file = xml_data_path / "export.xml"
        if export_file.exists():
            file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
            if file_size_mb > 100:
                print(f"Skipping XML export - file too large ({file_size_mb:.1f} MB)")
            else:
                parser.add_xml_export(export_file)
                print("Added XML export data")

        # Get all records
        all_records = parser.get_all_records()

        print("\nUnified Parser Results:")
        print(f"Total sleep records: {len(all_records['sleep'])}")
        print(f"Total activity records: {len(all_records['activity'])}")
        print(f"Total heart rate records: {len(all_records['heart_rate'])}")

        # Run feature extraction
        feature_service = FeatureExtractionService()
        features = feature_service.extract_features(
            sleep_records=all_records["sleep"],
            activity_records=all_records["activity"],
            heart_records=all_records["heart_rate"],
        )

        print(f"\nExtracted features for {len(features)} days")

        # Find days with complete data
        complete_days = [
            date
            for date, f in features.items()
            if f.sleep_duration_hours > 0 and f.total_steps > 0 and f.avg_resting_hr > 0
        ]
        print(f"Days with complete data: {len(complete_days)}")

    def test_parser_factory_format_detection(self, json_data_path, xml_data_path):
        """Test automatic format detection."""
        # Test JSON detection
        json_file = json_data_path / "Sleep Analysis.json"
        if json_file.exists():
            format_type = ParserFactory.detect_format(json_file)
            assert format_type == "json"
            print("✓ Correctly detected JSON format")

        # Test XML detection
        xml_file = xml_data_path / "export.xml"
        if xml_file.exists():
            format_type = ParserFactory.detect_format(xml_file)
            assert format_type == "xml"
            print("✓ Correctly detected XML format")

    def test_clinical_insights_comparison(self, json_data_path, xml_data_path):
        """Compare clinical insights from both data sources."""
        feature_service = FeatureExtractionService()

        # Get features from JSON only
        json_parser = UnifiedHealthDataParser()
        for data_type, filename in [
            ("sleep", "Sleep Analysis.json"),
            ("activity", "Step Count.json"),
            ("heart_rate", "Heart Rate.json"),
        ]:
            file_path = json_data_path / filename
            if file_path.exists():
                json_parser.add_json_source(file_path, data_type)

        json_data = json_parser.get_all_records()
        json_features = feature_service.extract_features(
            sleep_records=json_data["sleep"],
            activity_records=json_data["activity"],
            heart_records=json_data["heart_rate"],
        )

        # Count clinical insights
        json_clinical_days = sum(
            1 for f in json_features.values() if f.is_clinically_significant
        )

        print("\nClinical Insights from JSON:")
        print(f"Days analyzed: {len(json_features)}")
        print(f"Clinically significant days: {json_clinical_days}")

        # If XML available, compare
        export_file = xml_data_path / "export.xml"
        if export_file.exists():
            file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
            if file_size_mb > 100:
                print(
                    f"Skipping XML comparison - file too large ({file_size_mb:.1f} MB)"
                )
            else:
                xml_parser = UnifiedHealthDataParser()
                xml_parser.add_xml_export(export_file)
                xml_data = xml_parser.get_all_records()

                xml_features = feature_service.extract_features(
                    sleep_records=xml_data["sleep"],
                    activity_records=xml_data["activity"],
                    heart_records=xml_data["heart_rate"],
                )

                xml_clinical_days = sum(
                    1 for f in xml_features.values() if f.is_clinically_significant
                )

                print("\nClinical Insights from XML:")
                print(f"Days analyzed: {len(xml_features)}")
                print(f"Clinically significant days: {xml_clinical_days}")

                # Compare overlapping dates
                common_dates = set(json_features.keys()) & set(xml_features.keys())
                if common_dates:
                    print(f"\nComparing {len(common_dates)} overlapping days:")

                    agreement = 0
                    for date in list(common_dates)[:5]:  # Show first 5
                        json_sig = json_features[date].is_clinically_significant
                        xml_sig = xml_features[date].is_clinically_significant

                        if json_sig == xml_sig:
                            agreement += 1
                            print(
                                f"  {date}: Agreement (both {'significant' if json_sig else 'normal'})"
                            )
                        else:
                            print(
                                f"  {date}: Disagreement (JSON: {json_sig}, XML: {xml_sig})"
                            )

    @pytest.mark.slow
    def test_performance_comparison(self, json_data_path, xml_data_path):
        """Compare parsing performance between formats."""
        import time

        # Time JSON parsing
        json_start = time.time()
        json_parser = UnifiedHealthDataParser()

        for data_type, filename in [
            ("sleep", "Sleep Analysis.json"),
            ("activity", "Step Count.json"),
            ("heart_rate", "Heart Rate.json"),
        ]:
            file_path = json_data_path / filename
            if file_path.exists():
                json_parser.add_json_source(file_path, data_type)

        json_time = time.time() - json_start
        json_records = json_parser.get_all_records()
        json_total = sum(len(records) for records in json_records.values())

        print("\nJSON Performance:")
        print(f"Parse time: {json_time:.2f} seconds")
        print(f"Total records: {json_total}")
        print(f"Records/second: {json_total/json_time:.0f}")

        # Time XML parsing if available
        export_file = xml_data_path / "export.xml"
        if export_file.exists():
            file_size_mb = os.path.getsize(export_file) / (1024 * 1024)
            if file_size_mb > 100:
                print(
                    f"Skipping XML performance test - file too large ({file_size_mb:.1f} MB)"
                )
            else:
                xml_start = time.time()
                xml_parser = UnifiedHealthDataParser()
                xml_parser.add_xml_export(export_file)
                xml_time = time.time() - xml_start

                xml_records = xml_parser.get_all_records()
                xml_total = sum(len(records) for records in xml_records.values())

                print("\nXML Performance:")
                print(f"Parse time: {xml_time:.2f} seconds")
                print(f"Total records: {xml_total}")
                print(f"Records/second: {xml_total/xml_time:.0f}")

                print("\nPerformance Ratio:")
                print(f"XML is {xml_time/json_time:.1f}x slower than JSON")
                print(f"XML has {xml_total/json_total:.1f}x more records than JSON")
