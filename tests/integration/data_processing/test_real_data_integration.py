"""
Integration Test with Real Apple Health JSON Data
Validates the complete pipeline from parsing to feature extraction.
"""

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


class TestRealDataIntegration:
    """Integration tests using actual Apple Health export data."""

    @pytest.fixture
    def data_path(self):
        """Path to real data directory."""
        return Path("data/input/health_auto_export")

    @pytest.fixture
    def parsers(self):
        """Initialize all parsers."""
        return {
            "sleep": SleepJSONParser(),
            "activity": ActivityJSONParser(),
            "heart_rate": HeartRateJSONParser(),
        }

    def test_complete_pipeline_with_real_data(self, data_path, parsers):
        """Test the complete data processing pipeline with real Apple Health data."""
        # ARRANGE
        feature_service = FeatureExtractionService()

        # ACT - Parse all available data
        sleep_file = data_path / "Sleep Analysis.json"
        step_file = data_path / "Step Count.json"
        hr_file = data_path / "Heart Rate.json"

        # Check which files exist
        available_data = {}
        if sleep_file.exists():
            available_data["sleep"] = parsers["sleep"].parse_file(sleep_file)
            print(f"Parsed {len(available_data['sleep'])} sleep records")

        if step_file.exists():
            available_data["steps"] = parsers["activity"].parse_file(step_file)
            print(f"Parsed {len(available_data['steps'])} activity records")

        if hr_file.exists():
            available_data["heart_rate"] = parsers["heart_rate"].parse_file(hr_file)
            print(f"Parsed {len(available_data['heart_rate'])} heart rate records")

        # Extract features
        features = feature_service.extract_features(
            sleep_records=available_data.get("sleep", []),
            activity_records=available_data.get("steps", []),
            heart_records=available_data.get("heart_rate", []),
        )

        # ASSERT
        assert len(features) > 0, "Should extract features from real data"

        # Analyze the data
        print(f"\nExtracted features for {len(features)} days")

        # Find interesting patterns
        clinically_significant_days = [
            (date, f) for date, f in features.items() if f.is_clinically_significant
        ]

        if clinically_significant_days:
            print(
                f"\nFound {len(clinically_significant_days)} clinically significant days:"
            )
            for date, features_day in clinically_significant_days[:5]:  # Show first 5
                print(f"  {date}: {', '.join(features_day.clinical_notes)}")

        # Check data quality
        complete_days = [
            date
            for date, f in features.items()
            if f.sleep_duration_hours > 0 and f.total_steps > 0
        ]
        print(f"\nDays with complete data: {len(complete_days)}/{len(features)}")

        # Validate features are reasonable
        for _date, day_features in features.items():
            # Sleep should be 0-24 hours
            assert 0 <= day_features.sleep_duration_hours <= 24
            # Steps should be non-negative
            assert day_features.total_steps >= 0
            # Heart rate should be in valid range (0-220)
            if day_features.avg_resting_hr > 0:
                assert 20 <= day_features.avg_resting_hr <= 220

    def test_parse_additional_data_sources(self, data_path, parsers):
        """Test parsing of additional data sources we need to integrate."""
        # Check what other data files are available
        data_files = list(data_path.glob("*.json"))
        print(f"\nAvailable data files: {len(data_files)}")
        for file in sorted(data_files):
            print(f"  - {file.name}")

        # Try parsing HRV data
        hrv_file = data_path / "Heart Rate Variability.json"
        if hrv_file.exists():
            # Load the JSON data first
            import json

            with open(hrv_file) as f:
                hrv_data = json.load(f)

            hrv_records = parsers["heart_rate"].parse_hrv(hrv_data)
            print(f"\nParsed {len(hrv_records)} HRV records")
            if hrv_records:
                # Check HRV values are reasonable (0-200ms typically)
                for record in hrv_records[:5]:
                    assert 0 <= record.value <= 300, f"Invalid HRV: {record.value}"

        # Try parsing Resting Heart Rate
        resting_hr_file = data_path / "Resting Heart Rate.json"
        if resting_hr_file.exists():
            # Need to read the file first to check format
            import json

            with open(resting_hr_file) as f:
                resting_data = json.load(f)

            resting_records = parsers["heart_rate"].parse_resting_heart_rate(
                resting_data
            )
            print(f"\nParsed {len(resting_records)} resting HR records")

    def test_data_quality_analysis(self, data_path, parsers):
        """Analyze the quality and completeness of the real data."""
        # Parse all core data
        sleep_records = []
        activity_records = []
        heart_records = []

        try:
            sleep_records = parsers["sleep"].parse_file(
                data_path / "Sleep Analysis.json"
            )
        except Exception as e:
            print(f"Could not parse sleep data: {e}")

        try:
            activity_records = parsers["activity"].parse_file(
                data_path / "Step Count.json"
            )
        except Exception as e:
            print(f"Could not parse activity data: {e}")

        try:
            heart_records = parsers["heart_rate"].parse_file(
                data_path / "Heart Rate.json"
            )
        except Exception as e:
            print(f"Could not parse heart rate data: {e}")

        # Analyze date ranges
        if sleep_records:
            sleep_dates = [r.start_date.date() for r in sleep_records]
            print(f"\nSleep data range: {min(sleep_dates)} to {max(sleep_dates)}")

        if activity_records:
            activity_dates = [r.start_date.date() for r in activity_records]
            print(
                f"Activity data range: {min(activity_dates)} to {max(activity_dates)}"
            )

        if heart_records:
            hr_dates = [r.timestamp.date() for r in heart_records]
            print(f"Heart rate data range: {min(hr_dates)} to {max(hr_dates)}")

    @pytest.mark.parametrize(
        "filename,expected_keys",
        [
            ("Sleep Analysis.json", ["rem", "deep", "core", "totalSleep"]),
            ("Heart Rate.json", ["Min", "Max", "Avg"]),
            ("Step Count.json", ["qty", "source"]),
        ],
    )
    def test_data_format_validation(self, data_path, filename, expected_keys):
        """Validate that the JSON files have the expected format."""
        import json

        file_path = data_path / filename
        if not file_path.exists():
            pytest.skip(f"{filename} not found")

        with open(file_path) as f:
            data = json.load(f)

        # Check structure
        assert "data" in data
        assert "metrics" in data["data"]
        assert len(data["data"]["metrics"]) > 0

        # Check first data entry has expected keys
        first_metric = data["data"]["metrics"][0]
        if "data" in first_metric and len(first_metric["data"]) > 0:
            first_entry = first_metric["data"][0]
            for key in expected_keys:
                if key not in first_entry:
                    print(f"Warning: Expected key '{key}' not found in {filename}")
                    print(f"Available keys: {list(first_entry.keys())}")
