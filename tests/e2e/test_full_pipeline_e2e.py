"""
End-to-End Full Pipeline Test

Tests the complete flow: parse → features → predict
Uses minimal test data to verify integration.
"""

import json
import subprocess
import sys

import pytest

# Mark all tests in this module as e2e and set timeout
pytestmark = [pytest.mark.e2e, pytest.mark.timeout(30)]


class TestFullPipeline:
    """Test complete pipeline from data to predictions."""

    @pytest.fixture
    def sample_xml_data(self):
        """Create minimal valid Apple Health export XML."""
        # Using list join to avoid whitespace issues
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!DOCTYPE HealthData [",
            "<!ELEMENT HealthData (ExportDate, Record*)>",
            "<!ATTLIST ExportDate value CDATA #REQUIRED>",
            "<!ELEMENT Record EMPTY>",
            "<!ATTLIST Record type CDATA #REQUIRED",
            "    sourceName CDATA #REQUIRED",
            "    unit CDATA #IMPLIED",
            "    startDate CDATA #REQUIRED",
            "    endDate CDATA #REQUIRED",
            "    value CDATA #IMPLIED>",
            "]>",
            '<HealthData locale="en_US">',
            '  <ExportDate value="2024-01-15 10:00:00 -0800"/>',
            "",
            "  <!-- Sleep data for 7 days -->",
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-01 23:00:00 -0800" endDate="2024-01-02 07:00:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-02 23:30:00 -0800" endDate="2024-01-03 07:30:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-03 22:45:00 -0800" endDate="2024-01-04 06:45:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-04 23:15:00 -0800" endDate="2024-01-05 07:15:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-05 22:30:00 -0800" endDate="2024-01-06 06:30:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-06 23:00:00 -0800" endDate="2024-01-07 07:00:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-07 23:45:00 -0800" endDate="2024-01-08 07:45:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            "",
            "  <!-- Activity data (step counts) for all 7 days -->",
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-01 08:00:00 -0800" endDate="2024-01-01 09:00:00 -0800" value="500"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-01 12:00:00 -0800" endDate="2024-01-01 13:00:00 -0800" value="1000"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-01 18:00:00 -0800" endDate="2024-01-01 19:00:00 -0800" value="800"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-02 10:00:00 -0800" endDate="2024-01-02 11:00:00 -0800" value="1200"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-03 14:00:00 -0800" endDate="2024-01-03 15:00:00 -0800" value="900"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-04 16:00:00 -0800" endDate="2024-01-04 17:00:00 -0800" value="1100"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-05 09:00:00 -0800" endDate="2024-01-05 10:00:00 -0800" value="700"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-06 11:00:00 -0800" endDate="2024-01-06 12:00:00 -0800" value="1300"/>',
            '  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="iPhone" unit="count"',
            '    startDate="2024-01-07 13:00:00 -0800" endDate="2024-01-07 14:00:00 -0800" value="850"/>',
            "</HealthData>",
        ]
        return "\n".join(xml_lines)

    @pytest.mark.slow
    def test_predict_command_e2e(self, sample_xml_data, tmp_path):
        """Test full pipeline: XML → features → predictions."""
        # Given: Sample XML data
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(sample_xml_data)

        output_file = tmp_path / "predictions.json"

        # When: Running predict command
        import sys
        import os
        env = os.environ.copy()
        env["TESTING"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "big_mood_detector.interfaces.cli.main",
                "predict",
                str(xml_file),
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-07",
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        # Then: Command should succeed
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # And: Output file should exist
        assert output_file.exists()

        # And: JSON should have expected structure
        with open(output_file) as f:
            predictions = json.load(f)

        assert "summary" in predictions
        assert "daily_predictions" in predictions
        assert "confidence" in predictions
        assert "metadata" in predictions

        # Check summary structure
        summary = predictions["summary"]
        assert "avg_depression_risk" in summary
        assert "avg_hypomanic_risk" in summary
        assert "avg_manic_risk" in summary
        assert "days_analyzed" in summary

        # Risk values should be probabilities
        assert 0.0 <= summary["avg_depression_risk"] <= 1.0
        assert 0.0 <= summary["avg_hypomanic_risk"] <= 1.0
        assert 0.0 <= summary["avg_manic_risk"] <= 1.0

        # The model needs 3 days of historical data before making predictions
        # So with 7 days of data (Jan 1-7), we get 4 predictions (Jan 4-7)
        assert summary["days_analyzed"] == 4

        # Verify we have predictions for the expected days
        daily_predictions = predictions["daily_predictions"]
        expected_dates = ["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]
        assert sorted(daily_predictions.keys()) == expected_dates

    def test_predict_with_insufficient_data(self, tmp_path):
        """Test pipeline handles insufficient data gracefully."""
        # Given: XML with only 2 days of data
        minimal_xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<!DOCTYPE HealthData [",
            "<!ELEMENT HealthData (ExportDate, Record*)>",
            "<!ATTLIST ExportDate value CDATA #REQUIRED>",
            "<!ELEMENT Record EMPTY>",
            "<!ATTLIST Record type CDATA #REQUIRED",
            "    sourceName CDATA #REQUIRED",
            "    startDate CDATA #REQUIRED",
            "    endDate CDATA #REQUIRED",
            "    value CDATA #IMPLIED>",
            "]>",
            '<HealthData locale="en_US">',
            '  <ExportDate value="2024-01-15 10:00:00 -0800"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-01 23:00:00 -0800" endDate="2024-01-02 07:00:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            '  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="Apple Watch"',
            '    startDate="2024-01-02 23:00:00 -0800" endDate="2024-01-03 07:00:00 -0800"',
            '    value="HKCategoryValueSleepAnalysisAsleepCore"/>',
            "</HealthData>",
        ]
        minimal_xml = "\n".join(minimal_xml_lines)

        xml_file = tmp_path / "minimal.xml"
        xml_file.write_text(minimal_xml)

        # When: Using pipeline directly to avoid subprocess model loading
        from big_mood_detector.application.use_cases.process_health_data_use_case import (
            MoodPredictionPipeline,
            PipelineConfig,
        )
        from big_mood_detector.test_support.predictors import ConstantMoodPredictor
        from datetime import date
        
        config = PipelineConfig(min_days_required=7)
        pipeline = MoodPredictionPipeline.for_testing(
            predictor=ConstantMoodPredictor(),
            config=config,
            disable_ensemble=True
        )
        
        result = pipeline.process_apple_health_file(
            file_path=xml_file,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 7),
        )
        
        # Then: Should handle gracefully with warnings
        assert result is not None
        assert result.has_warnings is True
        assert "Insufficient data" in result.warnings[0]

    @pytest.mark.slow
    def test_label_import_export_e2e(self, tmp_path):
        """Test label CLI import/export functionality."""
        # Given: CSV with labeled episodes
        csv_content = """date,mood,severity,notes
2024-01-01,depressive,3,Test episode 1
2024-01-05,manic,4,Test episode 2
"""
        csv_file = tmp_path / "episodes.csv"
        csv_file.write_text(csv_content)

        db_file = tmp_path / "test_labels.db"
        export_file = tmp_path / "exported.csv"

        # When: Importing episodes
        import os
        env = os.environ.copy()
        env["TESTING"] = "1"
        import_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "big_mood_detector.interfaces.cli.main",
                "label",
                "import",
                str(csv_file),
                "--db",
                str(db_file),
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        # Then: Import should succeed
        assert import_result.returncode == 0
        assert "Imported 2 episodes" in import_result.stdout

        # When: Exporting from database
        export_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "big_mood_detector.interfaces.cli.main",
                "label",
                "export",
                "--db",
                str(db_file),
                "--output",
                str(export_file),
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        # Then: Export should succeed
        assert export_result.returncode == 0
        assert export_file.exists()

        # And: Exported data should match
        import pandas as pd

        df = pd.read_csv(export_file)
        assert len(df) == 2
        assert set(df["label"]) == {"depressive", "manic"}

    @pytest.mark.slow
    @pytest.mark.parametrize("format,extension", [("json", "json"), ("csv", "csv")])
    def test_predict_output_formats(self, sample_xml_data, tmp_path, format, extension):
        """Test different output formats work correctly."""
        xml_file = tmp_path / "export.xml"
        xml_file.write_text(sample_xml_data)

        output_file = tmp_path / f"predictions.{extension}"

        # When: Running predict with specific format
        import os
        env = os.environ.copy()
        env["TESTING"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "big_mood_detector.interfaces.cli.main",
                "predict",
                str(xml_file),
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-07",
                "--output",
                str(output_file),
                "--format",
                format,
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        # Then: Should succeed
        assert result.returncode == 0
        assert output_file.exists()

        # And: File should have content
        assert output_file.stat().st_size > 0
