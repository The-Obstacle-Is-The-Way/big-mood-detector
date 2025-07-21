"""
Integration test for XML date range filtering.

Tests that the date filtering feature works end-to-end.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

class TestXMLDateFilteringIntegration:
    """Test date filtering integration with the pipeline."""

    @pytest.fixture
    def sample_xml_file(self):
        """Create a sample XML file with records across multiple dates."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<HealthData locale="en_US">\n')

            # Add sleep records across 90 days
            base_date = datetime(2024, 1, 1)
            for i in range(90):
                date = base_date + timedelta(days=i)
                # Morning sleep record (ending in the morning)
                # Use HealthKit date format: "%Y-%m-%d %H:%M:%S %z"
                start_str = date.strftime("%Y-%m-%d %H:%M:%S +0000")
                end_str = (date + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S +0000")
                f.write(f'  <Record type="HKCategoryTypeIdentifierSleepAnalysis" '
                       f'sourceName="Apple Watch" '
                       f'sourceVersion="10.0" '
                       f'startDate="{start_str}" '
                       f'endDate="{end_str}" '
                       f'value="HKCategoryValueSleepAnalysisAsleep"/>\n')

            # Add some activity records
            for i in range(90):
                date = base_date + timedelta(days=i, hours=12)
                start_str = date.strftime("%Y-%m-%d %H:%M:%S +0000")
                end_str = (date + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S +0000")
                f.write(f'  <Record type="HKQuantityTypeIdentifierStepCount" '
                       f'sourceName="iPhone" '
                       f'sourceVersion="17.0" '
                       f'startDate="{start_str}" '
                       f'endDate="{end_str}" '
                       f'value="5000" '
                       f'unit="count"/>\n')

            f.write('</HealthData>')

            file_path = Path(f.name)

        yield file_path
        file_path.unlink()

    def test_process_with_days_back_filter(self, sample_xml_file):
        """Test processing with --days-back filter."""
        from big_mood_detector.application.use_cases.process_health_data_use_case import MoodPredictionPipeline

        pipeline = MoodPredictionPipeline()

        # Process only last 30 days
        end_date = datetime(2024, 3, 31).date()
        start_date = end_date - timedelta(days=30)

        # Process with date filter
        df = pipeline.process_health_export(
            export_path=sample_xml_file,
            output_path=Path("test_output.csv"),
            start_date=start_date,
            end_date=end_date
        )

        # Should have approximately 30 days of data
        assert len(df) <= 31  # Allow for edge cases
        assert len(df) >= 25  # Should have most days

        # Clean up
        Path("test_output.csv").unlink(missing_ok=True)

    def test_process_with_date_range_filter(self, sample_xml_file):
        """Test processing with specific date range."""
        from big_mood_detector.application.use_cases.process_health_data_use_case import MoodPredictionPipeline

        pipeline = MoodPredictionPipeline()

        # Process February 2024
        start_date = datetime(2024, 2, 1).date()
        end_date = datetime(2024, 2, 29).date()  # 2024 is a leap year

        # Process with date filter
        df = pipeline.process_health_export(
            export_path=sample_xml_file,
            output_path=Path("test_output.csv"),
            start_date=start_date,
            end_date=end_date
        )

        # Should have February data only
        assert len(df) <= 29
        assert len(df) >= 20  # Allow for some missing days

        # Check dates are in range
        if not df.empty:
            min_date = df.index.min()
            max_date = df.index.max()
            assert min_date >= start_date
            assert max_date <= end_date

        # Clean up
        Path("test_output.csv").unlink(missing_ok=True)

    def test_predict_with_date_filter(self, sample_xml_file):
        """Test prediction with date filtering."""
        from big_mood_detector.application.use_cases.process_health_data_use_case import MoodPredictionPipeline

        pipeline = MoodPredictionPipeline()

        # Predict for last 14 days
        end_date = datetime(2024, 3, 31).date()
        start_date = end_date - timedelta(days=14)

        # Run prediction with date filter
        result = pipeline.process_apple_health_file(
            file_path=sample_xml_file,
            start_date=start_date,
            end_date=end_date
        )

        # Should have predictions for filtered period
        assert len(result.daily_predictions) <= 14
        assert result.records_processed > 0

        # Check that dates are in range
        for pred_date in result.daily_predictions.keys():
            assert start_date <= pred_date <= end_date

    def test_no_filter_processes_all_data(self, sample_xml_file):
        """Test that no filter processes all available data."""
        from big_mood_detector.application.use_cases.process_health_data_use_case import MoodPredictionPipeline

        pipeline = MoodPredictionPipeline()

        # Process without date filter
        df = pipeline.process_health_export(
            export_path=sample_xml_file,
            output_path=Path("test_output.csv")
        )

        # Should have all 90 days
        assert len(df) >= 80  # Allow for some aggregation

        # Clean up
        Path("test_output.csv").unlink(missing_ok=True)

    def test_future_date_filter_returns_empty(self, sample_xml_file):
        """Test that filtering for future dates returns empty results."""
        from big_mood_detector.application.use_cases.process_health_data_use_case import MoodPredictionPipeline

        pipeline = MoodPredictionPipeline()

        # Try to process future dates
        start_date = datetime(2025, 1, 1).date()
        end_date = datetime(2025, 12, 31).date()

        # Process with future date filter
        df = pipeline.process_health_export(
            export_path=sample_xml_file,
            output_path=Path("test_output.csv"),
            start_date=start_date,
            end_date=end_date
        )

        # Should be empty
        assert len(df) == 0

        # Clean up
        Path("test_output.csv").unlink(missing_ok=True)
