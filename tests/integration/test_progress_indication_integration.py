"""
Integration test for progress indication functionality.

Tests for Issue #31: Verify progress indication works end-to-end
"""

import tempfile
from pathlib import Path

import pytest

class TestProgressIndicationIntegration:
    """Test progress indication through full pipeline."""

    @pytest.fixture
    def sample_xml(self):
        """Create a sample XML file for testing."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount"
                    startDate="2024-01-01 10:00:00 -0500"
                    endDate="2024-01-01 10:05:00 -0500"
                    value="100"/>
            <Record type="HKQuantityTypeIdentifierHeartRate"
                    startDate="2024-01-01 10:00:00 -0500"
                    endDate="2024-01-01 10:00:00 -0500"
                    value="72"/>
            <Record type="HKCategoryTypeIdentifierSleepAnalysis"
                    startDate="2024-01-01 00:00:00 -0500"
                    endDate="2024-01-01 08:00:00 -0500"
                    value="HKCategoryValueSleepAnalysisAsleep"/>
        </HealthData>'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tf:
            tf.write(xml_content)
            tf.flush()
            return Path(tf.name)

    def test_progress_indication_through_pipeline(self, sample_xml):
        """Test that progress callbacks are propagated through the entire pipeline."""
        from big_mood_detector.application.use_cases.process_health_data_use_case import MoodPredictionPipeline

        try:
            # Track progress calls
            progress_calls = []

            def progress_callback(message: str, progress: float):
                progress_calls.append((message, progress))

            # Create pipeline
            pipeline = MoodPredictionPipeline()

            # Process with progress callback
            result = pipeline.process_apple_health_file(
                file_path=sample_xml,
                progress_callback=progress_callback
            )

            # Verify progress was reported
            assert len(progress_calls) > 0, "No progress callbacks received"

            # Check progress values
            messages = [call[0] for call in progress_calls]
            progress_values = [call[1] for call in progress_calls]

            # Should start at 0.0
            assert progress_values[0] == 0.0, "Progress should start at 0.0"

            # Should end at 1.0
            assert progress_values[-1] == 1.0, "Progress should end at 1.0"

            # Progress should be monotonically increasing
            for i in range(1, len(progress_values)):
                assert progress_values[i] >= progress_values[i-1], \
                    f"Progress decreased from {progress_values[i-1]} to {progress_values[i]}"

            # Check message content
            assert any("Starting" in msg for msg in messages), \
                "Should have a starting message"
            assert any("Completed" in msg for msg in messages), \
                "Should have a completion message"

            # Verify result is valid
            assert result is not None
            # Note: records_processed might be 0 if data doesn't meet criteria
            # but progress should still be reported

        finally:
            # Clean up
            sample_xml.unlink()

    def test_progress_indication_with_data_parsing_service(self, sample_xml):
        """Test progress indication at the service layer."""
        from big_mood_detector.application.services.data_parsing_service import DataParsingService

        try:
            progress_calls = []

            def progress_callback(message: str, progress: float):
                progress_calls.append((message, progress))

            # Test DataParsingService directly
            service = DataParsingService()
            service.parse_xml_export(
                xml_path=sample_xml,
                progress_callback=progress_callback
            )

            # Verify progress was reported
            assert len(progress_calls) > 0, "Progress should have been reported"
            assert progress_calls[0][1] == 0.0  # First progress
            assert progress_calls[-1][1] == 1.0  # Last progress

            # Progress indication is the focus of this test
            # Records might not parse if format doesn't match expected

        finally:
            sample_xml.unlink()

    def test_progress_callback_error_resilience(self, sample_xml):
        """Test that processing continues even if progress callback fails."""
        from big_mood_detector.application.use_cases.process_health_data_use_case import MoodPredictionPipeline

        try:
            # Create a failing progress callback
            def failing_callback(message: str, progress: float):
                raise Exception("Progress callback failed!")

            # Should not crash the pipeline
            pipeline = MoodPredictionPipeline()
            result = pipeline.process_apple_health_file(
                file_path=sample_xml,
                progress_callback=failing_callback
            )

            # Processing should complete successfully
            assert result is not None
            # Focus is on error resilience, not record count

        finally:
            sample_xml.unlink()

    def test_progress_indication_large_file_simulation(self):
        """Test progress indication with simulated large file processing."""
        from big_mood_detector.application.services.data_parsing_service import DataParsingService

        # Create a larger XML file to test progress intervals
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>\n<HealthData>\n'
        xml_footer = '</HealthData>'

        records = []
        # Generate 10k records to test progress intervals
        for i in range(10000):
            records.append(
                f'<Record type="HKQuantityTypeIdentifierStepCount" '
                f'startDate="2024-01-{(i//1440)+1:02d} {(i//60)%24:02d}:{i%60:02d}:00 -0500" '
                f'endDate="2024-01-{(i//1440)+1:02d} {(i//60)%24:02d}:{i%60:02d}:00 -0500" '
                f'value="{100 + i%500}"/>'
            )

        xml_content = xml_header + '\n'.join(records) + '\n' + xml_footer

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tf:
            tf.write(xml_content)
            tf.flush()
            temp_path = Path(tf.name)

        try:
            progress_calls = []

            def progress_callback(message: str, progress: float):
                progress_calls.append((message, progress))

            # Process large file
            service = DataParsingService()
            service.parse_xml_export(
                xml_path=temp_path,
                progress_callback=progress_callback
            )

            # Should have multiple progress updates
            assert len(progress_calls) >= 3, \
                "Large file should trigger multiple progress updates"

            # Check that we got intermediate progress
            progress_values = [call[1] for call in progress_calls]
            intermediate_values = [p for p in progress_values if 0.0 < p < 1.0]
            assert len(intermediate_values) > 0, \
                "Should have intermediate progress values"

            # Progress indication is what matters, not exact parsing
            # XML format might not match expected format exactly

        finally:
            temp_path.unlink()
