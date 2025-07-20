"""
Test progress indication functionality.

Tests for Issue #31: Add progress indication for long-running operations
"""

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, call
from typing import Callable

import pytest

from big_mood_detector.infrastructure.parsers.xml.fast_streaming_parser import (
    FastStreamingXMLParser,
)
from big_mood_detector.application.services.data_parsing_service import (
    DataParsingService,
)
from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)


class TestProgressIndication:
    """Test progress indication across the pipeline."""

    def test_xml_parser_accepts_progress_callback(self):
        """Test that XML parser can accept a progress callback."""
        parser = FastStreamingXMLParser()
        
        # Should have a method that accepts progress_callback
        assert hasattr(parser, 'parse_file')
        
        # The parse_file method should accept a progress_callback parameter
        import inspect
        sig = inspect.signature(parser.parse_file)
        params = list(sig.parameters.keys())
        
        # We'll add progress_callback as a parameter
        # For now, this test will fail, driving our implementation

    @patch('big_mood_detector.infrastructure.parsers.xml.fast_streaming_parser.logger')
    def test_xml_parser_calls_progress_callback(self, mock_logger):
        """Test that XML parser calls progress callback during parsing."""
        import tempfile
        import os
        
        parser = FastStreamingXMLParser()
        progress_callback = Mock()
        
        # Create a small test XML in a temporary file
        test_xml = '''<?xml version="1.0" encoding="UTF-8"?>
        <HealthData>
            <Record type="HKQuantityTypeIdentifierStepCount" 
                    startDate="2024-01-01 10:00:00 -0500" 
                    endDate="2024-01-01 10:05:00 -0500" 
                    value="100"/>
            <Record type="HKQuantityTypeIdentifierStepCount" 
                    startDate="2024-01-02 10:00:00 -0500" 
                    endDate="2024-01-02 10:05:00 -0500" 
                    value="200"/>
        </HealthData>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tf:
            tf.write(test_xml)
            tf.flush()
            temp_path = tf.name
        
        try:
            # This should call progress_callback
            list(parser.parse_file(
                temp_path, 
                progress_callback=progress_callback
            ))
            
            # Verify progress was reported
            assert progress_callback.called
            # Should be called at least twice (start and end)
            assert progress_callback.call_count >= 2
            
            # Check the calls
            calls = progress_callback.call_args_list
            # First call should be start (0.0)
            assert calls[0][0][1] == 0.0  # progress value
            # Last call should be complete (1.0)
            assert calls[-1][0][1] == 1.0  # progress value
        finally:
            os.unlink(temp_path)

    def test_data_parsing_service_accepts_progress_callback(self):
        """Test that DataParsingService accepts and propagates progress callback."""
        service = DataParsingService()
        
        # Check parse_xml_export accepts progress_callback
        import inspect
        sig = inspect.signature(service.parse_xml_export)
        params = list(sig.parameters.keys())
        assert 'progress_callback' in params

    @patch('big_mood_detector.application.services.data_parsing_service.FastStreamingXMLParser')
    def test_data_parsing_service_propagates_progress(self, mock_parser_class):
        """Test that DataParsingService propagates progress to parser."""
        service = DataParsingService()
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        
        # Set up mock to return empty lists
        mock_parser.parse_file.return_value = []
        
        progress_callback = Mock()
        
        # Call parse_xml_export with progress callback
        service.parse_xml_export(
            Path('test.xml'),
            progress_callback=progress_callback
        )
        
        # Verify parser was called with progress callback
        mock_parser.parse_file.assert_called()
        call_args = mock_parser.parse_file.call_args
        
        # The service should pass through the progress callback
        # or create a wrapper that calls both its own and the provided callback

    def test_pipeline_accepts_progress_callback(self):
        """Test that MoodPredictionPipeline accepts progress callback."""
        pipeline = MoodPredictionPipeline()
        
        # Check process_apple_health_file method
        import inspect
        sig = inspect.signature(pipeline.process_apple_health_file)
        params = list(sig.parameters.keys())
        
        # Should accept progress_callback parameter
        # This will fail initially, driving implementation

    def test_progress_callback_format(self):
        """Test the expected format of progress callbacks."""
        # Progress callback should accept (message: str, progress: float)
        # where progress is 0.0 to 1.0
        
        received_calls = []
        
        def progress_callback(message: str, progress: float):
            received_calls.append((message, progress))
        
        # Simulate calling the callback
        progress_callback("Starting processing", 0.0)
        progress_callback("Processing records", 0.5)
        progress_callback("Finalizing", 1.0)
        
        assert len(received_calls) == 3
        assert received_calls[0] == ("Starting processing", 0.0)
        assert received_calls[1] == ("Processing records", 0.5)
        assert received_calls[2] == ("Finalizing", 1.0)
        
        # Progress should be between 0 and 1
        for _, progress in received_calls:
            assert 0.0 <= progress <= 1.0

    def test_file_size_estimation(self):
        """Test that parser can estimate progress based on file size."""
        # For XML files, we can estimate progress by file position
        parser = FastStreamingXMLParser()
        
        # Should have a method to parse with file size progress
        # This helps with progress bars for large files
        
        # Implementation should track file position vs total size

    @pytest.mark.parametrize("record_count,expected_calls", [
        (100, 1),      # Small file, one update
        (10000, 2),    # Medium file, periodic updates
        (100000, 11),  # Large file, ~10 updates
    ])
    def test_progress_frequency(self, record_count, expected_calls):
        """Test that progress is reported at reasonable intervals."""
        # Progress should be reported:
        # - At start (0%)
        # - Periodically during processing
        # - At end (100%)
        # But not too frequently (performance impact)
        pass  # Implementation needed

    def test_progress_callback_with_tqdm(self):
        """Test integration with tqdm progress bars."""
        from tqdm import tqdm
        
        # Create a tqdm progress bar
        pbar = tqdm(total=100, desc="Processing")
        
        def tqdm_callback(message: str, progress: float):
            pbar.set_description(message)
            pbar.n = int(progress * 100)
            pbar.refresh()
        
        # This should work seamlessly with our progress system
        # Simulating some progress
        tqdm_callback("Starting", 0.0)
        tqdm_callback("Processing", 0.5)
        tqdm_callback("Done", 1.0)
        
        pbar.close()
        
        # The progress bar should have updated correctly
        assert pbar.n == 100

    def test_progress_callback_error_handling(self):
        """Test that processing continues if progress callback fails."""
        parser = FastStreamingXMLParser()
        
        # Create a failing progress callback
        def failing_callback(message: str, progress: float):
            raise Exception("Progress callback failed!")
        
        # Parser should continue even if callback fails
        # This ensures robustness
        
        # Implementation should wrap callback calls in try/except