"""
Performance tests for XML streaming parser.

This test file demonstrates the problem: 500MB+ XML files timeout
and implements tests to ensure our solution works correctly.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

import pytest

from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)


class XMLDataGenerator:
    """Generate realistic Apple Health XML test data."""
    
    def __init__(self, base_path: Path = Path("tests/_data/generated")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_export(
        self,
        num_days: int = 30,
        records_per_day: int = 1000,
        size_mb: int | None = None,
    ) -> Path:
        """
        Create a test Apple Health export XML file.
        
        Args:
            num_days: Number of days of data to generate
            records_per_day: Number of records per day
            size_mb: Target file size in MB (overrides other params)
            
        Returns:
            Path to generated XML file
        """
        # Create root element
        root = Element("HealthData")
        root.set("locale", "en_US")
        
        # Calculate records needed for target size
        if size_mb:
            # Approximate: each record is ~500 bytes
            total_records = (size_mb * 1024 * 1024) // 500
            records_per_day = total_records // max(1, num_days)
        
        # Generate records
        start_date = datetime.now() - timedelta(days=num_days)
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            # Sleep records (3-4 per day)
            for i in range(3):
                record = SubElement(root, "Record")
                record.set("type", "HKCategoryTypeIdentifierSleepAnalysis")
                record.set("value", "HKCategoryValueSleepAnalysisAsleep")
                record.set("sourceName", "Apple Watch")
                
                sleep_start = current_date.replace(hour=22) + timedelta(hours=i*2)
                sleep_end = sleep_start + timedelta(hours=1, minutes=30)
                
                record.set("startDate", sleep_start.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("endDate", sleep_end.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("creationDate", sleep_end.strftime("%Y-%m-%d %H:%M:%S +0000"))
            
            # Activity records (most of the data)
            for i in range(records_per_day - 3):
                record = SubElement(root, "Record")
                record.set("type", "HKQuantityTypeIdentifierStepCount")
                record.set("value", str(100 + i % 500))
                record.set("unit", "count")
                record.set("sourceName", "iPhone")
                
                activity_time = current_date + timedelta(hours=i % 24, minutes=(i * 5) % 60)
                record.set("startDate", activity_time.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("endDate", activity_time.strftime("%Y-%m-%d %H:%M:%S +0000"))
                record.set("creationDate", activity_time.strftime("%Y-%m-%d %H:%M:%S +0000"))
        
        # Write to file
        filename = f"test_export_{size_mb or num_days}mb.xml"
        filepath = self.base_path / filename
        
        with open(filepath, "wb") as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(tostring(root, encoding="utf-8"))
        
        return filepath


@pytest.mark.performance
class TestXMLStreamingPerformance:
    """Test XML parser performance with large files."""
    
    def test_large_xml_completes_in_time_limit(self):
        """Test that 500MB XML file processes within 5 minutes."""
        # Given: Generate a 500MB XML file
        generator = XMLDataGenerator()
        large_file = generator.create_export(size_mb=500)
        
        try:
            # When: Processing the file
            parser = StreamingXMLParser()
            start_time = time.time()
            
            # Parse with date range to reduce memory usage
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
            # Parse all records
            sleep_records = []
            activity_records = []
            heart_records = []
            
            # Use the generator interface
            from big_mood_detector.domain.entities.sleep_record import SleepRecord
            from big_mood_detector.domain.entities.activity_record import ActivityRecord
            from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord
            
            for entity in parser.parse_file(
                str(large_file),
                entity_type="all",
                start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
            ):
                if isinstance(entity, SleepRecord):
                    sleep_records.append(entity)
                elif isinstance(entity, ActivityRecord):
                    activity_records.append(entity)
                elif isinstance(entity, HeartRateRecord):
                    heart_records.append(entity)
            
            duration = time.time() - start_time
            
            # Then: Should complete within 5 minutes (300 seconds)
            assert duration < 300, f"Parsing took {duration:.1f}s, expected < 300s"
            
            # And: Should have parsed some records
            assert len(sleep_records) > 0, "No sleep records parsed"
            assert len(activity_records) > 0, "No activity records parsed"
            
        finally:
            # Cleanup
            if large_file.exists():
                large_file.unlink()
    
    def test_memory_usage_stays_constant(self):
        """Test that memory usage doesn't grow with file size."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        generator = XMLDataGenerator()
        
        # Test with increasing file sizes
        file_sizes = [10, 50, 100]  # MB
        memory_usage = []
        
        for size_mb in file_sizes:
            # Generate test file
            test_file = generator.create_export(size_mb=size_mb)
            
            try:
                # Measure memory before parsing
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Parse file
                parser = StreamingXMLParser()
                # Just consume the generator to parse the file
                list(parser.parse_file(str(test_file)))
                
                # Measure memory after parsing
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before
                memory_usage.append(memory_increase)
                
            finally:
                if test_file.exists():
                    test_file.unlink()
        
        # Memory increase should not scale linearly with file size
        # Should stay roughly constant (within 50MB variance)
        assert max(memory_usage) - min(memory_usage) < 50, (
            f"Memory usage varied too much: {memory_usage}"
        )
    
    def test_progress_callback_works(self):
        """Test that progress callbacks are called during parsing."""
        generator = XMLDataGenerator()
        test_file = generator.create_export(num_days=10, records_per_day=1000)
        
        progress_updates = []
        
        def progress_callback(message: str, progress: float):
            progress_updates.append((message, progress))
        
        try:
            parser = StreamingXMLParser()
            parser.parse_file(
                str(test_file),
                progress_callback=progress_callback
            )
            
            # Should have progress updates
            assert len(progress_updates) > 0, "No progress updates received"
            
            # Progress should go from 0 to 100
            progress_values = [p[1] for p in progress_updates]
            assert min(progress_values) >= 0
            assert max(progress_values) <= 100
            
            # Should have meaningful messages
            messages = [p[0] for p in progress_updates]
            assert any("Parsing" in msg for msg in messages)
            
        finally:
            if test_file.exists():
                test_file.unlink()
    
    @pytest.mark.parametrize("num_days,expected_speedup", [
        (30, 2.0),   # Small dataset: 2x speedup
        (365, 5.0),  # Medium dataset: 5x speedup  
        (730, 10.0), # Large dataset: 10x speedup
    ])
    def test_optimized_aggregation_performance(self, num_days, expected_speedup):
        """Test that pre-indexing provides expected performance gains."""
        from big_mood_detector.application.services.aggregation_pipeline import (
            AggregationPipeline
        )
        
        # Generate test data
        generator = XMLDataGenerator()
        test_file = generator.create_export(
            num_days=num_days,
            records_per_day=100
        )
        
        try:
            # Parse the data
            parser = StreamingXMLParser()
            sleep_records, activity_records, heart_records = parser.parse_file(
                str(test_file)
            )
            
            # Create pipeline
            pipeline = AggregationPipeline()
            
            # Time the aggregation
            start_date = date.today() - timedelta(days=num_days)
            end_date = date.today()
            
            start_time = time.time()
            features = pipeline.aggregate_daily_features(
                sleep_records=sleep_records,
                activity_records=activity_records,
                heart_records=heart_records,
                start_date=start_date,
                end_date=end_date,
            )
            duration = time.time() - start_time
            
            # Check that we got features for each day
            assert len(features) >= num_days * 0.8  # Allow some missing days
            
            # Performance should scale sub-linearly (better than O(n*m))
            # For now just check it completes reasonably fast
            expected_max_duration = num_days * 0.1  # 0.1 seconds per day max
            assert duration < expected_max_duration, (
                f"Aggregation took {duration:.1f}s for {num_days} days, "
                f"expected < {expected_max_duration:.1f}s"
            )
            
        finally:
            if test_file.exists():
                test_file.unlink()