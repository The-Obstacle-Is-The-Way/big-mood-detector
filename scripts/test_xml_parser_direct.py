#!/usr/bin/env python3
"""Test XML parser directly to debug why it's not finding records."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser
from datetime import datetime

def test_xml_parser():
    """Test the XML parser with actual export.xml."""
    parser = StreamingXMLParser()
    xml_file = Path("apple_export/export.xml")
    
    print(f"Testing XML parser with: {xml_file}")
    print("="*60)
    
    # Track what we find
    sleep_count = 0
    activity_count = 0
    heart_count = 0
    
    # Sample some records
    sleep_samples = []
    activity_samples = []
    
    try:
        # First, let's see what parse_file returns
        print("\n1. Testing parse_file() method:")
        for i, entity in enumerate(parser.parse_file(xml_file, entity_type="all")):
            if i < 5:  # Show first 5 entities
                print(f"\nEntity {i+1}: {type(entity).__name__}")
                if hasattr(entity, '__dict__'):
                    for key, value in entity.__dict__.items():
                        print(f"  {key}: {value}")
            
            # Count by type
            entity_type = type(entity).__name__
            if entity_type == "SleepRecord":
                sleep_count += 1
                if len(sleep_samples) < 3:
                    sleep_samples.append(entity)
            elif entity_type == "ActivityRecord":
                activity_count += 1
                if len(activity_samples) < 3:
                    activity_samples.append(entity)
            elif entity_type == "HeartRateRecord":
                heart_count += 1
            
            if i >= 1000:  # Limit for testing
                print("\n...stopping at 1000 records for testing")
                break
    
    except Exception as e:
        print(f"\nError during parsing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"Sleep records found: {sleep_count}")
    print(f"Activity records found: {activity_count}")
    print(f"Heart rate records found: {heart_count}")
    
    # Now let's test the raw record iterator to see what's happening
    print("\n" + "="*60)
    print("2. Testing iter_records() to see raw data:")
    
    sleep_raw_count = 0
    activity_raw_count = 0
    
    for i, record in enumerate(parser.iter_records(xml_file)):
        record_type = record.get('type', '')
        
        if 'Sleep' in record_type:
            sleep_raw_count += 1
            if sleep_raw_count <= 2:
                print(f"\nRaw Sleep Record {sleep_raw_count}:")
                for key, value in record.items():
                    print(f"  {key}: {value}")
        
        elif 'StepCount' in record_type:
            activity_raw_count += 1
            if activity_raw_count <= 2:
                print(f"\nRaw Activity Record {activity_raw_count}:")
                for key, value in record.items():
                    print(f"  {key}: {value}")
        
        if i >= 10000:  # Check more records
            break
    
    print(f"\nRaw sleep records found: {sleep_raw_count}")
    print(f"Raw activity records found: {activity_raw_count}")
    
    # If we found sleep samples, show their date range
    if sleep_samples:
        print("\n" + "="*60)
        print("SLEEP RECORD SAMPLES:")
        for i, sleep in enumerate(sleep_samples):
            print(f"\nSleep {i+1}:")
            print(f"  Start: {sleep.start_date}")
            print(f"  End: {sleep.end_date}")
            print(f"  State: {sleep.state}")
            print(f"  Source: {sleep.source_name}")

if __name__ == "__main__":
    test_xml_parser()