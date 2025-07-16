#!/usr/bin/env python3
"""Debug XML parser to find why records aren't being detected."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import xml.etree.ElementTree as ET

from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser


def debug_parser():
    """Debug the XML parser step by step."""
    parser = StreamingXMLParser()
    xml_file = Path("apple_export/export.xml")

    print("DEBUGGING XML PARSER")
    print("=" * 60)

    # Check what types we're looking for
    print("\n1. CONFIGURED RECORD TYPES:")
    print(f"Sleep types: {['HKCategoryTypeIdentifierSleepAnalysis']}")
    print(f"Activity types: {parser.activity_parser.supported_activity_types}")
    print(f"Heart types: {parser.heart_parser.supported_heart_types}")

    # Now let's manually check the XML structure
    print("\n2. RAW XML STRUCTURE CHECK:")

    found_sleep = False
    found_activity = False
    record_count = 0

    for event, elem in ET.iterparse(str(xml_file), events=("end",)):
        if event == "end" and elem.tag == "Record":
            record_count += 1
            record_type = elem.get("type")

            # Check for sleep
            if (
                record_type == "HKCategoryTypeIdentifierSleepAnalysis"
                and not found_sleep
            ):
                found_sleep = True
                print(f"\nFound Sleep Record at position {record_count}:")
                print(f"  Type: {record_type}")
                print(f"  All attributes: {dict(elem.attrib)}")

            # Check for steps
            elif (
                record_type == "HKQuantityTypeIdentifierStepCount"
                and not found_activity
            ):
                found_activity = True
                print(f"\nFound Activity Record at position {record_count}:")
                print(f"  Type: {record_type}")
                print(f"  All attributes: {dict(elem.attrib)}")

            # Clear to save memory
            elem.clear()

            if found_sleep and found_activity:
                break

    print(f"\nScanned {record_count} records")

    # Now test the iter_records method with specific types
    print("\n3. TESTING iter_records WITH SPECIFIC TYPES:")

    # Test sleep records
    sleep_records = list(
        parser.iter_records(xml_file, ["HKCategoryTypeIdentifierSleepAnalysis"])
    )[:5]
    print(f"\nSleep records found: {len(sleep_records)}")
    if sleep_records:
        print("First sleep record:", sleep_records[0])

    # Test activity records
    activity_records = list(
        parser.iter_records(xml_file, ["HKQuantityTypeIdentifierStepCount"])
    )[:5]
    print(f"\nActivity records found: {len(activity_records)}")
    if activity_records:
        print("First activity record:", activity_records[0])


if __name__ == "__main__":
    debug_parser()
