#!/usr/bin/env python3
"""
Analyze record types in Apple Health XML export.

This helps us understand what data types are available in a raw export
so we can properly configure our parsers.
"""

import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime


def analyze_xml_export(file_path):
    """Analyze record types and dates in Apple Health export."""
    print(f"Analyzing {file_path}...")
    print("This may take a minute for large files...\n")

    record_types = Counter()
    date_ranges = {}
    sample_records = {}

    # Track specific categories we care about
    sleep_related = []
    activity_related = []
    heart_related = []

    try:
        for event, elem in ET.iterparse(file_path, events=("end",)):
            if event == "end" and elem.tag == "Record":
                record_type = elem.get("type", "Unknown")
                record_types[record_type] += 1

                # Track date ranges
                start_date = elem.get("startDate")
                if start_date:
                    date = datetime.fromisoformat(
                        start_date.replace("Z", "+00:00")
                    ).date()
                    if record_type not in date_ranges:
                        date_ranges[record_type] = {"min": date, "max": date}
                    else:
                        date_ranges[record_type]["min"] = min(
                            date_ranges[record_type]["min"], date
                        )
                        date_ranges[record_type]["max"] = max(
                            date_ranges[record_type]["max"], date
                        )

                # Categorize records
                if "Sleep" in record_type or "sleep" in record_type:
                    if record_type not in sleep_related:
                        sleep_related.append(record_type)
                        sample_records[record_type] = dict(elem.attrib)

                elif (
                    "Step" in record_type
                    or "Distance" in record_type
                    or "Exercise" in record_type
                    or "Activity" in record_type
                ):
                    if record_type not in activity_related:
                        activity_related.append(record_type)
                        sample_records[record_type] = dict(elem.attrib)

                elif "Heart" in record_type:
                    if record_type not in heart_related:
                        heart_related.append(record_type)
                        sample_records[record_type] = dict(elem.attrib)

                # Clear element to save memory
                elem.clear()

    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return

    # Print results
    print("=== RECORD TYPE SUMMARY ===")
    print(f"Total unique record types: {len(record_types)}")
    print(f"Total records: {sum(record_types.values())}\n")

    print("=== TOP 20 RECORD TYPES BY COUNT ===")
    for record_type, count in record_types.most_common(20):
        clean_type = record_type.replace("HKQuantityTypeIdentifier", "").replace(
            "HKCategoryTypeIdentifier", ""
        )
        print(f"{clean_type}: {count:,}")

    print("\n=== SLEEP-RELATED RECORDS ===")
    if sleep_related:
        for record_type in sleep_related:
            count = record_types[record_type]
            if record_type in date_ranges:
                dates = date_ranges[record_type]
                print(
                    f"{record_type}: {count:,} records ({dates['min']} to {dates['max']})"
                )
                if record_type in sample_records:
                    print(
                        f"  Sample attributes: {list(sample_records[record_type].keys())}"
                    )
    else:
        print("No sleep records found!")

    print("\n=== ACTIVITY-RELATED RECORDS ===")
    if activity_related:
        for record_type in activity_related[:10]:  # Limit to top 10
            count = record_types[record_type]
            if record_type in date_ranges:
                dates = date_ranges[record_type]
                print(
                    f"{record_type}: {count:,} records ({dates['min']} to {dates['max']})"
                )
    else:
        print("No activity records found!")

    print("\n=== HEART-RELATED RECORDS ===")
    if heart_related:
        for record_type in heart_related:
            count = record_types[record_type]
            if record_type in date_ranges:
                dates = date_ranges[record_type]
                print(
                    f"{record_type}: {count:,} records ({dates['min']} to {dates['max']})"
                )
    else:
        print("No heart records found!")

    # Save full analysis
    with open("xml_record_analysis.txt", "w") as f:
        f.write("COMPLETE RECORD TYPE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        for record_type, count in record_types.most_common():
            f.write(f"{record_type}: {count:,}\n")
            if record_type in date_ranges:
                dates = date_ranges[record_type]
                f.write(f"  Date range: {dates['min']} to {dates['max']}\n")
            if record_type in sample_records:
                f.write(f"  Attributes: {sample_records[record_type]}\n")
            f.write("\n")

    print("\nFull analysis saved to xml_record_analysis.txt")


if __name__ == "__main__":
    file_path = "apple_export/export.xml"
    analyze_xml_export(file_path)
