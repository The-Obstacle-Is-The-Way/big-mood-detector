#!/usr/bin/env python3
"""Test streaming XML parser with large Apple Health export."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
from datetime import datetime

from big_mood_detector.infrastructure.parsers.xml import StreamingXMLParser


def main():
    # Path to the large XML file
    xml_file = Path("apple_export/export.xml")

    if not xml_file.exists():
        print(f"Error: {xml_file} not found!")
        return

    print(f"Testing streaming XML parser with: {xml_file}")
    print(f"File size: {xml_file.stat().st_size / (1024**2):.1f} MB")

    # Initialize parser
    parser = StreamingXMLParser()

    # Track parsing progress
    start_time = time.time()
    record_counts = {"sleep": 0, "heart_rate": 0, "activity": 0, "other": 0}

    print("\nParsing XML file...")
    print("This may take a few minutes for a 520MB file...")

    # Parse and count records
    try:
        for i, entity in enumerate(parser.parse_file(str(xml_file))):
            # Determine record type based on actual attributes
            if hasattr(entity, "metric_type"):
                # It's a health metric
                metric_str = str(entity.metric_type)
                if "HEART_RATE" in metric_str:
                    record_counts["heart_rate"] += 1
                elif "SLEEP" in metric_str:
                    record_counts["sleep"] += 1
                else:
                    record_counts["other"] += 1
            elif hasattr(entity, "step_count"):
                record_counts["activity"] += 1
            elif hasattr(entity, "value") and hasattr(entity, "unit"):
                # Generic health record
                if entity.unit and "count/min" in str(entity.unit):
                    record_counts["heart_rate"] += 1
                else:
                    record_counts["other"] += 1
            else:
                record_counts["other"] += 1

            # Progress update every 10,000 records
            if (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Processed {i+1:,} records ({rate:.0f} records/sec)")

            # Sample first few records of each type
            if i < 5:
                print(f"\nSample record {i+1}:")
                if hasattr(entity, "__dict__"):
                    for key, value in entity.__dict__.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {entity}")

    except Exception as e:
        print(f"\nError during parsing: {e}")
        import traceback

        traceback.print_exc()
        return

    # Final statistics
    elapsed_time = time.time() - start_time
    total_records = sum(record_counts.values())

    print("\n" + "=" * 50)
    print("PARSING COMPLETE")
    print("=" * 50)
    print(f"\nTotal time: {elapsed_time:.1f} seconds")
    print(f"Total records: {total_records:,}")
    print(f"Processing rate: {total_records/elapsed_time:.0f} records/second")

    print("\nRecord breakdown:")
    for record_type, count in record_counts.items():
        percentage = (count / total_records * 100) if total_records > 0 else 0
        print(f"  {record_type}: {count:,} ({percentage:.1f}%)")

    # Memory usage estimate
    import psutil

    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"\nCurrent memory usage: {memory_mb:.1f} MB")

    # Save summary
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    summary = {
        "file": str(xml_file),
        "file_size_mb": xml_file.stat().st_size / (1024**2),
        "parsing_time_seconds": elapsed_time,
        "total_records": total_records,
        "records_per_second": total_records / elapsed_time if elapsed_time > 0 else 0,
        "memory_usage_mb": memory_mb,
        "record_counts": record_counts,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "xml_parsing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_dir / 'xml_parsing_summary.json'}")


if __name__ == "__main__":
    main()
