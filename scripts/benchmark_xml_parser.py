#!/usr/bin/env python3
"""
Benchmark XML parsing performance: stdlib vs lxml

Tests the performance improvement from using lxml for large XML files.
"""

import time
from pathlib import Path

from big_mood_detector.infrastructure.parsers.xml.fast_streaming_parser import (
    FastStreamingXMLParser,
)

# Test both parsers
from big_mood_detector.infrastructure.parsers.xml.streaming_adapter import (
    StreamingXMLParser,
)


def benchmark_parser(parser_class, xml_file: Path, entity_type: str = "all") -> tuple[int, float]:
    """Benchmark a parser and return count and time."""
    parser = parser_class()

    start_time = time.time()
    count = 0

    for _entity in parser.parse_file(xml_file, entity_type=entity_type):
        count += 1
        if count % 10000 == 0:
            print(f"  {parser_class.__name__}: {count:,} records...", end='\r')

    elapsed = time.time() - start_time
    print(f"  {parser_class.__name__}: {count:,} records in {elapsed:.2f}s")

    return count, elapsed


def benchmark_with_date_filter(parser_class, xml_file: Path, start_date: str, end_date: str) -> tuple[int, float]:
    """Benchmark a parser with date filtering."""
    parser = parser_class()

    start_time = time.time()
    count = 0

    for _entity in parser.parse_file(xml_file, entity_type="all", start_date=start_date, end_date=end_date):
        count += 1
        if count % 1000 == 0:
            print(f"  {parser_class.__name__} (filtered): {count:,} records...", end='\r')

    elapsed = time.time() - start_time
    print(f"  {parser_class.__name__} (filtered): {count:,} records in {elapsed:.2f}s")

    return count, elapsed


def main():
    # Find XML file
    xml_file = Path("data/input/apple_export/export.xml")
    if not xml_file.exists():
        print(f"XML file not found: {xml_file}")
        return

    file_size_mb = xml_file.stat().st_size / (1024 * 1024)
    print(f"\nBenchmarking XML parsers on {xml_file.name} ({file_size_mb:.1f} MB)")
    print("=" * 60)

    # Test 1: Parse all records
    print("\nTest 1: Parse all records")
    print("-" * 40)

    stdlib_count, stdlib_time = benchmark_parser(StreamingXMLParser, xml_file)
    lxml_count, lxml_time = benchmark_parser(FastStreamingXMLParser, xml_file)

    speedup = stdlib_time / lxml_time if lxml_time > 0 else 0
    print(f"\nSpeedup: {speedup:.1f}x faster with lxml")
    print(f"Records/second: {int(stdlib_count/stdlib_time):,} (stdlib) vs {int(lxml_count/lxml_time):,} (lxml)")

    # Test 2: Parse with date filtering (1 month)
    print("\n\nTest 2: Parse with date filtering (June 2025)")
    print("-" * 40)

    start_date = "2025-06-01"
    end_date = "2025-07-01"

    stdlib_filtered_count, stdlib_filtered_time = benchmark_with_date_filter(
        StreamingXMLParser, xml_file, start_date, end_date
    )
    lxml_filtered_count, lxml_filtered_time = benchmark_with_date_filter(
        FastStreamingXMLParser, xml_file, start_date, end_date
    )

    filtered_speedup = stdlib_filtered_time / lxml_filtered_time if lxml_filtered_time > 0 else 0
    print(f"\nFiltered speedup: {filtered_speedup:.1f}x faster with lxml")
    print(f"Date filtering efficiency: {lxml_filtered_count:,} records from {lxml_count:,} total")

    # Test 3: Count records quickly
    print("\n\nTest 3: Quick record count by type")
    print("-" * 40)

    fast_parser = FastStreamingXMLParser()
    start_time = time.time()
    counts = fast_parser.count_records_by_date(xml_file, start_date, end_date)
    count_time = time.time() - start_time

    print(f"Counted records in {count_time:.2f}s:")
    for record_type, count in counts.items():
        print(f"  {record_type}: {count:,}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Total records: {lxml_count:,}")
    print(f"Performance gain: {speedup:.1f}x faster")
    print(f"Processing rate: {int(lxml_count/lxml_time):,} records/second")
    print(f"Time to process full file: {lxml_time:.1f}s (vs {stdlib_time:.1f}s)")


if __name__ == "__main__":
    main()
