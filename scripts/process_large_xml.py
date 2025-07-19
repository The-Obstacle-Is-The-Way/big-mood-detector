#!/usr/bin/env python3
"""
Process Large XML Files Efficiently

This script helps process very large Apple Health XML exports by:
1. Providing progress updates
2. Offering date range extraction
3. Suggesting optimal processing strategies
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.parsers.xml.fast_streaming_parser import (
    FastStreamingXMLParser,
)


@click.command()
@click.argument("xml_file", type=click.Path(exists=True, path_type=Path))
@click.option("--count-only", is_flag=True, help="Just count records without processing")
@click.option("--start-date", type=str, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", type=str, help="End date (YYYY-MM-DD)")
@click.option("--extract-to", type=click.Path(path_type=Path), help="Extract matching records to new XML file")
def process_large_xml(
    xml_file: Path,
    count_only: bool,
    start_date: str | None,
    end_date: str | None,
    extract_to: Path | None,
):
    """Process large Apple Health XML exports efficiently."""
    file_size_mb = xml_file.stat().st_size / (1024 * 1024)
    click.echo(f"\nProcessing: {xml_file.name} ({file_size_mb:.1f} MB)")
    click.echo("=" * 60)
    
    parser = FastStreamingXMLParser()
    
    # Just count records
    if count_only:
        click.echo("\nCounting records...")
        start_time = time.time()
        
        counts = parser.count_records_by_date(xml_file, start_date, end_date)
        
        elapsed = time.time() - start_time
        click.echo(f"\nCompleted in {elapsed:.1f} seconds")
        click.echo(f"Processing rate: {int(counts['total']/elapsed):,} records/second")
        
        click.echo("\nRecord counts:")
        for record_type, count in counts.items():
            click.echo(f"  {record_type}: {count:,}")
        
        if counts['total'] > 0:
            # Estimate processing time
            est_process_time = counts['total'] / (counts['total']/elapsed) * 3  # 3x for full processing
            click.echo(f"\nEstimated full processing time: {int(est_process_time/60)} minutes")
            
            if est_process_time > 300:  # > 5 minutes
                click.echo("\n⚠️  RECOMMENDATIONS:")
                click.echo("1. Use date filtering to reduce processing time")
                click.echo("2. Consider using JSON export from Health Auto Export app instead")
                click.echo("3. Process in smaller date ranges and combine results")
        
        return
    
    # Extract to new file
    if extract_to:
        click.echo(f"\nExtracting records to: {extract_to}")
        click.echo("This feature is not yet implemented")
        # TODO: Implement XML extraction with date filtering
        return
    
    # Full processing with progress
    click.echo("\nProcessing records...")
    
    record_types = ["sleep", "activity", "heart"]
    all_records = {"sleep": [], "activity": [], "heart": []}
    
    for record_type in record_types:
        click.echo(f"\nParsing {record_type} records...")
        start_time = time.time()
        count = 0
        
        for entity in parser.parse_file(
            xml_file,
            entity_type=record_type,
            start_date=start_date,
            end_date=end_date
        ):
            all_records[record_type].append(entity)
            count += 1
            
            if count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed
                click.echo(f"  {count:,} records ({int(rate):,}/sec)", nl=False)
                click.echo("\r", nl=False)
        
        elapsed = time.time() - start_time
        click.echo(f"  {count:,} {record_type} records in {elapsed:.1f}s")
    
    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("PROCESSING COMPLETE")
    click.echo("=" * 60)
    
    total_records = sum(len(records) for records in all_records.values())
    click.echo(f"Total records processed: {total_records:,}")
    
    for record_type, records in all_records.items():
        if records:
            click.echo(f"\n{record_type.capitalize()} records: {len(records):,}")
            
            # Show date range
            if record_type == "sleep":
                dates = [r.start_date.date() for r in records]
            elif record_type == "activity":
                dates = [r.start_date.date() for r in records]
            else:  # heart
                dates = [r.timestamp.date() for r in records]
            
            if dates:
                click.echo(f"  Date range: {min(dates)} to {max(dates)}")
    
    # Save to CSV option
    if click.confirm("\nSave processed data to CSV?"):
        output_dir = xml_file.parent / "processed"
        output_dir.mkdir(exist_ok=True)
        
        # TODO: Implement CSV export
        click.echo("CSV export not yet implemented")


if __name__ == "__main__":
    process_large_xml()