#!/usr/bin/env python3
"""Extract a clean week of Apple Health data for testing."""

import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

# Target dates for extraction
END_DATE = datetime(2024, 7, 31)
START_DATE = END_DATE - timedelta(days=7)

def main():
    source_file = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/data/input/apple_export/export.xml")
    output_file = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/tests/fixtures/health/week_sample_clean.xml")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse and filter
    print("Parsing large XML file...")
    records = []
    
    with open(source_file, 'r', encoding='utf-8') as f:
        # Read header
        header_lines = []
        for line in f:
            if '<HealthData' in line:
                header_lines.append(line)
                break
            header_lines.append(line)
        
        # Process records one by one
        record_count = 0
        for line in f:
            if '<Record' in line and line.strip().endswith('/>'):
                # Self-closing record tag
                if 'startDate="' in line:
                    date_start = line.find('startDate="') + 11
                    date_str = line[date_start:date_start+19]
                    try:
                        record_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        if START_DATE <= record_date <= END_DATE:
                            records.append(line)
                            record_count += 1
                            if record_count % 100 == 0:
                                print(f"Found {record_count} records...")
                    except:
                        pass
            elif '</HealthData>' in line:
                break
    
    # Write clean output
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in header_lines:
            f.write(line)
        
        for record in records:
            f.write(record)
        
        f.write('</HealthData>\n')
    
    print(f"Extracted {len(records)} records to {output_file}")

if __name__ == "__main__":
    main()