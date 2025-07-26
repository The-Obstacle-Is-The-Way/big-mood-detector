#!/usr/bin/env python3
"""Extract a week of real Apple Health data for testing."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Target dates for extraction
END_DATE = datetime(2024, 8, 1)
START_DATE = END_DATE - timedelta(days=7)

def main():
    source_file = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/data/input/apple_export/export.xml")
    output_file = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/tests/fixtures/health/week_sample.xml")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(source_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # Copy header
        in_header = True
        for line in f_in:
            if in_header:
                f_out.write(line)
                if '<HealthData' in line:
                    in_header = False
                continue
            
            # Check if this is a record
            if '<Record' in line:
                # Extract date from line
                if 'startDate="' in line:
                    date_start = line.find('startDate="') + 11
                    date_str = line[date_start:date_start+19]  # YYYY-MM-DD HH:MM:SS
                    try:
                        record_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        if START_DATE <= record_date <= END_DATE:
                            f_out.write(line)
                    except:
                        pass
            elif '</HealthData>' in line:
                f_out.write(line)
                break
    
    print(f"Extracted test data to {output_file}")

if __name__ == "__main__":
    main()