#!/usr/bin/env python3
"""Quick test to check NHANES files are readable."""

import time
from pathlib import Path

import pandas as pd

data_dir = Path("data/nhanes/2013-2014")

print("Testing NHANES file loading...")

# Test each file
files = {
    "PAXMIN_H.xpt": "Minute-level actigraphy",
    "PAXDAY_H.xpt": "Day-level assignments",
    "DPQ_H.xpt": "Depression (PHQ-9) scores"
}

for filename, description in files.items():
    filepath = data_dir / filename
    print(f"\n{filename} ({description}):")
    print(f"  Path: {filepath}")
    print(f"  Exists: {filepath.exists()}")

    if filepath.exists():
        print(f"  Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

        # Try to load first few rows
        try:
            start = time.time()
            df = pd.read_sas(filepath, chunksize=1000)
            first_chunk = next(df)
            elapsed = time.time() - start

            print(f"  Loaded first 1000 rows in {elapsed:.1f}s")
            print(f"  Columns: {list(first_chunk.columns)[:5]}... ({len(first_chunk.columns)} total)")
            print(f"  First subject ID: {first_chunk['SEQN'].iloc[0]}")
        except Exception as e:
            print(f"  ERROR loading: {e}")

print("\nIf PAXMIN_H takes >5s for 1000 rows, the full load will be VERY slow!")
print("The full file has millions of rows...")
