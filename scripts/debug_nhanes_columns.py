#!/usr/bin/env python3
"""Debug script to check NHANES file columns."""

from pathlib import Path

import pandas as pd

# Load both actigraphy files to see their structure
nhanes_dir = Path("data/nhanes/2013-2014")

print("Checking PAXHD_H.xpt columns:")
paxhd = pd.read_sas(nhanes_dir / "PAXHD_H.xpt")
print(f"Shape: {paxhd.shape}")
print(f"Columns: {list(paxhd.columns)}")
print("\nFirst few rows:")
print(paxhd.head())

print("\n" + "="*60 + "\n")

print("Checking PAXMIN_H.xpt columns:")
paxmin = pd.read_sas(nhanes_dir / "PAXMIN_H.xpt")
print(f"Shape: {paxmin.shape}")
print(f"Columns: {list(paxmin.columns)[:20]}...")  # First 20 columns
print("\nFirst few rows:")
print(paxmin.head())

# Check if PAXDAY is in either file
print(f"\n'PAXDAY' in PAXHD: {'PAXDAY' in paxhd.columns}")
print(f"'PAXDAY' in PAXMIN: {'PAXDAY' in paxmin.columns}")

# Check depression data structure
print("\n" + "="*60 + "\n")
print("Checking DPQ_H.xpt columns:")
dpq = pd.read_sas(nhanes_dir / "DPQ_H.xpt")
print(f"Shape: {dpq.shape}")
print(f"Columns: {list(dpq.columns)}")
print("\nFirst few rows:")
print(dpq.head())
