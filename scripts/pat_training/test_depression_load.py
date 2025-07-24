#!/usr/bin/env python3
"""Test depression data loading."""

from pathlib import Path

from big_mood_detector.infrastructure.fine_tuning.nhanes_processor import (
    NHANESProcessor,
)

# Test loading depression scores
processor = NHANESProcessor(data_dir=Path("data/nhanes/2013-2014"))
depression_df = processor.load_depression_scores("DPQ_H.xpt")

print(f"Depression data shape: {depression_df.shape}")
print(f"Columns: {list(depression_df.columns)}")
print("\nFirst few rows of key columns:")
print(depression_df[['SEQN', 'PHQ9_total', 'depressed']].head(10))

# Check distribution
print("\nDepression distribution:")
print(f"Total subjects: {len(depression_df)}")
print(f"Depressed (PHQ9 >= 10): {depression_df['depressed'].sum()}")
print(f"Not depressed: {(1 - depression_df['depressed']).sum()}")
