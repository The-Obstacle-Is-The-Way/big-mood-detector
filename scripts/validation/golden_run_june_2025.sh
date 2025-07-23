#!/usr/bin/env bash
set -euo pipefail

# Golden run script for June 2025 data
# This tests the full pipeline with real data and validates outputs

echo "========================================="
echo "BIG MOOD DETECTOR - GOLDEN RUN JUNE 2025"
echo "========================================="

INPUT=data/input/health_auto_export
OUT_DIR=data/output/golden_june_2025

# Create output directory
mkdir -p "$OUT_DIR"

echo ""
echo "1. Processing features for June 2025..."
python3 src/big_mood_detector/main.py process \
  "$INPUT" \
  --start-date 2025-06-01 \
  --end-date 2025-06-30 \
  -o "$OUT_DIR/features.csv"

echo ""
echo "2. Generating predictions with report..."
python3 src/big_mood_detector/main.py predict \
  "$INPUT" \
  --start-date 2025-06-01 \
  --end-date 2025-06-30 \
  --report \
  -o "$OUT_DIR/report.txt"

echo ""
echo "3. Validating outputs..."
python3 scripts/validation/validate_golden_output.py "$OUT_DIR"

echo ""
echo "✅ Golden run completed successfully!"