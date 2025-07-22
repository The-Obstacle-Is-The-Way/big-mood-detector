#!/bin/bash
set -euo pipefail

err=0
for d in apple_export health_auto_export; do
  # directory exists *and* is not empty
  if [ -d "$d" ] && [ "$(find "$d" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "❌ ERROR: data directory '$d' must not live at repo root"
    err=1
  fi
done

# same idea for rogue output files (>0 matches triggers error)
if ls *_predictions.csv *_clinical_report.txt *.summary.json 2>/dev/null | grep -q .; then
  echo "❌ ERROR: output artefacts found at repo root"
  err=1
fi

# Check for large files outside allowed directories
large_files=$(find . -type f -size +1M | grep -v -E '(\.git/|model_weights/|literature/|data/|reference_repos/|\.venv/|\.mypy_cache/|site/)' || true)
if [ -n "$large_files" ]; then
  echo "❌ ERROR: Large files found outside allowed directories:"
  echo "$large_files"
  err=1
fi

if [ "$err" -eq 0 ]; then
  echo "✅ Repository is clean"
fi
exit "$err"