#!/bin/bash
# CI script to prevent the sleep_percentage * 24 bug from returning
#
# This script fails if it finds the problematic pattern in the codebase,
# forcing developers to use the correct sleep duration calculation.

set -e

echo "🔍 Checking for sleep_percentage * 24 pattern..."

# Search for the problematic pattern - exclude tests, comments, and specific known safe files
RESULTS=$(grep -rn "sleep_percentage.*\*.*24" \
    --include="*.py" \
    --exclude-dir=".git" \
    --exclude-dir="__pycache__" \
    --exclude-dir=".pytest_cache" \
    --exclude-dir="reference_repos" \
    --exclude-dir="tests" \
    --exclude="*test*.py" \
    --exclude="test_xml_complete_flow.py" \
    . 2>/dev/null | \
    # Remove lines that are clearly comments or documentation
    grep -v "# WARNING:" | \
    grep -v "# This" | \
    grep -v "# DO NOT" | \
    grep -v "This fixes the bug" | \
    grep -v "bogus sleep_percentage" | \
    # Check if there's actual code (assignment or calculation)
    grep -E "=|return|print" || true)

if [ -n "$RESULTS" ]; then
    echo "❌ FOUND PROBLEMATIC PATTERN: sleep_percentage * 24"
    echo ""
    echo "The following files contain the bug-prone calculation:"
    echo "$RESULTS"
    echo ""
    echo "Please use SleepAggregator.aggregate_daily() instead of sleep_percentage * 24"
    echo "See: src/big_mood_detector/application/services/aggregation_pipeline.py"
    echo "     _get_actual_sleep_duration() for the correct implementation"
    exit 1
fi

echo "✅ No sleep_percentage * 24 pattern found - code is clean!"

# Also check for direct sleep_percentage usage without sleep_duration_hours
echo ""
echo "🔍 Checking for sleep_percentage usage without sleep_duration_hours..."

# More lenient check - just warn about sleep_percentage usage
WARNINGS=$(grep -r "sleep_percentage" \
    --include="*.py" \
    --exclude-dir=".git" \
    --exclude-dir="__pycache__" \
    --exclude-dir=".pytest_cache" \
    --exclude-dir="reference_repos" \
    --exclude="check_no_sleep_percentage.sh" \
    . 2>/dev/null | grep -v "sleep_duration_hours" | grep -v "test_" || true)

if [ -n "$WARNINGS" ]; then
    echo "⚠️  WARNING: Found sleep_percentage usage without sleep_duration_hours:"
    echo "$WARNINGS" | head -5
    echo ""
    echo "Consider using sleep_duration_hours instead for clarity."
fi

echo ""
echo "✅ Sleep calculation checks complete!"