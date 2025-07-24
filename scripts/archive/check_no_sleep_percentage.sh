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

# Check that we have exactly 36 features in the Seoul XGBoost format
echo "🔍 Checking that DailyFeatures maintains exactly 36 features..."

# Expected feature names (10 sleep × 3 + 2 circadian × 3 = 36)
EXPECTED_FEATURES=(
    "sleep_percentage_MN" "sleep_percentage_SD" "sleep_percentage_Z"
    "sleep_amplitude_MN" "sleep_amplitude_SD" "sleep_amplitude_Z"
    "long_num_MN" "long_num_SD" "long_num_Z"
    "long_len_MN" "long_len_SD" "long_len_Z"
    "long_ST_MN" "long_ST_SD" "long_ST_Z"
    "long_WT_MN" "long_WT_SD" "long_WT_Z"
    "short_num_MN" "short_num_SD" "short_num_Z"
    "short_len_MN" "short_len_SD" "short_len_Z"
    "short_ST_MN" "short_ST_SD" "short_ST_Z"
    "short_WT_MN" "short_WT_SD" "short_WT_Z"
    "circadian_amplitude_MN" "circadian_amplitude_SD" "circadian_amplitude_Z"
    "circadian_phase_MN" "circadian_phase_SD" "circadian_phase_Z"
)

# Count features in DailyFeatures.to_dict() method
FEATURE_COUNT=$(grep -E '"(sleep_|long_|short_|circadian_).*(_MN|_SD|_Z)"' \
    src/big_mood_detector/application/services/aggregation_pipeline.py | \
    grep -v "activity_" | \
    grep -v "daily_" | \
    grep -v "#" | \
    wc -l | tr -d ' ')

if [ "$FEATURE_COUNT" -ne "36" ]; then
    echo "❌ ERROR: DailyFeatures should have exactly 36 features, found $FEATURE_COUNT"
    echo ""
    echo "The Seoul XGBoost models expect exactly 36 features:"
    echo "- 10 sleep indexes × 3 (mean, SD, Z-score) = 30"
    echo "- 2 circadian indexes × 3 (mean, SD, Z-score) = 6"
    echo ""
    echo "Do not add or remove features without retraining the models!"
    exit 1
fi

echo "✅ DailyFeatures has exactly 36 features as expected!"
echo ""
echo "✅ All sleep calculation checks complete!"