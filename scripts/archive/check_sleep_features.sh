#!/usr/bin/env bash
# Sleep feature regression checks
#
# Usage:
#   ./scripts/check_sleep_features.sh
#   VERBOSE=1 ./scripts/check_sleep_features.sh
#
# Exits non-zero on any hard failure so make test / CI will stop.

set -euo pipefail

# 1. Critical bug guard - prevent sleep_percentage * 24 pattern
echo "🔍 Checking for 'sleep_percentage * 24' pattern..."

RESULTS=$(grep -rn "sleep_percentage.*\*.*24" \
    --include="*.py" \
    --exclude-dir=".git" \
    --exclude-dir="__pycache__" \
    --exclude-dir=".pytest_cache" \
    --exclude-dir="reference_repos" \
    --exclude-dir="tests" \
    --exclude="*test*.py" \
    . 2>/dev/null | \
    grep -v "# WARNING:" | \
    grep -v "# This" | \
    grep -v "# DO NOT" | \
    grep -v "This fixes the bug" | \
    grep -v "bogus sleep_percentage" | \
    grep -E "=|return|print" || true)

if [ -n "$RESULTS" ]; then
    echo "❌ FOUND PROBLEMATIC PATTERN: sleep_percentage * 24"
    echo ""
    echo "The following files contain the bug-prone calculation:"
    echo "$RESULTS"
    echo ""
    echo "Please use SleepAggregator.aggregate_daily() instead."
    exit 1
fi

echo "✅ No sleep_percentage * 24 pattern found!"

# 2. Schema guard - verify Seoul feature set (36 features)
echo ""
echo "🔍 Verifying Seoul-schema feature set (36 features)..."

if ! python3 scripts/assert_feature_schema.py; then
    echo "❌ Feature schema validation failed!"
    exit 1
fi

# 3. Legacy heuristic (optional, verbose only)
if [[ "${VERBOSE:-0}" == "1" ]]; then
    echo ""
    echo "ℹ️  [verbose] Checking legacy sleep_percentage usage patterns..."
    
    WARNINGS=$(grep -rn '\bsleep_percentage\b' \
        --include='*.py' \
        --exclude-dir=".git" \
        --exclude-dir="__pycache__" \
        --exclude-dir=".pytest_cache" \
        --exclude-dir="reference_repos" \
        src/ 2>/dev/null | \
        grep -vE '^\s*#' | \
        grep -vE '""".*"""' | \
        grep -v 'sleep_duration_hours' | \
        head -5 || true)
    
    if [ -n "$WARNINGS" ]; then
        echo "⚠️  Found sleep_percentage usage without sleep_duration_hours:"
        echo "$WARNINGS"
        echo ""
        echo "Note: This is informational only and does not fail the check."
    fi
fi

echo ""
echo "✅ All sleep feature checks complete!"