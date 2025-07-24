#!/bin/bash
# Script to create GitHub issues for tracked tech debt
# Run this with: ./scripts/create-tech-debt-issues.sh

echo "Creating GitHub issues for tech debt..."

# Issue 1: Streaming parser date bug
gh issue create \
  --title "Streaming parser date filtering fails with string/datetime comparison" \
  --body-file issues/streaming-parser-date-bug.md \
  --label bug \
  --label "tech debt" \
  --label "high priority"

# Issue 2: Baseline persistence legacy API  
gh issue create \
  --title "Baseline persistence tests use outdated domain entity APIs" \
  --body-file issues/baseline-persistence-legacy-api.md \
  --label "tech debt" \
  --label testing \
  --label "high priority"

# Issue 3: XGBoost Booster predict_proba
gh issue create \
  --title "XGBoost Booster objects loaded from JSON lack predict_proba method" \
  --body-file issues/xgboost-booster-predict-proba.md \
  --label bug \
  --label "tech debt" \
  --label architecture \
  --label "high priority"

echo "Issues created! Update xfail markers with the issue numbers."