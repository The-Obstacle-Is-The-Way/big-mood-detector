#!/bin/bash
# Create GitHub issues for remaining TODOs in Big Mood Detector codebase

echo "Creating GitHub issues for 5 remaining TODOs..."
echo "================================================"

# Issue 1: Label service check
echo "Creating issue for label service TODO..."
gh issue create \
  --title "Check if label is in use before deletion" \
  --body "## Description
Before deleting a label, we should check if it's in use by any records to prevent orphaned data.

## Location
- **File**: \`src/big_mood_detector/application/services/label_service.py\`
- **Line**: 180

## Category
\`TODO\`

## Acceptance Criteria
- [ ] Add check for label usage in records
- [ ] Return appropriate error if label is in use
- [ ] Add tests for the validation
- [ ] Update API documentation

## Priority
- [ ] High (blocking)
- [x] Medium (should fix soon)
- [ ] Low (nice to have)

## Assignee
@claude - Please work on this autonomously" \
  --label "tech-debt" \
  --label "enhancement"

# Issue 2: Structlog compatibility
echo "Creating issue for structlog compatibility..."
gh issue create \
  --title "Fix type compatibility with structlog for PrivacyFilter" \
  --body "## Description
The PrivacyFilter needs proper type compatibility with structlog processors. Currently commented out due to type mismatch.

## Location
- **File**: \`src/big_mood_detector/infrastructure/security/privacy.py\`
- **Line**: 156

## Category
\`TODO\`

## Acceptance Criteria
- [ ] Fix type compatibility issue
- [ ] Enable PrivacyFilter in structlog configuration
- [ ] Add tests to verify PII redaction works in logs
- [ ] Update logging documentation

## Priority
- [x] High (blocking) - Privacy/GDPR compliance
- [ ] Medium (should fix soon)
- [ ] Low (nice to have)

## Assignee
@claude - Please work on this autonomously" \
  --label "tech-debt" \
  --label "privacy" \
  --label "bug"

# Issue 3: Remove deprecated stubs
echo "Creating issue for removing deprecated stubs..."
gh issue create \
  --title "Remove deprecated feature engineering stubs after Q1 2025" \
  --body "## Description
Remove the deprecated stub methods that were moved to SleepFeatureCalculator. These are kept temporarily for backward compatibility.

## Location
- **File**: \`src/big_mood_detector/domain/services/advanced_feature_engineering.py\`
- **Line**: 516

## Category
\`TODO\`

## Acceptance Criteria
- [ ] Remove deprecated methods after Q1 2025
- [ ] Ensure all code has migrated to SleepFeatureCalculator
- [ ] Update any remaining imports
- [ ] Run full test suite to verify no breakage

## Priority
- [ ] High (blocking)
- [ ] Medium (should fix soon)
- [x] Low (nice to have) - Scheduled for Q1 2025" \
  --label "tech-debt" \
  --label "deprecation"

# Issue 4: Fix HR/HRV baseline calculation
echo "Creating issue for HR/HRV baseline calculation..."
gh issue create \
  --title "Fix HR/HRV baseline calculation to use actual values instead of defaults" \
  --body "## Description
The HR/HRV baseline calculation test is currently using default values instead of calculating from actual data. This needs to be fixed to ensure accurate baseline calculations.

## Location
- **File**: \`tests/integration/test_baseline_persistence_e2e.py\`
- **Line**: 255

## Category
\`TODO\`

## Acceptance Criteria
- [ ] Update baseline calculation to use actual HR/HRV values
- [ ] Remove magic defaults from calculation
- [ ] Update test assertions to verify correct values
- [ ] Add edge case tests for missing HR/HRV data

## Priority
- [x] High (blocking) - Affects clinical accuracy
- [ ] Medium (should fix soon)
- [ ] Low (nice to have)" \
  --label "tech-debt" \
  --label "bug" \
  --label "clinical"

# Issue 5: Implement large XML processing features
echo "Creating issue for large XML processing..."
gh issue create \
  --title "Implement XML extraction with date filtering and CSV export" \
  --body "## Description
The large XML processing script needs two features implemented:
1. XML extraction with date filtering (line 77)
2. CSV export functionality (line 137)

## Location
- **File**: \`scripts/process_large_xml.py\`
- **Lines**: 77, 137

## Category
\`TODO\`

## Acceptance Criteria
- [ ] Implement XML extraction with date range filtering
- [ ] Implement CSV export for processed data
- [ ] Add progress bars for large file processing
- [ ] Add error handling for malformed XML
- [ ] Update script documentation

## Priority
- [ ] High (blocking)
- [x] Medium (should fix soon)
- [ ] Low (nice to have)

## Assignee
@claude - Please work on this autonomously" \
  --label "tech-debt" \
  --label "enhancement" \
  --label "performance"

echo ""
echo "✅ All 5 GitHub issues created!"
echo ""
echo "Next steps:"
echo "1. Check the created issues on GitHub"
echo "2. Update code comments to reference issue numbers"
echo "3. Add pre-commit hook to enforce TODO(gh-XXX) format"