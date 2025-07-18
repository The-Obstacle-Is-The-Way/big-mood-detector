# Test Label CLI End-to-End Flow

## Context

The label CLI is fully implemented with the following commands:
- `label single` - Label a single date
- `label range` - Label a date range
- `label baseline` - Mark baseline periods
- `label batch` - Import labels from CSV
- `label interactive` - Interactive labeling mode
- `label stats` - View labeling statistics
- `label list` - List all labels
- `label export` - Export labels to CSV
- `label undo` - Remove last label

## What Needs Testing

### 1. E2E Test Flow
Create a comprehensive test that:
1. Processes a sample Apple Health export
2. Uses the label CLI to add episodes
3. Exports labels to CSV
4. Runs the training command with labeled data
5. Makes predictions with the fine-tuned model

### 2. Integration Tests
- Test database persistence across CLI invocations
- Test CSV import/export round-trip
- Test interactive mode with simulated input
- Test validation of date ranges and episode types

### 3. User Experience Testing
- Verify all error messages are helpful
- Test keyboard shortcuts in interactive mode
- Ensure progress indicators work correctly
- Validate help text is accurate

## Test Data Needed

1. Small Apple Health export for testing (~1 month of data)
2. Sample CSV with pre-labeled episodes
3. Expected outputs for comparison

## Acceptance Criteria

- [ ] All label commands work without errors
- [ ] Labels persist correctly in SQLite database
- [ ] CSV export format matches training requirements
- [ ] Interactive mode provides smooth UX
- [ ] Integration with training pipeline works
- [ ] Performance is acceptable (< 100ms per command)

@claude Please create comprehensive E2E tests for the label CLI that:
1. Test the full workflow from labeling to training
2. Include both happy path and error cases
3. Verify the CSV export format works with the training pipeline
4. Test the interactive mode with mocked user input