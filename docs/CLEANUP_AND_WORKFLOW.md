# Branch Cleanup & Improved Workflow

## Remote Branches to Delete

Execute these commands to clean up the failed parallel development branches:

```bash
# Delete Claude's issue branches
git push origin --delete claude/issue-29-20250720-1739
git push origin --delete claude/issue-30-20250720-1740  
git push origin --delete claude/issue-34-20250720-1742

# Delete merged feature branch
git push origin --delete feature/add-progress-indication-issue-31
```

## New Sequential Workflow

### 1. Issue Creation Template

```markdown
## Issue: [Feature Name]

### Problem
Clear description of what's broken/missing

### Success Criteria
- [ ] Specific measurable outcome
- [ ] Performance target (if applicable)
- [ ] Memory usage limit (if applicable)

### Technical Approach
High-level approach (to be refined in PR)

### Test Requirements
- [ ] Unit tests for core logic
- [ ] Integration tests for feature
- [ ] Performance tests (if applicable)
```

### 2. Development Workflow

```bash
# 1. Always start from latest development
git checkout development
git pull origin development

# 2. Create feature branch
git checkout -b feature/issue-{number}-{short-description}

# 3. Write failing test FIRST
cat > tests/unit/test_new_feature.py << 'EOF'
def test_feature_does_x():
    # Given
    # When  
    # Then
    assert False  # Start with failing test
EOF

# 4. Implement minimum code to pass
# 5. Refactor for clean architecture
# 6. Add integration tests
# 7. Run full test suite
make test

# 8. Create PR with template
```

### 3. PR Template

```markdown
## Summary
Brief description of changes

## Issue
Fixes #[issue-number]

## Approach
- How you solved the problem
- Key design decisions
- Trade-offs considered

## Testing
- [ ] Unit tests added
- [ ] Integration tests added  
- [ ] Performance validated (if applicable)
- [ ] No architecture violations

## Checklist
- [ ] No code duplication (DRY)
- [ ] No magic numbers (configuration used)
- [ ] Single Responsibility Principle
- [ ] Clean Architecture boundaries respected
- [ ] All tests passing
- [ ] Coverage maintained/improved

## Breaking Changes
None / List any breaking changes
```

### 4. Configuration-First Development

```python
# Bad: Magic numbers
if file_size > 50 * 1024 * 1024:  # What is 50MB?

# Good: Configuration
@dataclass
class ParserConfig:
    large_file_threshold_mb: int = 50
    batch_size: int = 10000
    memory_limit_mb: int = 1000
    
if file_size > config.large_file_threshold_mb * 1024 * 1024:
```

### 5. TDD Enforcement

```python
# Step 1: Write the test
@pytest.mark.performance
def test_large_xml_completes_in_time_limit():
    # Given: 500MB XML file
    generator = XMLDataGenerator()
    large_file = generator.create_export(size_mb=500)
    
    # When: Processing the file
    start = time.time()
    result = parser.parse_file(large_file)
    duration = time.time() - start
    
    # Then: Completes within 5 minutes
    assert duration < 300  # 5 minutes
    assert len(result) > 0

# Step 2: Run test (should fail)
pytest -xvs tests/test_large_xml.py::test_large_xml_completes_in_time_limit

# Step 3: Implement feature
# Step 4: Test passes
# Step 5: Refactor for cleanliness
```

## Priority Issues to Implement

### 1. Issue #29: XML Streaming Performance
```python
# Requirements:
- Process 500MB+ files without timeout
- Constant memory usage (streaming)
- O(n+m) complexity with pre-indexing
- Progress reporting

# Approach:
1. Write performance test suite
2. Refactor parser to use streaming
3. Add pre-indexing for date lookups
4. Configure thresholds
```

### 2. Issue #40: XGBoost predict_proba
```python
# Requirements:
- Fix JSON loading to include predict_proba
- Maintain backward compatibility
- Enable ensemble predictions

# Approach:
1. Write test showing the bug
2. Fix XGBoost model loading
3. Validate ensemble works
```

### 3. Issue #27: True Ensemble
```python
# Requirements:  
- PAT provides predictions, not just embeddings
- Weighted voting between models
- Configurable ensemble weights

# Approach:
1. Design ensemble interface
2. Implement voting mechanism
3. Add weight configuration
```

## Lessons Learned

1. **Sequential > Parallel** for feature development
2. **TDD prevents** architectural violations
3. **Configuration > Magic Numbers** always
4. **Clean Architecture** must be enforced
5. **One PR at a time** prevents conflicts

## Implementation Order

1. Delete old branches (cleanup)
2. Implement Issue #29 (XML streaming) - Critical user impact
3. Implement Issue #40 (XGBoost fix) - Enables ensemble
4. Implement Issue #27 (True ensemble) - Improves accuracy

Each issue should take 1-2 days with proper TDD approach.