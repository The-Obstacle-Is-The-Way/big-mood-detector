# Remove Hardcoded Paths from Tests and Scripts

## Problem

The audit revealed hardcoded paths throughout the codebase that will break in different environments:

### Test Files with Hardcoded Paths
- `tests/conftest.py`: `/Users/ray/Documents/Research/health/apple_export/`
- `tests/benchmarks/test_pat_benchmark.py`: `/Users/ray/Desktop/PAT-research-paper/`
- Multiple test files reference user-specific paths

### Scripts with Hardcoded Paths
- Various scripts assume `apple_export/` and `health_auto_export/` directories
- Processing scripts have embedded paths instead of using configuration

## Solution

### 1. Create Settings Module
```python
# src/big_mood_detector/infrastructure/settings/paths.py
from pathlib import Path
from pydantic import BaseSettings

class PathSettings(BaseSettings):
    # Test data paths
    test_data_dir: Path = Path("tests/fixtures/data")
    test_apple_export: Path = test_data_dir / "apple_export"
    test_health_export: Path = test_data_dir / "health_auto_export"
    
    # Default processing paths
    default_apple_export: Path = Path("apple_export")
    default_health_export: Path = Path("health_auto_export")
    
    class Config:
        env_prefix = "BIG_MOOD_"
```

### 2. Update Tests
Replace all hardcoded paths with fixture references or environment variables

### 3. Update Scripts
Make all scripts accept path arguments or read from environment

## Files to Update
- [ ] `tests/conftest.py`
- [ ] `tests/benchmarks/test_pat_benchmark.py`
- [ ] All scripts in `scripts/` directory
- [ ] Any other files with `/Users/` paths

## Testing
- [ ] Tests pass on different machines
- [ ] Scripts work with custom paths
- [ ] Docker container works without local paths

@claude Please fix all hardcoded paths by:
1. Creating a centralized path configuration
2. Updating all tests to use fixtures or settings
3. Making scripts accept path arguments
4. Ensuring no user-specific paths remain in the codebase