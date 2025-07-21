# Test Data Directory

This directory is for temporary test data that should NOT be committed to the repository.

All files in this directory (except this README) are ignored by git.

## Usage

When writing tests that need temporary files:
1. Use pytest's `tmp_path` fixture when possible
2. If you must create files, put them in this directory
3. Clean up after your tests

## Example

```python
def test_with_temp_file(tmp_path):
    # Preferred: use pytest's tmp_path
    temp_file = tmp_path / "test.xml"
    temp_file.write_text("<data>test</data>")
    
    # Your test code here
    
    # Cleanup happens automatically
```