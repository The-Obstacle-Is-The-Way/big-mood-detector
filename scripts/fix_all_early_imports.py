#!/usr/bin/env python3
"""Fix ALL early imports in test files to restore coverage."""

import os
import re
import sys
from pathlib import Path


def find_all_test_files():
    """Find all test files in the tests directory."""
    test_dir = Path(__file__).parent.parent / 'tests'
    test_files = []

    for root, dirs, files in os.walk(test_dir):
        # Skip __pycache__ and .venv
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.venv')]

        for file in files:
            if file.endswith('.py') and (file.startswith('test_') or file == 'conftest.py'):
                test_files.append(Path(root) / file)

    return test_files


def move_imports_in_file(file_path):
    """Move early imports in a single file."""
    try:
        content = file_path.read_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    lines = content.split('\n')

    # Find imports to move
    imports_to_move = []
    import_indices = []

    for i, line in enumerate(lines):
        if re.match(r'^(from big_mood_detector|import big_mood_detector)', line.strip()):
            # Check if it's at module level (not indented)
            if line and not line[0].isspace():
                imports_to_move.append(line)
                import_indices.append(i)

    if not imports_to_move:
        return False

    # Remove imports from their current location
    for idx in reversed(import_indices):
        lines.pop(idx)

    # Special handling for conftest.py
    if file_path.name == 'conftest.py':
        # Find first fixture function
        for i, line in enumerate(lines):
            if '@pytest.fixture' in line:
                # Find the function definition
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().startswith('def '):
                        # Insert imports at the beginning of the fixture
                        for imp in imports_to_move:
                            lines.insert(j + 1, '    ' + imp)
                        file_path.write_text('\n'.join(lines))
                        return True

        # If no fixture found, find any function
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                for imp in imports_to_move:
                    lines.insert(i + 1, '    ' + imp)
                file_path.write_text('\n'.join(lines))
                return True

    # Find first test method or fixture
    insert_idx = None
    base_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Look for test methods, fixtures, or setup methods
        if (stripped.startswith('def test_') or
            stripped.startswith('def setup') or
            stripped.startswith('@pytest.fixture') or
            (stripped.startswith('def ') and 'conftest' not in str(file_path))):

            if stripped.startswith('@'):
                # Find the actual function after decorator
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().startswith('def '):
                        insert_idx = j
                        base_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
            else:
                insert_idx = i
                base_indent = len(line) - len(line.lstrip())

            if insert_idx is not None:
                break

    if insert_idx is None:
        # No method found, try to find first class
        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                # Find first method in class
                for j in range(i+1, len(lines)):
                    if lines[j].strip().startswith('def '):
                        insert_idx = j
                        base_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                break

    if insert_idx is None:
        print(f"Warning: Could not find insertion point in {file_path}")
        return False

    # Insert imports
    indent = ' ' * (base_indent + 4)
    insert_pos = insert_idx + 1

    # Skip docstring if present
    if insert_pos < len(lines) and lines[insert_pos].strip().startswith('"""'):
        while insert_pos < len(lines) and '"""' not in lines[insert_pos][1:]:
            insert_pos += 1
        insert_pos += 1

    for imp in imports_to_move:
        lines.insert(insert_pos, indent + imp)
        insert_pos += 1

    file_path.write_text('\n'.join(lines))
    return True


def main():
    """Fix all test files."""
    print("Finding all test files...")
    test_files = find_all_test_files()
    print(f"Found {len(test_files)} test files")

    fixed_count = 0
    for file_path in sorted(test_files):
        if move_imports_in_file(file_path):
            print(f"Fixed: {file_path.relative_to(Path(__file__).parent.parent)}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")
    return 0


if __name__ == '__main__':
    sys.exit(main())
