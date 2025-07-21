#!/usr/bin/env python3
"""Fix indentation errors from the import moves."""

import re
from pathlib import Path


def fix_orphaned_imports(file_path: Path) -> bool:
    """Fix files with orphaned import lines."""
    try:
        content = file_path.read_text()
    except Exception:
        return False

    lines = content.split('\n')
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for orphaned import parts (indented without context)
        if re.match(r'^    (ActivityRecord|ActivityType|HeartRateRecord|SleepRecord|[A-Z]\w+),?$', line):
            # This is an orphaned import line, skip it
            i += 1
            continue

        # Check for incomplete imports
        if line.strip().endswith(',') and i + 1 < len(lines):
            next_line = lines[i + 1]
            if re.match(r'^    \w+', next_line) and not next_line.strip().startswith('from '):
                # Skip the orphaned continuation
                fixed_lines.append(line.rstrip(',') + ')')
                i += 2
                continue

        fixed_lines.append(line)
        i += 1

    # Write back
    new_content = '\n'.join(fixed_lines)
    if new_content != content:
        file_path.write_text(new_content)
        return True
    return False


def main():
    """Fix all test files."""
    test_dir = Path('tests')
    fixed_count = 0

    for file_path in test_dir.rglob('test_*.py'):
        if fix_orphaned_imports(file_path):
            print(f"Fixed: {file_path}")
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == '__main__':
    main()
