#!/usr/bin/env python3
"""Restore test imports to a working state after the botched move."""

import re
from pathlib import Path


def restore_imports(file_path: Path) -> bool:
    """Restore imports in test files."""
    try:
        content = file_path.read_text()
    except Exception:
        return False

    # Remove orphaned closing parentheses and empty import lines
    lines = content.split('\n')
    cleaned_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip orphaned closing parens
        if line.strip() == ')' and i > 0:
            prev_line = lines[i-1].strip()
            # Only skip if previous line doesn't end with something that needs closing
            if not (prev_line.endswith(',') or prev_line.endswith('(') or
                    prev_line.endswith('{') or prev_line.endswith('[')):
                i += 1
                continue

        # Skip empty lines after imports were removed
        if i > 0 and lines[i-1].strip() == 'import pytest' and line.strip() == '':
            # Check if next line is also empty or a paren
            if i + 1 < len(lines) and lines[i+1].strip() in ('', ')'):
                i += 1
                continue

        cleaned_lines.append(line)
        i += 1

    new_content = '\n'.join(cleaned_lines)

    # Clean up excessive blank lines
    new_content = re.sub(r'\n\n\n+', '\n\n', new_content)

    # Fix specific patterns
    new_content = re.sub(r'import pytest\n\n\)', 'import pytest', new_content)

    if new_content != content:
        file_path.write_text(new_content)
        return True
    return False


def main():
    """Main function."""
    test_dir = Path('tests')
    fixed = 0

    for file_path in test_dir.rglob('*.py'):
        if restore_imports(file_path):
            print(f"Fixed: {file_path}")
            fixed += 1

    print(f"\nFixed {fixed} files")


if __name__ == '__main__':
    main()
