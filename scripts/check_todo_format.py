#!/usr/bin/env python3
"""
Check that all TODO comments follow the format TODO(gh-XXX).

This script is used as a pre-commit hook to ensure all TODOs
reference GitHub issues.
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple


class TodoMatch(NamedTuple):
    """A TODO comment match."""
    file_path: Path
    line_number: int
    line_content: str
    todo_text: str


def check_todo_format(file_path: Path) -> list[TodoMatch]:
    """Check TODO format in a single file."""
    invalid_todos = []
    
    # Pattern for valid TODO format: TODO(gh-123)
    valid_pattern = re.compile(r'TODO\(gh-\d+\)')
    # Pattern to find any TODO
    todo_pattern = re.compile(r'TODO[:\s]')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if 'TODO' in line:
                    # Skip if it's a valid format
                    if valid_pattern.search(line):
                        continue
                    
                    # Check if it's a TODO that needs formatting
                    if todo_pattern.search(line):
                        # Extract the TODO text
                        todo_match = todo_pattern.search(line)
                        if todo_match:
                            invalid_todos.append(TodoMatch(
                                file_path=file_path,
                                line_number=line_num,
                                line_content=line.strip(),
                                todo_text=line[todo_match.start():].strip()
                            ))
    except Exception:
        # Skip files that can't be read
        pass
    
    return invalid_todos


def main():
    """Check all Python files for TODO format."""
    # Get all Python files in src and tests
    src_path = Path('src')
    tests_path = Path('tests')
    scripts_path = Path('scripts')
    
    all_files = []
    for base_path in [src_path, tests_path, scripts_path]:
        if base_path.exists():
            all_files.extend(base_path.rglob('*.py'))
    
    # Check each file
    all_invalid_todos = []
    for file_path in all_files:
        invalid_todos = check_todo_format(file_path)
        all_invalid_todos.extend(invalid_todos)
    
    # Report results
    if all_invalid_todos:
        print("❌ Found TODOs not referencing GitHub issues:")
        print("=" * 60)
        
        for todo in all_invalid_todos:
            print(f"\n{todo.file_path}:{todo.line_number}")
            print(f"  {todo.line_content}")
        
        print("\n" + "=" * 60)
        print(f"Total: {len(all_invalid_todos)} invalid TODOs")
        print("\nPlease update TODOs to format: TODO(gh-XXX): description")
        print("where XXX is the GitHub issue number.")
        
        return 1
    else:
        print("✅ All TODOs properly reference GitHub issues!")
        return 0


if __name__ == "__main__":
    sys.exit(main())