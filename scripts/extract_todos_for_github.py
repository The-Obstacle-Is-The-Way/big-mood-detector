#!/usr/bin/env python3
"""Extract all TODO/FIXME comments and format them for GitHub issues."""

import re
import sys
from pathlib import Path
from typing import NamedTuple


class TodoItem(NamedTuple):
    """A TODO/FIXME comment found in the code."""
    file_path: str
    line_number: int
    content: str
    category: str  # TODO, FIXME, HACK, XXX


def extract_todos(root_dir: Path) -> list[TodoItem]:
    """Extract all TODO-like comments from the codebase."""
    todos = []
    patterns = [
        (r'#\s*(TODO)(?:\(.*?\))?:\s*(.+)', 'TODO'),
        (r'#\s*(FIXME)(?:\(.*?\))?:\s*(.+)', 'FIXME'),
        (r'#\s*(HACK):\s*(.+)', 'HACK'),
        (r'#\s*(XXX):\s*(.+)', 'XXX'),
    ]
    
    # Directories to skip
    skip_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache', 'htmlcov'}
    
    for py_file in root_dir.rglob('*.py'):
        # Skip if in excluded directory
        if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
            continue
            
        # Skip if in docs/archive (old documentation)
        if 'docs/archive' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern, category in patterns:
                        match = re.search(pattern, line)
                        if match:
                            content = match.group(2).strip()
                            todos.append(TodoItem(
                                file_path=str(py_file.relative_to(root_dir)),
                                line_number=line_num,
                                content=content,
                                category=category
                            ))
        except Exception as e:
            print(f"Error reading {py_file}: {e}", file=sys.stderr)
    
    return sorted(todos, key=lambda x: (x.file_path, x.line_number))


def format_github_issue(todo: TodoItem, issue_number: int) -> str:
    """Format a TODO as a GitHub issue."""
    title = f"{todo.category}: {todo.content[:60]}{'...' if len(todo.content) > 60 else ''}"
    
    body = f"""## Description
{todo.content}

## Location
- **File**: `{todo.file_path}`
- **Line**: {todo.line_number}

## Category
`{todo.category}`

## Acceptance Criteria
- [ ] Issue has been addressed
- [ ] Tests added/updated if applicable
- [ ] Documentation updated if applicable

## Priority
- [ ] High (blocking)
- [ ] Medium (should fix soon)
- [ ] Low (nice to have)
"""
    
    return f"""gh issue create \\
  --title "{title}" \\
  --body "{body}" \\
  --label "tech-debt" \\
  --label "{todo.category.lower()}"
"""


def format_code_update(todo: TodoItem, issue_number: int) -> str:
    """Format the code update to reference the issue."""
    return f"{todo.file_path}:{todo.line_number} -> # {todo.category}(gh-{issue_number}): {todo.content}"


def main():
    """Main function."""
    root_dir = Path(__file__).parent.parent
    todos = extract_todos(root_dir)
    
    if not todos:
        print("No TODOs found!")
        return
    
    print(f"Found {len(todos)} TODO items:\n")
    
    # Group by file
    from itertools import groupby
    for file_path, group in groupby(todos, key=lambda x: x.file_path):
        items = list(group)
        print(f"\n📁 {file_path} ({len(items)} items)")
        for todo in items:
            print(f"  L{todo.line_number}: {todo.category} - {todo.content[:80]}{'...' if len(todo.content) > 80 else ''}")
    
    print("\n" + "="*80)
    print("GitHub Issue Creation Commands:")
    print("="*80 + "\n")
    
    # Start issue numbers from 100 (adjust based on your repo)
    base_issue_num = 100
    
    for i, todo in enumerate(todos, start=base_issue_num):
        print(f"# Issue #{i}")
        print(format_github_issue(todo, i))
        print()
    
    print("\n" + "="*80)
    print("Code Updates (after creating issues):")
    print("="*80 + "\n")
    
    for i, todo in enumerate(todos, start=base_issue_num):
        print(format_code_update(todo, i))


if __name__ == "__main__":
    main()