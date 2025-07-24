#!/usr/bin/env python3
"""Fix import issues properly by rewriting test files."""

import re
from pathlib import Path

# Map of common import patterns to proper module paths
IMPORT_MAP = {
    r"^ActivityRecord,$": "from big_mood_detector.domain.entities.activity_record import ActivityRecord",
    r"^ActivityType,$": "from big_mood_detector.domain.entities.activity_record import ActivityType",
    r"^SleepRecord, SleepState$": "from big_mood_detector.domain.entities.sleep_record import SleepRecord, SleepState",
    r"^HeartRateRecord,$": "from big_mood_detector.domain.entities.heart_rate_record import HeartRateRecord",
    r"^HeartMetricType,$": "from big_mood_detector.domain.entities.heart_rate_record import HeartMetricType",
}


def fix_file(file_path: Path) -> bool:
    """Fix a single test file by removing orphaned imports."""
    try:
        content = file_path.read_text()
    except Exception:
        return False

    lines = content.split("\n")
    fixed_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Remove lines that are just closing parens
        if line.strip() == ")" and i > 0 and not lines[i - 1].strip().endswith(")"):
            continue

        # Skip orphaned import fragments
        if re.match(
            r"^(ActivityRecord|ActivityType|SleepRecord|HeartRateRecord|HeartMetricType|[A-Z]\w+),$",
            line.strip(),
        ):
            continue

        # Fix imports on single lines
        stripped = line.strip()
        if stripped in IMPORT_MAP:
            fixed_lines.append(IMPORT_MAP[stripped])
            continue

        fixed_lines.append(line)

    new_content = "\n".join(fixed_lines)

    # Clean up multiple blank lines
    new_content = re.sub(r"\n\n\n+", "\n\n", new_content)

    if new_content != content:
        file_path.write_text(new_content)
        return True
    return False


def main():
    """Fix all test files."""
    test_files = []

    # Find all affected test files
    for pattern in ["test_*.py", "*_test.py"]:
        test_files.extend(Path("tests").rglob(pattern))

    fixed = 0
    for file_path in test_files:
        if fix_file(file_path):
            print(f"Fixed: {file_path}")
            fixed += 1

    print(f"\nFixed {fixed} files")


if __name__ == "__main__":
    main()
