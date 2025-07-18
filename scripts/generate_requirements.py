#!/usr/bin/env python3
"""
Generate locked requirements files from pyproject.toml.

Creates:
- requirements.txt (base dependencies)
- requirements-ml.txt (with ML dependencies)
- requirements-dev.txt (with dev dependencies)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list[str]) -> None:
    """Run a command and exit on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}:")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)


def main():
    """Generate all requirements files."""
    print("Installing pip-tools...")
    run_command([sys.executable, "-m", "pip", "install", "pip-tools"])
    
    # Change to project root
    import os
    os.chdir(PROJECT_ROOT)
    
    print("\nGenerating requirements.txt (base dependencies)...")
    run_command([
        sys.executable, "-m", "piptools", "compile",
        "--resolver=backtracking",
        "--output-file=requirements.txt",
        "pyproject.toml"
    ])
    
    print("\nGenerating requirements-ml.txt (with ML dependencies)...")
    run_command([
        sys.executable, "-m", "piptools", "compile",
        "--resolver=backtracking",
        "--extra=ml",
        "--output-file=requirements-ml.txt",
        "pyproject.toml"
    ])
    
    print("\nGenerating requirements-dev.txt (with dev dependencies)...")
    run_command([
        sys.executable, "-m", "piptools", "compile",
        "--resolver=backtracking",
        "--extra=dev",
        "--output-file=requirements-dev.txt",
        "pyproject.toml"
    ])
    
    print("\n✅ All requirements files generated!")
    print("\nTo update dependencies:")
    print("  1. Update pyproject.toml")
    print("  2. Run: python scripts/generate_requirements.py")
    print("  3. Commit all changes")


if __name__ == "__main__":
    main()