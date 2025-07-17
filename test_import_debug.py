#!/usr/bin/env python
"""Debug script to test the import command."""

import traceback

try:
    from typer.testing import CliRunner
    from src.big_mood_detector.interfaces.cli.label_commands import app

    runner = CliRunner()

    # Create a test file
    import json
    import tempfile
    import pathlib

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([{"name": "Test", "description": "Test", "category": "mood", "color": "#FF0000"}], f)
        temp_path = pathlib.Path(f.name)

    print(f"Created temp file: {temp_path}")
    print(f"File exists: {temp_path.exists()}")

    # Run the command
    result = runner.invoke(app, ["import", str(temp_path)])
    print(f"Exit code: {result.exit_code}")
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")

    # Clean up
    temp_path.unlink()
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()