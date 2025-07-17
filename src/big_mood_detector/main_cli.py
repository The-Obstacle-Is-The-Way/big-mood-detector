"""
Big Mood Detector CLI - Entry Point

This module serves as the main entry point for the CLI.
It delegates to the properly organized CLI in the interfaces layer.

This maintains backward compatibility while following Clean Architecture.
"""

from big_mood_detector.interfaces.cli.main import cli

if __name__ == "__main__":
    cli()
