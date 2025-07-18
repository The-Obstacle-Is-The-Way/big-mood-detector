"""
Big Mood Detector - Main Application Entry Point

Multi-interface backend supporting:
- CLI interface (for testing and automation)
- API server (for external integrations)
- Future web interface (when UI is ready)

This follows professional backend patterns where main.py routes to different interfaces.
"""

from typing import Any

from big_mood_detector.interfaces.cli.main import cli as cli_interface


def main() -> None:
    """
    Main application entry point.

    Routes to appropriate interface based on context:
    - CLI interface (current default for testing)
    - API server (via 'serve' command)
    - Future web interface
    """
    # For now, default to CLI interface
    # This allows testing the full pipeline while we build other interfaces
    cli_interface()


def run_api_server(host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
    """
    Start the API server programmatically.

    This will be useful when we add web interface that needs to start
    the API server as a background service.
    """
    import uvicorn

    from big_mood_detector.interfaces.api.main import app

    uvicorn.run(app, host=host, port=port, **kwargs)


def run_cli() -> None:
    """Run CLI interface explicitly."""
    cli_interface()


# Future: def run_web_ui() -> None:
#     """Launch web interface (when implemented)."""
#     pass


if __name__ == "__main__":
    # When run directly, use CLI interface for testing
    cli_interface()
