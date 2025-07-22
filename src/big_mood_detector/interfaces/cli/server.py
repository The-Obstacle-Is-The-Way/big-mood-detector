"""
Server Command

API server management for the Big Mood Detector.
"""

import sys
from typing import TYPE_CHECKING, NoReturn

import click

if TYPE_CHECKING:
    import uvicorn
else:
    try:
        import uvicorn
    except ImportError:
        uvicorn = None


def bail_with_error(message: str) -> NoReturn:
    """Exit with error message. Properly typed to indicate no return."""
    click.echo(message, err=True)
    sys.exit(1)


@click.command(name="serve")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload/--no-reload", default=True, help="Enable auto-reload")
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of worker processes (disabled with --reload)",
)
def serve_command(host: str, port: int, reload: bool, workers: int) -> None:
    """Start the API server."""
    if uvicorn is None:
        bail_with_error(
            "Error: API dependencies not installed.\n"
            "Install with: pip install big-mood-detector[api]"
        )

    click.echo(f"Starting API server on {host}:{port}")

    if reload and workers > 1:
        click.echo("Note: --reload disables multiple workers")
        workers = 1

    uvicorn.run(
        "big_mood_detector.interfaces.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )
