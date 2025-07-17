"""
Shared Rich Console Instance

Provides a centralized console for consistent formatting across CLI.
"""

from rich.console import Console

# Create a single console instance with consistent configuration
console = Console(
    # Add any theme customization here
    highlight=True,
    soft_wrap=True,
)

# Export for convenience
__all__ = ["console"]