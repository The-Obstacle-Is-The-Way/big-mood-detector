"""
Bridge between Click and Typer CLIs.

This module provides utilities to integrate Typer apps into Click groups.
"""

import sys
from typing import Any, List, Optional

import click
import typer
from typer.main import get_command


def typer_to_click(typer_app: typer.Typer, name: Optional[str] = None, **kwargs: Any) -> click.Command:
    """
    Convert a Typer application to a Click command.
    
    This allows Typer apps to be used as subcommands in Click groups.
    
    Args:
        typer_app: The Typer application to convert
        name: Optional name for the command
        **kwargs: Additional arguments passed to Click
        
    Returns:
        A Click command that wraps the Typer app
    """
    # Get the Click command from Typer
    click_command = get_command(typer_app)
    
    # Update with any additional kwargs
    if name:
        click_command.name = name
    
    # Preserve Typer's rich help formatting
    if hasattr(typer_app, 'info'):
        if typer_app.info.help:
            click_command.help = typer_app.info.help
    
    return click_command


class ClickTyperGroup(click.Group):
    """
    A Click group that can contain both Click and Typer commands.
    
    This allows seamless integration of Typer apps as subcommands.
    """
    
    def add_typer(self, typer_app: typer.Typer, name: str, **kwargs: Any) -> None:
        """
        Add a Typer application as a subcommand.
        
        Args:
            typer_app: The Typer application to add
            name: Name for the subcommand
            **kwargs: Additional arguments
        """
        click_command = typer_to_click(typer_app, name=name, **kwargs)
        self.add_command(click_command)