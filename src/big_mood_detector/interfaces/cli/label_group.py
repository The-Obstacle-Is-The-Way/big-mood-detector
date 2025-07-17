"""
Unified label commands combining episode labeling and label management.

This module creates a unified 'label' command group that includes:
- Episode labeling commands (existing Click-based)
- Label management commands (new Typer-based)
"""

import click

from big_mood_detector.interfaces.cli.label_commands import app as label_mgmt_app
from big_mood_detector.interfaces.cli.labeling.commands import (
    label_baseline_command,
    label_episode_command,
    label_undo_command,
)
from big_mood_detector.interfaces.cli.typer_bridge import typer_to_click


@click.group(name="label", invoke_without_command=True)
@click.pass_context
def unified_label_group(ctx: click.Context) -> None:
    """
    Manage labels and create ground truth annotations.

    This command group provides:

    \b
    Episode Labeling:
      - episode: Label mood episodes with clinical validation
      - baseline: Mark stable baseline periods
      - undo: Undo the last label entry

    \b
    Label Management:
      - manage: Create, list, search, and manage label definitions

    Examples:

    \b
    # Label a depressive episode
    $ bigmood label episode --date-range 2024-01-01:2024-01-14 --mood depressive

    \b
    # Manage label definitions
    $ bigmood label manage list
    $ bigmood label manage create --name "Anxiety" --category mood
    """
    if ctx.invoked_subcommand is None:
        # Show help if no subcommand
        ctx.get_help()


# Add existing episode labeling commands
unified_label_group.add_command(label_episode_command, name="episode")
unified_label_group.add_command(label_baseline_command, name="baseline")
unified_label_group.add_command(label_undo_command, name="undo")

# Add new label management commands as a subgroup
label_mgmt_click = typer_to_click(label_mgmt_app, name="manage")
label_mgmt_click.help = "Manage label definitions (create, list, search, etc.)"
unified_label_group.add_command(label_mgmt_click)
