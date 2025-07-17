"""
Label CLI Commands

Main entry points for the labeling interface.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import click

from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
from big_mood_detector.infrastructure.logging import get_module_logger
from big_mood_detector.interfaces.cli.utils import console

from .interactive import InteractiveSession
from .validators import ClinicalValidator, parse_date_range

logger = get_module_logger(__name__)

# Mood aliases for user convenience
MOOD_ALIASES = {
    'dep': 'depressive', 'depression': 'depressive', 'd': 'depressive',
    'hypo': 'hypomanic', 'hypomanic': 'hypomanic', 'h': 'hypomanic',
    'mania': 'manic', 'manic': 'manic', 'm': 'manic',
    'mixed': 'mixed', 'mix': 'mixed', 'x': 'mixed',
    'stable': 'baseline', 'normal': 'baseline', 'none': 'baseline', 'n': 'baseline',
}


def normalize_mood(mood: str) -> str:
    """Normalize mood input to standard type."""
    return MOOD_ALIASES.get(mood.lower(), mood.lower())


@click.group(name="label", invoke_without_command=True)
@click.pass_context
def label_group(ctx: click.Context) -> None:
    """Create ground truth labels for model training."""
    if ctx.invoked_subcommand is None:
        # Default to episode subcommand
        ctx.invoke(label_episode_command)


@label_group.command(name="episode")
@click.option(
    "--predictions",
    type=click.Path(exists=True, path_type=Path),
    help="Predictions file for assisted labeling"
)
@click.option(
    "--date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Single date to label"
)
@click.option(
    "--date-range",
    type=str,
    help="Date range YYYY-MM-DD:YYYY-MM-DD"
)
@click.option(
    "--mood",
    type=str,
    help="Mood state (depressive, hypomanic, manic, mixed)"
)
@click.option(
    "--severity",
    type=int,
    default=3,
    help="Severity 1-5 (default: 3)"
)
@click.option(
    "--rater-id",
    type=str,
    default="default",
    help="Rater identifier for multi-rater support"
)
@click.option(
    "--notes",
    type=str,
    help="Additional notes about the episode"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for labels (CSV)"
)
@click.option(
    "--interactive/--no-interactive",
    default=True,
    help="Interactive prompts vs batch mode"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview without saving"
)
def label_episode_command(
    predictions: Optional[Path],
    date: Optional[datetime],
    date_range: Optional[str],
    mood: Optional[str],
    severity: int,
    rater_id: str,
    notes: Optional[str],
    output: Optional[Path],
    interactive: bool,
    dry_run: bool,
) -> None:
    """Label mood episodes with clinical validation."""
    
    # Initialize components
    labeler = EpisodeLabeler()
    validator = ClinicalValidator()
    
    # Handle interactive mode with predictions
    if interactive and predictions:
        session = InteractiveSession(labeler, validator, predictions)
        session.run()
        return
    
    # Parse dates
    if date:
        start_date = end_date = date.date()
    elif date_range:
        start_date, end_date = parse_date_range(date_range)
    else:
        if not interactive:
            raise click.BadParameter("Must specify --date or --date-range")
        # In interactive mode without dates, we'll prompt
        click.echo("Please specify a date or date range.")
        return
    
    # Normalize mood
    if mood:
        mood = normalize_mood(mood)
        if mood not in ['depressive', 'hypomanic', 'manic', 'mixed']:
            raise click.BadParameter(f"Invalid mood: {mood}")
    elif not interactive:
        raise click.BadParameter("Must specify --mood")
    
    # Validate episode duration
    duration_days = (end_date - start_date).days + 1
    validation = validator.validate_episode_duration(mood, start_date, end_date)
    
    if not validation.valid:
        click.echo(click.style(f"Warning: {validation.warning}", fg="yellow"))
        if validation.suggestion:
            click.echo(validation.suggestion)
        
        if not interactive or not click.confirm("Continue anyway?"):
            raise click.Abort()
    
    # Check for very long spans
    if duration_days > 90:
        click.echo(click.style(
            f"Warning: {duration_days} days is unusually long for a single episode.",
            fg="yellow"
        ))
        if not interactive or not click.confirm("Continue anyway?"):
            raise click.Abort()
    
    # Check for conflicts
    if labeler.check_overlap(start_date, end_date):
        click.echo(click.style("Conflict detected: Overlaps with existing episode", fg="red"))
        if not interactive or not click.confirm("Override existing label?"):
            raise click.Abort()
    
    # Perform labeling (or dry run)
    if dry_run:
        click.echo(click.style("DRY RUN - No changes will be saved", fg="cyan"))
        click.echo(f"Would label {start_date} to {end_date} as {mood} (severity {severity})")
        return
    
    # Add the episode
    if start_date == end_date:
        labeler.add_episode(
            date=start_date,
            episode_type=mood,
            severity=severity,
            notes=notes,
            rater_id=rater_id
        )
        click.echo(click.style(f"✓ Labeled {start_date} as {mood}", fg="green"))
    else:
        labeler.add_episode(
            start_date=start_date,
            end_date=end_date,
            episode_type=mood,
            severity=severity,
            notes=notes,
            rater_id=rater_id
        )
        click.echo(click.style(
            f"✓ Labeled {duration_days}-day {mood} episode", 
            fg="green"
        ))
    
    # Export to CSV if output specified
    if output:
        df = labeler.to_dataframe()
        df.to_csv(output, index=False)
        click.echo(f"Saved labels to {output}")


@label_group.command(name="baseline")
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Start date of baseline period"
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="End date of baseline period"
)
@click.option(
    "--notes",
    type=str,
    help="Notes about the baseline period"
)
@click.option(
    "--rater-id",
    type=str,
    default="default",
    help="Rater identifier"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview without saving"
)
def label_baseline_command(
    start: datetime,
    end: datetime,
    notes: Optional[str],
    rater_id: str,
    dry_run: bool,
) -> None:
    """Mark stable baseline periods."""
    labeler = EpisodeLabeler()
    
    start_date = start.date()
    end_date = end.date()
    
    if dry_run:
        click.echo(click.style("DRY RUN - No changes will be saved", fg="cyan"))
        click.echo(f"Would mark baseline from {start_date} to {end_date}")
        return
    
    labeler.add_baseline(
        start_date=start_date,
        end_date=end_date,
        notes=notes,
        rater_id=rater_id
    )
    
    duration = (end_date - start_date).days + 1
    click.echo(click.style(
        f"✓ Marked baseline period ({duration} days)", 
        fg="green"
    ))


@label_group.command(name="undo")
def label_undo_command() -> None:
    """Undo the last label entry."""
    labeler = EpisodeLabeler()
    
    if labeler.undo_last():
        click.echo(click.style("✓ Undid last label", fg="green"))
    else:
        click.echo(click.style("No labels to undo", fg="yellow"))