"""
Label Statistics Command

Display statistics about labeled episodes.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
from big_mood_detector.interfaces.cli.utils import console


@click.command(name="stats")
@click.option(
    "--db", type=click.Path(exists=True, path_type=Path), help="SQLite database path"
)
@click.option("--rater", type=str, help="Filter by specific rater")
@click.option("--date-range", type=str, help="Date range YYYY-MM-DD:YYYY-MM-DD")
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def label_stats_command(
    db: Path | None, rater: str | None, date_range: str | None, format: str
) -> None:
    """Display statistics about labeled episodes."""
    labeler = EpisodeLabeler()

    # Load from database if specified
    if db:
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )

        repo = SQLiteEpisodeRepository(db_path=db)
        repo.load_into_labeler(labeler)

    # Calculate statistics
    total_episodes = len(labeler.episodes)
    total_baselines = len(labeler.baseline_periods)

    if total_episodes == 0 and total_baselines == 0:
        console.print("[yellow]No labeled data found[/yellow]")
        return

    # Episode type distribution
    episode_types = Counter(ep["episode_type"] for ep in labeler.episodes)

    # Rater statistics
    all_raters = set()
    for ep in labeler.episodes:
        all_raters.add(ep.get("rater_id", "default"))
    for bl in labeler.baseline_periods:
        all_raters.add(bl.get("rater_id", "default"))

    # Date range
    all_dates = []
    for ep in labeler.episodes:
        if "date" in ep:
            all_dates.append(ep["date"])
        else:
            all_dates.extend([ep["start_date"], ep["end_date"]])
    for bl in labeler.baseline_periods:
        all_dates.extend([bl["start_date"], bl["end_date"]])

    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range_str = f"{min_date} to {max_date}"
    else:
        date_range_str = "No dates"

    if format == "json":
        # JSON output
        import json

        stats = {
            "total_episodes": total_episodes,
            "total_baselines": total_baselines,
            "episode_distribution": dict(episode_types),
            "raters": list(all_raters),
            "date_range": {"start": min_date, "end": max_date} if all_dates else None,
        }
        console.print_json(data=stats)
    else:
        # Table output
        panel = Panel(
            f"[bold]Total Episodes:[/bold] {total_episodes}\n"
            f"[bold]Total Baselines:[/bold] {total_baselines}\n"
            f"[bold]Date Range:[/bold] {date_range_str}\n"
            f"[bold]Raters:[/bold] {len(all_raters)} ({', '.join(sorted(all_raters))})",
            title="[bold blue]Label Statistics[/bold blue]",
            border_style="blue",
        )
        console.print(panel)

        if episode_types:
            # Episode distribution table
            table = Table(
                title="Episode Distribution",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Type", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right")

            total = sum(episode_types.values())
            for episode_type, count in sorted(
                episode_types.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total) * 100
                table.add_row(
                    episode_type.capitalize(), str(count), f"{percentage:.1f}%"
                )

            console.print(table)


@click.command(name="export")
@click.option(
    "--db", type=click.Path(exists=True, path_type=Path), help="SQLite database path"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path (CSV)",
)
@click.option(
    "--format",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Export format",
)
def label_export_command(db: Path | None, output: Path, format: str) -> None:
    """Export labeled episodes to file."""
    labeler = EpisodeLabeler()

    # Load from database if specified
    if db:
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )

        repo = SQLiteEpisodeRepository(db_path=db)
        repo.load_into_labeler(labeler)

    if not labeler.episodes and not labeler.baseline_periods:
        console.print("[yellow]No data to export[/yellow]")
        return

    # Export to dataframe
    df = labeler.to_dataframe()

    if format == "csv":
        df.to_csv(output, index=False)
    else:
        df.to_json(output, orient="records", indent=2)

    console.print(f"[green]✓ Exported to {output}[/green]")
