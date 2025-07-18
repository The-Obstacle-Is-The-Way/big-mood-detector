"""
Import Command for Label CLI

Import episodes from CSV files.
"""

import csv
from pathlib import Path

import click

from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
    SQLiteEpisodeRepository,
)
from big_mood_detector.interfaces.cli.utils import console


@click.command(name="import")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--db", type=click.Path(path_type=Path), help="SQLite database for persistence"
)
@click.option("--dry-run", is_flag=True, help="Preview without importing")
def label_import_command(input_file: Path, db: Path | None, dry_run: bool) -> None:
    """Import episodes from CSV file."""
    labeler = EpisodeLabeler()

    # Load existing data if database specified
    if db:
        repo = SQLiteEpisodeRepository(db_path=db)
        repo.load_into_labeler(labeler)
        console.print(
            f"[dim]Loaded {len(labeler.episodes)} existing episodes from database[/dim]"
        )

    # Read CSV file
    imported_count = 0
    skipped_count = 0

    with open(input_file) as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Parse row data
                if "date" in row:
                    # Single day episode
                    if not dry_run:
                        labeler.add_episode(
                            date=row["date"],
                            episode_type=row.get(
                                "mood", row.get("episode_type", "depressive")
                            ),
                            severity=int(row.get("severity", 3)),
                            notes=row.get("notes", ""),
                            rater_id=row.get("rater_id", "import"),
                        )
                elif "start_date" in row and "end_date" in row:
                    # Date range episode
                    if not dry_run:
                        labeler.add_episode(
                            start_date=row["start_date"],
                            end_date=row["end_date"],
                            episode_type=row.get(
                                "mood", row.get("episode_type", "depressive")
                            ),
                            severity=int(row.get("severity", 3)),
                            notes=row.get("notes", ""),
                            rater_id=row.get("rater_id", "import"),
                        )
                else:
                    console.print(
                        "[yellow]Skipping row without date information[/yellow]"
                    )
                    skipped_count += 1
                    continue

                imported_count += 1

            except Exception as e:
                console.print(f"[red]Error importing row: {e}[/red]")
                skipped_count += 1

    # Show results
    console.print(f"\n[green]✓ Imported {imported_count} episodes[/green]")
    if skipped_count > 0:
        console.print(f"[yellow]⚠ Skipped {skipped_count} rows[/yellow]")

    # Save to database if specified
    if db and not dry_run:
        repo.save_labeler(labeler)
        console.print(f"[dim]Saved to database: {db}[/dim]")
    elif dry_run:
        console.print("[cyan]DRY RUN - No changes saved[/cyan]")
