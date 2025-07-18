"""
Label management CLI commands.

Beautiful CLI for managing mood and health labels using Typer and Rich.
Implements patterns from rich-cli, typer, and rich-click reference repos.
"""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from big_mood_detector.application.services.label_service import LabelService
from big_mood_detector.domain.entities.label import Label
from big_mood_detector.infrastructure.repositories.in_memory_label_repository import (
    InMemoryLabelRepository,
)

# Initialize dependencies
# In production, these would be injected via dependency injection
label_repository = InMemoryLabelRepository()
label_service = LabelService(label_repository)

# Create Typer app with help text
app = typer.Typer(
    name="label",
    help="Manage mood and health labels",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Initialize Rich console for beautiful output
console = Console()
error_console = Console(stderr=True, style="bold red")


def create_label_table(labels: list[Label]) -> Table:
    """Create a beautiful Rich table for displaying labels."""
    table = Table(
        title="Labels",
        show_header=True,
        header_style="bold magenta",
        title_style="bold blue",
    )

    table.add_column("ID", style="dim", width=12)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Color", justify="center")
    table.add_column("Description", style="dim")

    for label in labels:
        # Create a color swatch for the color column
        color_swatch = Text("███", style=label.color)

        table.add_row(
            label.id, label.name, label.category, color_swatch, label.description
        )

    return table


@app.command(name="list")
def list_labels(
    category: str | None = typer.Option(
        None, "--category", "-c", help="Filter labels by category"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table or json"
    ),
) -> None:
    """
    List all labels in a beautiful table format.

    Examples:
        label list
        label list --category mood
        label list --format json
    """
    try:
        # Get labels from service
        if category:
            labels = label_service.list_labels(category=category)
        else:
            labels = label_service.list_labels()

        # Handle empty results
        if not labels:
            console.print("[yellow]No labels found[/yellow]")
            return

        # Output based on format
        if format == "json":
            # JSON output for programmatic use
            labels_dict = [
                {
                    "id": label.id,
                    "name": label.name,
                    "description": label.description,
                    "color": label.color,
                    "category": label.category,
                    "metadata": label.metadata,
                }
                for label in labels
            ]
            console.print_json(data=labels_dict)
        else:
            # Beautiful table output
            table = create_label_table(labels)
            console.print(table)

    except Exception as e:
        error_console.print(f"Error listing labels: {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def create(
    name: str | None = typer.Option(None, "--name", "-n", help="Label name"),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Label description"
    ),
    category: str | None = typer.Option(
        None, "--category", "-c", help="Label category"
    ),
    color: str | None = typer.Option(None, "--color", help="Label color (hex format)"),
) -> None:
    """
    Create a new label interactively or with arguments.

    Examples:
        label create
        label create --name "Anxiety" --description "Anxiety indicators" --category mood
    """
    try:
        # Interactive prompts if arguments not provided
        if not name:
            name = Prompt.ask("[cyan]Label name[/cyan]")

        if not description:
            description = Prompt.ask("[cyan]Description[/cyan]")

        if not category:
            category = Prompt.ask(
                "[cyan]Category[/cyan]",
                choices=["mood", "sleep", "activity", "other"],
                default="mood",
            )

        if not color:
            color = Prompt.ask("[cyan]Color (hex)[/cyan]", default="#7F8C8D")

        # Create the label
        new_label = label_service.create_label(
            name=name,
            description=description,
            category=category,
            color=color,
            metadata={},
        )

        # Display success message with the label details
        console.print(
            Panel(
                f"[green]Label '{new_label.name}' created successfully![/green]\n\n"
                f"[dim]ID:[/dim] {new_label.id}\n"
                f"[dim]Category:[/dim] {new_label.category}\n"
                f"[dim]Color:[/dim] {Text('███', style=new_label.color)} {new_label.color}",
                title="✨ Label Created",
                border_style="green",
            )
        )

    except ValueError as e:
        error_console.print(f"Error: {str(e)}")
        raise typer.Exit(1) from e
    except Exception as e:
        error_console.print(f"Error creating label: {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show"),
) -> None:
    """
    Search labels by name or description.

    Examples:
        label search sleep
        label search "mood disorder" --limit 5
    """
    try:
        # Search labels
        results = label_service.search_labels(query, limit=limit)

        if not results:
            console.print(f"[yellow]No labels found matching '{query}'[/yellow]")
            return

        # Create table with search results
        table = Table(
            title=f"Search Results for '{query}'",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Name", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Description")

        for label in results:
            # Highlight search terms in the output
            highlighted_name = label.name
            highlighted_desc = label.description

            if query.lower() in label.name.lower():
                highlighted_name = label.name.replace(
                    query, f"[bold yellow]{query}[/bold yellow]"
                )

            if query.lower() in label.description.lower():
                highlighted_desc = label.description.replace(
                    query, f"[bold yellow]{query}[/bold yellow]"
                )

            table.add_row(
                highlighted_name,
                label.category,
                (
                    highlighted_desc[:60] + "..."
                    if len(highlighted_desc) > 60
                    else highlighted_desc
                ),
            )

        console.print(table)

    except Exception as e:
        error_console.print(f"Error searching labels: {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def delete(
    label_id: str = typer.Argument(..., help="Label ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """
    Delete a label with confirmation.

    Examples:
        label delete label-123
        label delete label-123 --force
    """
    try:
        # Get the label first to show details
        label = label_service.get_label(label_id)

        if not label:
            error_console.print(f"Label '{label_id}' not found")
            raise typer.Exit(1)

        # Show label details and confirm
        console.print(
            Panel(
                f"[yellow]Label Details:[/yellow]\n"
                f"Name: {label.name}\n"
                f"Category: {label.category}\n"
                f"Description: {label.description}",
                title="⚠️  Delete Label",
                border_style="yellow",
            )
        )

        # Confirm deletion unless --force is used
        if not force:
            confirmed = Confirm.ask(
                f"Are you sure you want to delete label '{label.name}'?", default=False
            )

            if not confirmed:
                console.print("[yellow]Deletion cancelled[/yellow]")
                return

        # Delete the label
        label_service.delete_label(label_id)
        console.print(f"[green]Label '{label.name}' deleted successfully[/green]")

    except ValueError as e:
        error_console.print(f"Error: {str(e)}")
        raise typer.Exit(1) from e
    except Exception as e:
        error_console.print(f"Error deleting label: {str(e)}")
        raise typer.Exit(1) from e


@app.command()
def update(
    label_id: str = typer.Argument(..., help="Label ID to update"),
    name: str | None = typer.Option(None, "--name", "-n", help="New name"),
    description: str | None = typer.Option(
        None, "--description", "-d", help="New description"
    ),
    category: str | None = typer.Option(None, "--category", "-c", help="New category"),
    color: str | None = typer.Option(None, "--color", help="New color (hex)"),
) -> None:
    """
    Update an existing label.

    Examples:
        label update label-123 --description "Updated description"
        label update label-123 --name "New Name" --color "#FF0000"
    """
    try:
        # Get current label
        label = label_service.get_label(label_id)

        if not label:
            error_console.print(f"Label '{label_id}' not found")
            raise typer.Exit(1)

        # Prepare update data
        update_data = {}
        if name:
            update_data["name"] = name
        if description:
            update_data["description"] = description
        if category:
            update_data["category"] = category
        if color:
            update_data["color"] = color

        if not update_data:
            console.print("[yellow]No updates specified[/yellow]")
            return

        # Update the label
        updated_label = label_service.update_label(
            label_id,
            name=update_data.get("name"),
            description=update_data.get("description"),
            category=update_data.get("category"),
            color=update_data.get("color"),
            metadata=(
                json.loads(update_data.get("metadata", "{}"))
                if "metadata" in update_data
                else None
            ),
        )

        # Show success message
        console.print(
            Panel(
                f"[green]Label '{updated_label.name}' updated successfully![/green]\n\n"
                f"[dim]Updated fields:[/dim] {', '.join(update_data.keys())}",
                title="✅ Label Updated",
                border_style="green",
            )
        )

        # Show updated label in a table
        table = create_label_table([updated_label])
        console.print(table)

    except Exception as e:
        error_console.print(f"Error updating label: {str(e)}")
        raise typer.Exit(1) from e


@app.command(name="import")
def import_labels(
    file_path: Path = typer.Argument(..., help="Path to JSON file with labels"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without importing"),
) -> None:
    """
    Import multiple labels from a JSON file with progress tracking.

    Example JSON format:
    [
        {
            "name": "Depression",
            "description": "Major depressive episode",
            "category": "mood",
            "color": "#5B6C8F"
        }
    ]

    Examples:
        label import labels.json
        label import labels.json --dry-run
    """
    try:
        # Read and parse the file
        with open(file_path) as f:
            labels_data = json.load(f)

        if not isinstance(labels_data, list):
            error_console.print("File must contain a JSON array of labels")
            raise typer.Exit(1)

        console.print(f"[cyan]Found {len(labels_data)} labels to import[/cyan]")

        if dry_run:
            console.print("[yellow]DRY RUN - No labels will be created[/yellow]")

        # Import labels with progress bar
        created_labels = []
        failed_labels = []

        for label_data in track(
            labels_data,
            description=f"Importing {len(labels_data)} labels...",
            console=console,
        ):
            try:
                if not dry_run:
                    label = label_service.create_label(
                        name=label_data.get("name", ""),
                        description=label_data.get("description", ""),
                        category=label_data.get("category", "other"),
                        color=label_data.get("color", "#7F8C8D"),
                        metadata=label_data.get("metadata", {}),
                    )
                    created_labels.append(label)
                else:
                    # Just validate in dry run
                    if not label_data.get("name"):
                        raise ValueError("Missing required field: name")
                    created_labels.append(label_data)

            except Exception as e:
                failed_labels.append({"data": label_data, "error": str(e)})

        # Show results
        if created_labels:
            console.print(
                f"[green]Successfully imported {len(created_labels)} labels[/green]"
            )

        if failed_labels:
            console.print(f"[red]Failed to import {len(failed_labels)} labels[/red]")
            for fail in failed_labels[:5]:  # Show first 5 failures
                console.print(
                    f"  - {fail['data'].get('name', 'Unknown')}: {fail['error']}"
                )

    except FileNotFoundError as e:
        error_console.print(f"File not found: {file_path}")
        raise typer.Exit(1) from e
    except json.JSONDecodeError as e:
        error_console.print(f"Invalid JSON file: {str(e)}")
        raise typer.Exit(1) from e
    except Exception as e:
        import traceback

        error_console.print(f"Error importing labels: {str(e)}")
        # Print full traceback for debugging
        if "--debug" in sys.argv:
            traceback.print_exc()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
