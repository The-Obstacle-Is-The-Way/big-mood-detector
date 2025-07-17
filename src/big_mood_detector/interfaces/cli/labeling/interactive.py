"""
Interactive Labeling Session

Provides interactive UI for prediction-assisted labeling.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
from big_mood_detector.infrastructure.logging import get_module_logger

from .validators import ClinicalValidator

logger = get_module_logger(__name__)


class InteractiveSession:
    """Interactive labeling session with predictions."""
    
    def __init__(
        self, 
        labeler: EpisodeLabeler,
        validator: ClinicalValidator,
        predictions_file: Optional[Path] = None
    ):
        self.labeler = labeler
        self.validator = validator
        self.predictions = self._load_predictions(predictions_file) if predictions_file else []
        self.progress = {"labeled": 0, "skipped": 0, "total": len(self.predictions)}
        self.console = Console()
    
    def _load_predictions(self, predictions_file: Path) -> List[Dict[str, Any]]:
        """Load predictions from JSON file."""
        with open(predictions_file) as f:
            data = json.load(f)
            return data.get("predictions", [])
    
    def run(self) -> None:
        """Run the interactive labeling session."""
        click.echo(click.style(
            "\n╔══════════════════════════════════════════════════════════════╗",
            fg="bright_blue"
        ))
        click.echo(click.style(
            "║                  Big Mood Detector Labeling Tool              ║",
            fg="bright_blue"
        ))
        click.echo(click.style(
            "╚══════════════════════════════════════════════════════════════╝\n",
            fg="bright_blue"
        ))
        
        click.echo(f"Found {len(self.predictions)} days with predictions")
        
        for pred in self.predictions:
            self._process_day(pred)
            
        self._show_summary()
    
    def _process_day(self, prediction: Dict[str, Any]) -> None:
        """Process a single day's labeling."""
        date_str = prediction["date"]
        day_date = date.fromisoformat(date_str)
        
        # Display prediction context
        self._display_day_context(prediction)
        
        # Get user input
        mood_choice = self._prompt_mood()
        if mood_choice == "skip":
            self.progress["skipped"] += 1
            return
        
        severity = self._prompt_severity()
        
        # Check for episode span
        if self._prompt_episode_span():
            start_date, end_date = self._get_episode_dates(day_date)
        else:
            start_date = end_date = day_date
        
        # Add label
        self.labeler.add_episode(
            start_date=start_date,
            end_date=end_date,
            episode_type=mood_choice,
            severity=severity
        )
        
        self.progress["labeled"] += 1
        click.echo(click.style("✓ Label saved", fg="green"))
    
    def _display_day_context(self, prediction: Dict[str, Any]) -> None:
        """Display predictions and biomarkers with Rich formatting."""
        # Create a table for predictions
        pred_table = Table(title="Model Predictions", show_header=False, padding=(0, 1))
        pred_table.add_column("Metric", style="cyan")
        pred_table.add_column("Value", justify="right")
        
        # Risk levels with color coding
        dep_risk = prediction.get("depression_risk", 0)
        dep_color = "red" if dep_risk > 0.6 else "yellow" if dep_risk > 0.4 else "green"
        pred_table.add_row("Depression Risk", f"[{dep_color}]{dep_risk:.0%}[/{dep_color}]")
        
        hypo_risk = prediction.get("hypomanic_risk", 0)
        hypo_color = "yellow" if hypo_risk > 0.4 else "green"
        pred_table.add_row("Hypomanic Risk", f"[{hypo_color}]{hypo_risk:.0%}[/{hypo_color}]")
        
        manic_risk = prediction.get("manic_risk", 0)
        manic_color = "red" if manic_risk > 0.4 else "green"
        pred_table.add_row("Manic Risk", f"[{manic_color}]{manic_risk:.0%}[/{manic_color}]")
        
        # Create biomarkers table if available
        if "features" in prediction:
            features = prediction["features"]
            bio_table = Table(title="Digital Biomarkers", show_header=False, padding=(0, 1))
            bio_table.add_column("Metric", style="cyan")
            bio_table.add_column("Value", justify="right")
            
            if "sleep_hours" in features:
                bio_table.add_row("Sleep", f"{features['sleep_hours']:.1f} hrs")
            if "activity_steps" in features:
                bio_table.add_row("Activity", f"{features['activity_steps']:,} steps")
            if "sleep_efficiency" in features:
                eff = features['sleep_efficiency']
                eff_color = "red" if eff < 0.7 else "yellow" if eff < 0.85 else "green"
                bio_table.add_row("Sleep Efficiency", f"[{eff_color}]{eff:.0%}[/{eff_color}]")
        
        # Create main panel
        date_str = prediction['date']
        day_num = self.progress['labeled'] + self.progress['skipped'] + 1
        total = self.progress['total']
        
        panel = Panel(
            f"[bold]Date: {date_str}[/bold]\n[dim]Day {day_num} of {total}[/dim]\n",
            title=f"[bold blue]Labeling Session[/bold blue]",
            border_style="blue"
        )
        
        self.console.print(panel)
        self.console.print(pred_table)
        if "features" in prediction:
            self.console.print(bio_table)
    
    def _prompt_mood(self) -> str:
        """Prompt for mood type with Rich formatting."""
        choices = [
            "[red]1[/red] - Depressed",
            "[yellow]2[/yellow] - Hypomanic",
            "[magenta]3[/magenta] - Manic", 
            "[cyan]4[/cyan] - Mixed",
            "[green]5[/green] - Stable/Normal",
            "[dim]6[/dim] - Skip"
        ]
        
        self.console.print("\n[bold]What was the mood state on this day?[/bold]")
        for choice in choices:
            self.console.print(f"  {choice}")
        
        # Use regular click prompt instead of IntPrompt which has issues with choices
        choice = click.prompt("Choice", type=click.IntRange(1, 6))
        
        mood_map = {
            1: "depressive",
            2: "hypomanic",
            3: "manic",
            4: "mixed",
            5: "baseline",
            6: "skip"
        }
        
        return mood_map.get(choice, "skip")
    
    def _prompt_severity(self) -> int:
        """Prompt for severity."""
        return click.prompt("Severity (1-5)", type=int, default=3)
    
    def _prompt_episode_span(self) -> bool:
        """Ask if part of longer episode."""
        return click.confirm("Part of a longer episode?", default=False)
    
    def _get_episode_dates(self, current_date: date) -> tuple[date, date]:
        """Get episode start and end dates."""
        start_str = click.prompt("Episode start date (YYYY-MM-DD)", 
                                default=str(current_date))
        end_str = click.prompt("Episode end date (YYYY-MM-DD)",
                              default=str(current_date))
        
        start_date = date.fromisoformat(start_str)
        end_date = date.fromisoformat(end_str)
        
        return start_date, end_date
    
    def _show_summary(self) -> None:
        """Show labeling summary."""
        click.echo(f"\n{'═' * 60}")
        click.echo("Labeling Complete!")
        click.echo(f"  • Labeled: {self.progress['labeled']} days")
        click.echo(f"  • Skipped: {self.progress['skipped']} days")
        click.echo(f"  • Total: {self.progress['total']} days")