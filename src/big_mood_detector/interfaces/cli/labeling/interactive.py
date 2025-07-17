"""
Interactive Labeling Session

Provides interactive UI for prediction-assisted labeling.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

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
        """Display predictions and biomarkers."""
        click.echo(f"\n{'─' * 60}")
        click.echo(f"Date: {click.style(prediction['date'], bold=True)}")
        click.echo("\nModel Predictions:")
        
        # Risk levels
        dep_risk = prediction.get("depression_risk", 0)
        risk_color = "red" if dep_risk > 0.6 else "yellow" if dep_risk > 0.4 else "green"
        click.echo(f"  • Depression Risk: {click.style(f'{dep_risk:.0%}', fg=risk_color)}")
        click.echo(f"  • Hypomanic Risk: {prediction.get('hypomanic_risk', 0):.0%}")
        click.echo(f"  • Manic Risk: {prediction.get('manic_risk', 0):.0%}")
        
        # Features if available
        if "features" in prediction:
            features = prediction["features"]
            click.echo("\nDigital Biomarkers:")
            if "sleep_hours" in features:
                click.echo(f"  • Sleep: {features['sleep_hours']:.1f} hrs")
            if "activity_steps" in features:
                click.echo(f"  • Activity: {features['activity_steps']:,} steps")
            if "sleep_efficiency" in features:
                click.echo(f"  • Sleep Efficiency: {features['sleep_efficiency']:.0%}")
    
    def _prompt_mood(self) -> str:
        """Prompt for mood type."""
        click.echo("\nWhat was the mood state on this day?")
        click.echo("  [1] Depressed")
        click.echo("  [2] Hypomanic") 
        click.echo("  [3] Manic")
        click.echo("  [4] Mixed")
        click.echo("  [5] Stable/Normal")
        click.echo("  [6] Skip")
        
        choice = click.prompt("Choice (1-6)", type=int)
        
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