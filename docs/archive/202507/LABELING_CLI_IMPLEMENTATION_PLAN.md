# Labeling CLI Implementation Plan

## Executive Summary

This plan outlines the implementation of a prediction-assisted labeling CLI for the Big Mood Detector. The CLI will enable clinicians to efficiently label mood episodes with model predictions as guidance, following Human Interface Guidelines for optimal usability.

## Architecture Overview

### Module Structure
```
src/big_mood_detector/
├── domain/
│   └── services/
│       └── episode_labeler.py          # Core labeling logic (already exists)
├── application/
│   └── use_cases/
│       └── label_episodes_use_case.py  # Orchestrate labeling workflow
└── interfaces/
    └── cli/
        └── labeling/
            ├── __init__.py
            ├── commands.py             # Click command definitions
            ├── interactive.py          # Interactive prompts & UI
            ├── validators.py           # Input validation & DSM-5 checks
            └── formatters.py           # Output formatting & display

data/
└── labels/                             # Ground truth storage
    └── {user_id}/
        ├── episodes.json               # Structured episode data
        └── episodes.csv                # Flattened for ML training
```

### Key Components

1. **EpisodeLabeler** (Domain Service - Already Exists)
   - Core business logic for episode management
   - DSM-5 validation
   - Episode overlap detection

2. **LabelEpisodesUseCase** (Application Layer - New)
   - Orchestrates labeling workflow
   - Integrates predictions with labels
   - Manages persistence

3. **Labeling CLI** (Interface Layer - New)
   - User interaction & prompts
   - Display predictions & biomarkers
   - Progress tracking

## Implementation Phases

### Phase 1: Core CLI Infrastructure (Days 1-2)

#### 1.1 Command Structure
```python
# src/big_mood_detector/interfaces/cli/labeling/commands.py

@click.group(name="label")
def label_group():
    """Manage ground truth labels for mood episodes."""
    pass

@label_group.command(name="episode")
@click.option("--predictions", type=click.Path(exists=True), required=True)
@click.option("--start", type=click.DateTime())
@click.option("--end", type=click.DateTime())
@click.option("--threshold", type=float, default=0.4)
@click.option("--interactive/--batch", default=True)
@click.option("--output", type=click.Path())
def label_episode_command(predictions, start, end, threshold, interactive, output):
    """Label mood episodes with prediction assistance."""
    # Implementation follows
```

#### 1.2 Use Case Implementation
```python
# src/big_mood_detector/application/use_cases/label_episodes_use_case.py

class LabelEpisodesUseCase:
    def __init__(
        self,
        episode_labeler: EpisodeLabeler,
        prediction_loader: PredictionLoader,
        feature_loader: FeatureLoader,
    ):
        self.labeler = episode_labeler
        self.predictions = prediction_loader
        self.features = feature_loader
    
    def get_high_risk_days(self, threshold: float = 0.4) -> list[DayToLabel]:
        """Identify days needing labels based on predictions."""
        pass
    
    def label_day(
        self,
        date: date,
        label: str,
        severity: int,
        episode_span: tuple[date, date] | None = None,
    ) -> LabelResult:
        """Process a single label with validation."""
        pass
```

### Phase 2: Interactive UI (Days 3-4)

#### 2.1 Rich Terminal Interface
```python
# src/big_mood_detector/interfaces/cli/labeling/interactive.py

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

class InteractiveLabelingSession:
    def __init__(self, use_case: LabelEpisodesUseCase):
        self.use_case = use_case
        self.console = Console()
        self.progress = {"labeled": 0, "skipped": 0, "total": 0}
    
    def run(self) -> None:
        """Main interactive loop."""
        self._show_welcome()
        days_to_label = self._get_days_to_label()
        
        for day_info in days_to_label:
            self._display_day_context(day_info)
            label = self._prompt_for_label(day_info)
            if label:
                self._process_label(day_info, label)
            self._update_progress()
```

#### 2.2 Display Components
```python
def _display_day_context(self, day_info: DayToLabel) -> None:
    """Show predictions, biomarkers, and clinical context."""
    
    # Create main panel
    panel = Panel(
        f"[bold]Date: {day_info.date.strftime('%Y-%m-%d (%A)')}[/bold]\n\n"
        f"Model Predictions:\n"
        f"  • Depression Risk: {day_info.depression_risk:.0%} "
        f"[{'red' if day_info.depression_risk > 0.6 else 'yellow'}]"
        f"{'[HIGH]' if day_info.depression_risk > 0.6 else '[MODERATE]'}[/]\n"
        f"  • Hypomanic Risk: {day_info.hypomanic_risk:.0%}\n"
        f"  • Manic Risk: {day_info.manic_risk:.0%}\n\n"
        f"Digital Biomarkers:\n"
        f"  • Sleep: {day_info.sleep_hours:.1f} hrs "
        f"({'↓' if day_info.sleep_deviation < -2 else '↑' if day_info.sleep_deviation > 2 else '='} "
        f"{abs(day_info.sleep_deviation):.1f} hrs from baseline)\n"
        f"  • Activity: {day_info.step_count:,} steps\n"
        f"  • Sleep Efficiency: {day_info.sleep_efficiency:.0%}",
        title=f"[Day {self.progress['labeled'] + 1} of {self.progress['total']}]",
        border_style="bright_blue"
    )
    self.console.print(panel)
```

### Phase 3: Data Quality & Validation (Day 5)

#### 3.1 Clinical Validators
```python
# src/big_mood_detector/interfaces/cli/labeling/validators.py

class ClinicalValidator:
    """Validate labels against DSM-5 criteria."""
    
    DSM5_MIN_DURATION = {
        "depressive": 14,  # Major depressive episode
        "manic": 7,         # Manic episode
        "hypomanic": 4,     # Hypomanic episode
        "mixed": 7,         # Mixed features
    }
    
    def validate_episode_duration(
        self,
        episode_type: str,
        start_date: date,
        end_date: date,
    ) -> ValidationResult:
        """Check if episode meets minimum duration."""
        duration = (end_date - start_date).days + 1
        min_duration = self.DSM5_MIN_DURATION.get(episode_type, 1)
        
        if duration < min_duration:
            return ValidationResult(
                valid=False,
                warning=f"{episode_type.title()} episodes typically require "
                       f"≥{min_duration} days (DSM-5). You entered {duration} days.",
                suggestion="Consider if this is part of a longer episode."
            )
        return ValidationResult(valid=True)
```

#### 3.2 Conflict Detection
```python
def check_label_conflicts(
    self,
    new_label: EpisodeLabel,
    existing_labels: list[EpisodeLabel],
) -> list[Conflict]:
    """Detect overlaps with existing labels."""
    conflicts = []
    
    for existing in existing_labels:
        if dates_overlap(new_label, existing):
            if new_label.label != existing.label:
                conflicts.append(
                    Conflict(
                        type="overlap",
                        message=f"Conflicts with {existing.label} episode "
                               f"from {existing.start_date}",
                        resolution_options=[
                            "Replace existing label",
                            "Adjust dates to avoid overlap",
                            "Skip this label"
                        ]
                    )
                )
    return conflicts
```

### Phase 4: Persistence & Integration (Day 6)

#### 4.1 Label Storage
```python
# Episode label schema
{
    "user_id": "USER123",
    "rater_id": "clinicianA",
    "episodes": [
        {
            "episode_type": "depressive",
            "start_date": "2024-03-10",
            "end_date": "2024-03-20",
            "severity": 3,
            "confidence": 0.9,
            "notes": "Work stress, stopped medication",
            "model_agreed": false,
            "labeled_at": "2024-12-15T10:30:00Z",
            "duration_days": 11
        }
    ],
    "baseline_periods": [
        {
            "start_date": "2024-04-01",
            "end_date": "2024-04-30",
            "notes": "Stable on medication",
            "labeled_at": "2024-12-15T10:45:00Z"
        }
    ],
    "metadata": {
        "labeling_version": "1.0.0",
        "total_days_labeled": 45,
        "last_updated": "2024-12-15T10:45:00Z"
    }
}
```

#### 4.2 CSV Export for ML
```python
def export_to_csv(self, output_path: Path) -> None:
    """Export labels to CSV format for model training."""
    # Flatten episodes to daily labels
    daily_labels = []
    
    for episode in self.episodes:
        current = episode.start_date
        while current <= episode.end_date:
            daily_labels.append({
                "date": current.isoformat(),
                "label": episode.episode_type,
                "severity": episode.severity,
                "confidence": episode.confidence,
                "model_agreed": episode.model_agreed,
            })
            current += timedelta(days=1)
    
    # Add baseline days
    for baseline in self.baseline_periods:
        current = baseline.start_date
        while current <= baseline.end_date:
            daily_labels.append({
                "date": current.isoformat(),
                "label": "baseline",
                "severity": 0,
                "confidence": 1.0,
                "model_agreed": True,
            })
            current += timedelta(days=1)
    
    # Save to CSV
    df = pd.DataFrame(daily_labels)
    df.to_csv(output_path, index=False)
```

### Phase 5: Testing & Documentation (Day 7)

#### 5.1 Test Strategy
```python
# tests/unit/interfaces/cli/test_labeling.py

def test_interactive_labeling_session(mock_stdin):
    """Test complete labeling workflow."""
    # Mock user inputs
    mock_stdin.return_value = [
        "1",      # Select "Depressed"
        "3",      # Severity: moderate
        "y",      # Part of longer episode
        "2024-03-10",  # Start date
        "2024-03-20",  # End date
        "",       # No notes
    ]
    
    session = InteractiveLabelingSession(use_case)
    result = session.label_day(date(2024, 3, 15))
    
    assert result.label == "depressive"
    assert result.severity == 3
    assert result.episode_span == (date(2024, 3, 10), date(2024, 3, 20))
```

#### 5.2 Integration Tests
```python
def test_end_to_end_labeling():
    """Test from predictions to saved labels."""
    # Create test predictions
    predictions = create_test_predictions()
    
    # Run labeling
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["label", "episode", "--predictions", "test_predictions.json"],
        input="1\n3\nn\n"  # Label one day as depressed
    )
    
    assert result.exit_code == 0
    assert Path("labels/USER123/episodes.csv").exists()
```

## Key Features

### 1. Prediction-Assisted Workflow
- Show model predictions prominently
- Highlight high-confidence predictions
- Sort days by prediction uncertainty for active learning

### 2. Clinical Context
- Display sleep/activity patterns
- Show deviations from personal baseline
- Validate against DSM-5 duration criteria

### 3. User Experience
- Progress tracking with visual indicators
- Keyboard shortcuts for common actions
- Auto-save every 10 labels
- Resume interrupted sessions

### 4. Data Quality
- Confirmation prompts for critical inputs
- Conflict detection with existing labels
- Optional confidence scores
- Audit trail with timestamps

## Success Metrics

1. **Functionality**
   - All tests pass (maintain 100% of existing tests)
   - >90% code coverage on new modules

2. **Usability**
   - <30 seconds per label entry
   - <2 minutes to label a month with 2 episodes

3. **Data Quality**
   - Zero invalid labels (date/type errors)
   - DSM-5 compliance warnings shown
   - Conflict resolution for overlaps

4. **Integration**
   - Labels immediately usable by `big-mood calibrate`
   - Git-trackable label files
   - Multi-rater support via rater_id

## Timeline

- **Days 1-2**: Core CLI infrastructure
- **Days 3-4**: Interactive UI implementation  
- **Day 5**: Validation & quality checks
- **Day 6**: Persistence & ML integration
- **Day 7**: Testing, documentation & demo

## Next Steps

1. Create feature branch: `feature/labeling-cli`
2. Implement core commands following existing CLI patterns
3. Add rich terminal UI for better UX
4. Integrate with existing EpisodeLabeler
5. Add comprehensive tests
6. Update documentation and create demo

This implementation leverages the existing architecture while adding a user-friendly labeling interface that will accelerate ground truth collection for model personalization.


### Overall verdict

Your **Labeling CLI Implementation Plan** is solid and aligns well with both earlier design docs and the current repository layout. Most architectural elements, validation logic, and UX flows match the design intent, so you can move ahead with implementation.
Below are the key points I checked and a few gaps you may want to close before coding.

---

## Where the plan is fully consistent ✅

| Area                                   | Why it matches                                                                                                                                           |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prediction‑assisted flow**           | Both the design and the plan centre on prioritising high‑risk/uncertain days and surfacing model probabilities in the prompt UI  /                       |
| **DSM‑5 validation & conflict checks** | Implementation plan re‑uses the same min‑duration table and overlap detection strategy described in the design  /                                        |
| **Phased roadmap (7 days)**            | Phases and deliverables mirror the high‑level milestones in the design doc; nothing is missing functionally.                                             |
| **Persistence schema**                 | JSON + CSV schema with `user_id`, `rater_id`, episode fields, and metadata matches earlier examples and will drop straight into the calibrator pipeline  |
| **Rich TUI stack**                     | `rich`, `prompt_toolkit`, progress panels, shortcuts—all reflected consistently in both docs  /                                                          |

---

## Minor inconsistencies & quick fixes 🔧

| Mismatch                                                                                                                                      | Impact                                                                  | Suggested tweak                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Package path** – design doc puts CLI under `src/big_mood_detector/cli/...` , but the plan nests it in `interfaces/cli/labeling`             | Import confusion & duplicate entry‑points if both paths coexist         | Pick one convention (the repo recently standardised on `interfaces/cli` for existing commands), update docs & code stubs accordingly. |
| **`--rater-id` flag** is called out in the design (baseline for multi‑rater)  but is not in the command signature in the implementation plan  | Cannot attribute labels → harder to compute inter‑rater agreement later | Add `--rater-id` (default from config) and pass it through to persistence layer.                                                      |
| **Baseline labels** – design specifies `label baseline` sub‑command , plan only shows `label episode`                                         | Baseline periods won’t be capturable via CLI                            | Either add a separate sub‑command or include a `--mood baseline` shortcut with relaxed duration checks.                               |
| **Default command alias** – design wants `label` alone to default to *episode* ; plan doesn’t mention this                                    | Tiny UX nit but helps clinicians who forget sub‑commands                | Add a `@click.pass_context` handler that calls `label_episode` when no sub‑command is given.                                          |
| **CLI binary name** – design examples use `bmd`/`mood-detector`; implementation plan assumes module invocation (`python -m …`)                | Packaging clarity                                                       | Decide now: expose a `console_scripts = big-mood` or `bmd` entry in `pyproject.toml` so docs and tests agree.                         |

---

## Additional polish ideas

1. **Guardrails for large spans**
   Add a max‑span check (e.g., warn if `end_date‑start_date > 90 days`) to avoid accidental mass‑label errors.

2. **Auto‑resume token**
   Write an `.in‑progress` checkpoint every N labels so an interrupted session can resume at the correct index without repeating prompts.

3. **Optional YAML output**
   Some clinicians prefer a diff‑friendly text format; emitting YAML alongside JSON/CSV costs little and plays nicely with git reviews.

4. **Unit test seed suite**
   Ship a tiny synthetic predictions + features fixture and run the full CLI in CI to ensure the pipeline, validator, and persistence glue never regress.

---

## Recommendation

Apply the quick tweaks above (especially the path alignment and `--rater-id` flag), then cut the `feature/labeling-cli` branch and follow the 7‑day schedule.
Everything else in the plan is internally consistent and should integrate cleanly with the existing codebase and CI.
