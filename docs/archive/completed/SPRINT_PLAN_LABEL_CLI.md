# Sprint Plan: Label CLI Implementation

## Current State Summary

### ✅ What's Working
- 700+ tests passing (CLI watcher hang fixed)
- Type checking & linting clean (mypy, ruff, black)
- Unified CLI under interfaces/cli
- Repository pattern implemented with File*Repository classes
- IoC container working with proper DI

### ⚠️ Technical Debt
- `core/` directory still contains infrastructure files
- Deprecated `cli/` directory needs removal
- No Docker containerization
- Long test runtime (>5 min)

### 🚧 Missing Features
- Label CLI not implemented
- No E2E trace from label files to model fine-tuning
- FastAPI only has skeleton endpoints
- No episode CRUD, auth, CORS, or Swagger docs

## Sprint Decision: Label CLI First (Data-First Approach)

**Rationale**: Get real episode data flowing immediately to unblock model tuning, then harden API/containerization.

## Implementation Roadmap

### Day 0: Infrastructure Cleanup (Immediate)

```bash
# 1. Move infrastructure files
git mv src/big_mood_detector/core/dependencies.py src/big_mood_detector/infrastructure/di/
git mv src/big_mood_detector/core/config.py src/big_mood_detector/infrastructure/settings/
git mv src/big_mood_detector/core/logging.py src/big_mood_detector/infrastructure/logging/

# 2. Create deprecation shim
cat > src/big_mood_detector/core/__init__.py << 'EOF'
"""Deprecated module - use infrastructure.* instead."""
import warnings
warnings.warn(
    "core.* modules are deprecated. Import from infrastructure.* instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility (temporary)
from big_mood_detector.infrastructure.di.dependencies import *
from big_mood_detector.infrastructure.settings.config import *
from big_mood_detector.infrastructure.logging import *
EOF

# 3. Delete old CLI directory
git rm -rf src/big_mood_detector/cli/

# 4. Fix imports and verify
ruff check src --fix
make quality
```

### Day 1-2: Label CLI Core Implementation

#### Command Structure
```python
# src/big_mood_detector/interfaces/cli/labeling/commands.py

@click.group(name="label", invoke_without_command=True)
@click.pass_context
def label_group(ctx):
    """Create ground truth labels for model training."""
    if ctx.invoked_subcommand is None:
        # Default to episode subcommand
        ctx.invoke(label_episode_command)

@label_group.command(name="episode")
@click.option("--predictions", type=click.Path(exists=True), 
              help="Predictions file for assisted labeling")
@click.option("--date", type=click.DateTime(formats=["%Y-%m-%d"]),
              help="Single date to label")
@click.option("--date-range", type=str,
              help="Date range YYYY-MM-DD:YYYY-MM-DD")
@click.option("--interactive/--batch", default=True,
              help="Interactive prompts vs batch mode")
@click.option("--dry-run", is_flag=True,
              help="Preview without saving")
def label_episode_command(predictions, date, date_range, interactive, dry_run):
    """Label mood episodes with clinical validation."""
    pass

@label_group.command(name="baseline")
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--end", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--notes", type=str)
def label_baseline_command(start, end, notes):
    """Mark stable baseline periods."""
    pass

@label_group.command(name="undo")
def label_undo_command():
    """Undo the last label entry."""
    pass
```

#### TDD Test Cases First
```python
# tests/unit/interfaces/cli/test_labeling.py

class TestLabelingCLI:
    def test_label_single_day_depression(self, runner, mock_labeler):
        """Test labeling a single day as depressed."""
        result = runner.invoke(
            cli, 
            ["label", "--date", "2024-03-15", "--batch"],
            input="depressive\n3\n"
        )
        assert result.exit_code == 0
        assert "Labeled 2024-03-15 as depressive" in result.output
        mock_labeler.add_episode.assert_called_once()
    
    def test_label_episode_range(self, runner, mock_labeler):
        """Test labeling multi-day episode."""
        result = runner.invoke(
            cli,
            ["label", "--date-range", "2024-03-10:2024-03-20", "--batch"],
            input="hypomanic\n3\n"
        )
        assert result.exit_code == 0
        assert "Labeled 11-day hypomanic episode" in result.output
    
    def test_dsm5_duration_warning(self, runner):
        """Test DSM-5 validation warnings."""
        result = runner.invoke(
            cli,
            ["label", "--date-range", "2024-03-10:2024-03-12", "--batch"],
            input="manic\n4\ny\n"  # Too short, confirm anyway
        )
        assert "Manic episodes require ≥7 days" in result.output
    
    def test_interactive_mode_with_predictions(self, runner):
        """Test prediction-assisted interactive flow."""
        # More complex test with mocked predictions
        pass
```

### Day 3-4: Interactive UI & HIG Compliance

#### Human Interface Guidelines Implementation
```python
# src/big_mood_detector/interfaces/cli/labeling/interactive.py

class InteractiveSession:
    def __init__(self, use_case: LabelEpisodesUseCase):
        self.use_case = use_case
        self.last_dates = {}  # Remember previous inputs
        
    def prompt_with_default(self, prompt: str, default: Any = None) -> str:
        """Prompt with default value (Enter to accept)."""
        if default:
            prompt += f" [{click.style(str(default), fg='cyan')}]"
        
        value = click.prompt(prompt, default=default or "", show_default=False)
        return value if value else default
    
    def show_day_context(self, day_info: DayToLabel) -> None:
        """Display predictions and biomarkers with color coding."""
        # Risk levels with color
        risk_color = "red" if day_info.depression_risk > 0.7 else "yellow"
        risk_label = "HIGH" if day_info.depression_risk > 0.7 else "MODERATE"
        
        click.echo(f"\n{'─' * 60}")
        click.echo(f"Date: {click.style(day_info.date.strftime('%Y-%m-%d (%A)'), bold=True)}")
        click.echo(f"\nModel Predictions:")
        click.echo(f"  • Depression Risk: {click.style(f'{day_info.depression_risk:.0%} [{risk_label}]', fg=risk_color)}")
        # ... more formatted output
```

#### Smart Defaults & Autocompletion
```python
# Case-insensitive mood mapping
MOOD_ALIASES = {
    'dep': 'depressive', 'depression': 'depressive', 'd': 'depressive',
    'hypo': 'hypomanic', 'hypomanic': 'hypomanic', 'h': 'hypomanic',
    'mania': 'manic', 'manic': 'manic', 'm': 'manic',
    'mixed': 'mixed', 'mix': 'mixed', 'x': 'mixed',
    'stable': 'baseline', 'normal': 'baseline', 'none': 'baseline', 'n': 'baseline',
}

def get_mood_choice(self) -> str:
    """Get mood with aliases and validation."""
    mood_completer = WordCompleter(
        list(MOOD_ALIASES.keys()), 
        ignore_case=True
    )
    
    while True:
        mood = prompt("Mood state: ", completer=mood_completer)
        normalized = MOOD_ALIASES.get(mood.lower())
        if normalized:
            return normalized
        click.echo(click.style("Invalid mood. Try: dep, hypo, manic, mixed, or stable", fg="red"))
```

### Day 5: BDD/E2E Testing

#### Gherkin Specifications
```gherkin
# tests/bdd/features/labeling.feature

Feature: Mood Episode Labeling
  As a clinician
  I want to label mood episodes efficiently
  So that I can train personalized models

  Scenario: Label depressive episode with validation
    Given I have predictions for March 2024
    When I run "label --date-range 2024-03-10:2024-03-20"
    And I enter "depressive" for mood type
    And I enter "3" for severity
    Then I should see "11-day depressive episode labeled"
    And the CSV should contain 11 daily labels

  Scenario: Reject invalid episode duration
    Given I am labeling episodes
    When I try to label a 2-day manic episode
    Then I should see "Manic episodes require ≥7 days (DSM-5)"
    And I should be prompted to confirm or adjust

  Scenario: End-to-end calibration
    Given I have Apple Health data and labeled episodes
    When I run "calibrate --episodes labels/episodes.csv"
    Then the model should train without errors
    And I should see "Calibrated on 30 labeled days"
```

### Day 6: Integration & Polish

#### Progress Tracking
```python
class ProgressTracker:
    def __init__(self, total_days: int):
        self.total = total_days
        self.completed = 0
        self.episodes = defaultdict(int)
        
    def update(self, label: str) -> None:
        self.completed += 1
        self.episodes[label] += 1
        
    def display(self) -> None:
        """Show progress bar and statistics."""
        progress_bar = click.progressbar(
            length=self.total,
            label='Labeling progress',
            show_percent=True,
            show_pos=True
        )
        progress_bar.update(self.completed)
        
        # Summary stats
        click.echo(f"\nEpisodes identified:")
        for label, count in self.episodes.items():
            icon = "🔴" if label == "depressive" else "🟡" if label == "hypomanic" else "🟢"
            click.echo(f"  {icon} {label.title()}: {count} days")
```

#### Auto-save & Resume
```python
class LabelingSession:
    def __init__(self, session_file: Path = Path(".labeling_session.json")):
        self.session_file = session_file
        self.state = self._load_or_create_session()
        
    def autosave(self) -> None:
        """Save session state every 10 labels."""
        if self.state['labels_this_session'] % 10 == 0:
            self._save_session()
            click.echo(click.style("✓ Progress saved", fg="green", dim=True))
```

## Definition of Done

### Functionality
- [ ] All existing tests pass (700+)
- [ ] New label CLI tests >90% coverage
- [ ] E2E test: label → calibrate → model trains

### Usability  
- [ ] Label 30 days in <3 minutes
- [ ] No unhandled exceptions
- [ ] Clear validation messages
- [ ] Progress tracking works

### Integration
- [ ] CSV format compatible with calibrate command
- [ ] Git-friendly JSON/CSV output
- [ ] Deprecation warnings for moved files
- [ ] Updated README quickstart

### Performance
- [ ] Interactive response <100ms
- [ ] File I/O optimized for large datasets
- [ ] Memory efficient for year+ of data

## Next Steps After Label CLI

1. **API Hardening** (Days 7-8)
   - Episode CRUD endpoints
   - OpenAPI documentation
   - CORS configuration

2. **Containerization** (Days 9-10)
   - Multi-stage Dockerfile
   - docker-compose setup
   - CI/CD pipeline

3. **Multi-rater Support**
   - Database for labels
   - Inter-rater reliability metrics
   - Conflict resolution UI

## Implementation Start

Ready to begin with infrastructure cleanup, then TDD implementation of the label CLI following this plan.