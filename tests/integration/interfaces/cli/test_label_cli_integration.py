"""
Integration Tests for Label CLI

These tests use real objects (no mocks) to ensure the CLI works correctly
with actual domain services and validators.
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner


class TestLabelCLIIntegration:
    """Integration tests for label CLI with real services."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_labels_file(self):
        """Create a temporary labels file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"labels": []}, f)
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink(missing_ok=True)

    def test_label_episode_real_validation(self, runner):
        """Test labeling with real DSM-5 validation."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Test that single-day depressive episode triggers warning
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-15", "--mood", "depressive", 
             "--severity", "3", "--no-interactive"],
        )
        
        # Should abort due to DSM-5 validation
        assert result.exit_code == 1
        assert "Depressive episodes require ≥14 days" in result.output
        
        # Test with valid duration should work
        result = runner.invoke(
            cli,
            ["label", "episode", "--date-range", "2024-03-01:2024-03-20", 
             "--mood", "depressive", "--severity", "3", "--dry-run"],
        )
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Would label" in result.output

    def test_label_baseline_real_flow(self, runner):
        """Test baseline labeling with real services."""
        from big_mood_detector.interfaces.cli.main import cli
        
        result = runner.invoke(
            cli,
            ["label", "baseline", "--start", "2024-05-01", "--end", "2024-05-14",
             "--notes", "Stable on medication", "--dry-run"],
        )
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Would mark baseline" in result.output

    def test_label_export_csv(self, runner, tmp_path):
        """Test CSV export functionality."""
        from big_mood_detector.interfaces.cli.main import cli
        
        output_file = tmp_path / "labels.csv"
        
        # Label a valid hypomanic episode (4+ days)
        result = runner.invoke(
            cli,
            ["label", "episode", "--date-range", "2024-03-10:2024-03-14",
             "--mood", "hypomanic", "--severity", "3", "--output", str(output_file),
             "--dry-run"],
        )
        
        assert result.exit_code == 0
        # In dry-run mode, CSV won't be created
        assert not output_file.exists()

    def test_interactive_predictions_file(self, runner, tmp_path):
        """Test interactive mode with predictions file."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Create predictions file
        predictions_file = tmp_path / "predictions.json"
        predictions_file.write_text(json.dumps({
            "predictions": [
                {
                    "date": "2024-03-15",
                    "depression_risk": 0.72,
                    "hypomanic_risk": 0.12,
                    "manic_risk": 0.08,
                    "features": {
                        "sleep_hours": 3.2,
                        "activity_steps": 2341,
                        "sleep_efficiency": 0.42
                    }
                }
            ]
        }))
        
        result = runner.invoke(
            cli,
            ["label", "episode", "--predictions", str(predictions_file), 
             "--interactive"],
            input="6\n"  # Skip this day
        )
        
        assert result.exit_code == 0
        assert "Found 1 days with predictions" in result.output
        assert "Labeling Complete!" in result.output
        assert "Skipped: 1 days" in result.output

    def test_mood_aliases_normalization(self, runner):
        """Test mood alias normalization works correctly."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Test various aliases get normalized
        result = runner.invoke(
            cli,
            ["label", "episode", "--date-range", "2024-03-01:2024-03-07",
             "--mood", "mania", "--severity", "4", "--dry-run"],
        )
        
        assert result.exit_code == 0
        assert "manic" in result.output.lower()

    @pytest.mark.parametrize("mood,min_days,should_warn", [
        ("depressive", 10, True),   # < 14 days
        ("depressive", 14, False),  # = 14 days
        ("manic", 5, True),         # < 7 days
        ("manic", 7, False),        # = 7 days
        ("hypomanic", 3, True),     # < 4 days
        ("hypomanic", 4, False),    # = 4 days
    ])
    def test_dsm5_duration_validation(self, runner, mood, min_days, should_warn):
        """Test DSM-5 duration validation for various episode types."""
        from big_mood_detector.interfaces.cli.main import cli
        
        start_date = "2024-03-01"
        end_date = date(2024, 3, min_days).isoformat()
        
        result = runner.invoke(
            cli,
            ["label", "episode", "--date-range", f"{start_date}:{end_date}",
             "--mood", mood, "--severity", "3", "--no-interactive"],
        )
        
        if should_warn:
            assert result.exit_code == 1
            assert "require" in result.output
            assert "DSM-5" in result.output
        else:
            # Should succeed (with dry-run to avoid needing persistence)
            result = runner.invoke(
                cli,
                ["label", "episode", "--date-range", f"{start_date}:{end_date}",
                 "--mood", mood, "--severity", "3", "--dry-run"],
            )
            assert result.exit_code == 0
            assert "DRY RUN" in result.output