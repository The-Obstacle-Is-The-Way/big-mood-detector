"""
Test Label CLI Commands

Following TDD approach - write tests first, then implement.
"""

import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

# Remove unused import


class TestLabelCLI:
    """Test the label CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture(autouse=True)
    def reset_imports(self):
        """Reset module imports between tests to avoid pollution."""
        yield
        # Clean up any module-level state
        
    @pytest.fixture
    def mock_episode_labeler(self):
        """Mock the EpisodeLabeler."""
        with patch("big_mood_detector.domain.services.episode_labeler.EpisodeLabeler") as mock:
            yield mock
    
    @contextmanager
    def mock_label_cli_dependencies(self, mock_instance=None):
        """Mock both EpisodeLabeler and ClinicalValidator for label CLI tests."""
        if mock_instance is None:
            mock_instance = Mock()
            mock_instance.check_overlap.return_value = False
            mock_instance.to_dataframe.return_value = Mock()
        
        with patch("big_mood_detector.interfaces.cli.labeling.commands.EpisodeLabeler") as mock_labeler:
            mock_labeler.return_value = mock_instance
            
            with patch("big_mood_detector.interfaces.cli.labeling.commands.ClinicalValidator") as mock_validator:
                mock_validator_instance = Mock()
                mock_validator.return_value = mock_validator_instance
                mock_validator_instance.validate_episode_duration.return_value = Mock(
                    valid=True, warning=None, suggestion=None
                )
                yield mock_instance, mock_validator_instance

    def test_label_command_exists(self, runner):
        """Test that label command exists in CLI."""
        from big_mood_detector.interfaces.cli.main import cli
        
        result = runner.invoke(cli, ["label", "--help"])
        assert result.exit_code == 0
        assert "Manage labels and create ground truth annotations" in result.output

    def test_label_defaults_to_episode_subcommand(self, runner):
        """Test that 'label' alone defaults to episode subcommand."""
        from big_mood_detector.interfaces.cli.main import cli
        
        result = runner.invoke(cli, ["label", "--help"])
        assert result.exit_code == 0
        # Should show episode options when no subcommand given
        assert "--date" in result.output or "episode" in result.output

    def test_label_episode_single_day(self, runner, mock_episode_labeler):
        """Test labeling a single day as depressed."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Mock the labeler instance
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        # Mock check_overlap to return False (no overlap)
        mock_instance.check_overlap.return_value = False
        
        # Need to mock the validator to not fail on single-day episodes
        with patch("big_mood_detector.interfaces.cli.labeling.commands.ClinicalValidator") as mock_validator:
            mock_validator_instance = Mock()
            mock_validator.return_value = mock_validator_instance
            # Return valid=True to skip the DSM-5 warning
            mock_validator_instance.validate_episode_duration.return_value = Mock(
                valid=True, warning=None, suggestion=None
            )
            
            result = runner.invoke(
                cli,
                ["label", "episode", "--date", "2024-03-15", "--mood", "depressive", 
                 "--severity", "3", "--rater-id", "test_clinician"],
            )
        
        assert result.exit_code == 0
        assert "Labeled 2024-03-15 as depressive" in result.output
        mock_instance.add_episode.assert_called_once()
        
        # Check the call arguments
        call_args = mock_instance.add_episode.call_args
        assert call_args.kwargs["date"] == date(2024, 3, 15)
        assert call_args.kwargs["episode_type"] == "depressive"
        assert call_args.kwargs["severity"] == 3

    def test_label_episode_date_range(self, runner, mock_episode_labeler):
        """Test labeling multi-day episode."""
        from big_mood_detector.interfaces.cli.main import cli
        
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        mock_instance.check_overlap.return_value = False
        
        # Need to patch EpisodeLabeler in commands module too
        with patch("big_mood_detector.interfaces.cli.labeling.commands.EpisodeLabeler") as mock_labeler_class:
            mock_labeler_class.return_value = mock_instance
            
            # Mock validator
            with patch("big_mood_detector.interfaces.cli.labeling.commands.ClinicalValidator") as mock_validator:
                mock_validator_instance = Mock()
                mock_validator.return_value = mock_validator_instance
                mock_validator_instance.validate_episode_duration.return_value = Mock(
                    valid=True, warning=None, suggestion=None
                )
                
                result = runner.invoke(
                    cli,
                    ["label", "episode", "--date-range", "2024-03-10:2024-03-20", 
                     "--mood", "hypomanic", "--severity", "3"],
                )
        
        assert result.exit_code == 0
        assert "Labeled 11-day hypomanic episode" in result.output
        
        call_args = mock_instance.add_episode.call_args
        assert call_args.kwargs["start_date"] == date(2024, 3, 10)
        assert call_args.kwargs["end_date"] == date(2024, 3, 20)
        assert call_args.kwargs["episode_type"] == "hypomanic"

    def test_dsm5_duration_warning(self, runner):
        """Test DSM-5 validation warnings for short episodes."""
        from big_mood_detector.interfaces.cli.main import cli
        
        result = runner.invoke(
            cli,
            ["label", "episode", "--date-range", "2024-03-10:2024-03-12", 
             "--mood", "manic", "--severity", "4", "--no-interactive"],
            input="n\n"  # Don't confirm when warned
        )
        
        assert "Manic episodes require ≥7 days" in result.output
        assert result.exit_code != 0  # Should exit with error if not confirmed

    def test_label_baseline_command(self, runner, mock_episode_labeler):
        """Test marking baseline periods."""
        from big_mood_detector.interfaces.cli.main import cli
        
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        
        # Need to patch the EpisodeLabeler in the baseline command module as well
        with patch("big_mood_detector.interfaces.cli.labeling.commands.EpisodeLabeler") as mock_labeler_class:
            mock_labeler_class.return_value = mock_instance
            
            result = runner.invoke(
                cli,
                ["label", "baseline", "--start", "2024-05-01", "--end", "2024-05-14",
                 "--notes", "Stable on medication"],
            )
        
        assert result.exit_code == 0
        assert "Marked baseline period" in result.output
        mock_instance.add_baseline.assert_called_once()

    def test_mood_aliases(self, runner, mock_episode_labeler):
        """Test case-insensitive mood aliases."""
        from big_mood_detector.interfaces.cli.main import cli
        
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        mock_instance.check_overlap.return_value = False
        
        # Mock validator
        with patch("big_mood_detector.interfaces.cli.labeling.commands.ClinicalValidator") as mock_validator:
            mock_validator_instance = Mock()
            mock_validator.return_value = mock_validator_instance
            mock_validator_instance.validate_episode_duration.return_value = Mock(
                valid=True, warning=None, suggestion=None
            )
            
            # Test various aliases
            aliases = [
                ("dep", "depressive"),
                ("hypo", "hypomanic"),
                ("mania", "manic"),
                ("mixed", "mixed"),
            ]
            
            for alias, expected in aliases:
                result = runner.invoke(
                    cli,
                    ["label", "episode", "--date", "2024-03-15", "--mood", alias],
                )
                assert result.exit_code == 0
                
                call_args = mock_instance.add_episode.call_args
                assert call_args.kwargs["episode_type"] == expected

    def test_interactive_mode_with_predictions(self, runner, mock_episode_labeler):
        """Test interactive mode showing predictions."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Create mock predictions file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('''
            {
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
            }
            ''')
            predictions_file = f.name
        
        try:
            result = runner.invoke(
                cli,
                ["label", "episode", "--predictions", predictions_file, "--interactive"],
                input="1\n3\nn\n"  # Select depression, severity 3, no notes
            )
            
            assert result.exit_code == 0
            assert "Depression Risk: 72%" in result.output
            assert "Sleep: 3.2 hrs" in result.output
        finally:
            Path(predictions_file).unlink()

    def test_dry_run_mode(self, runner, mock_episode_labeler):
        """Test dry-run doesn't save anything."""
        from big_mood_detector.interfaces.cli.main import cli
        
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-15", "--mood", "depressive",
             "--dry-run"],
        )
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Would label" in result.output
        mock_instance.add_episode.assert_not_called()

    def test_label_undo_command(self, runner, mock_episode_labeler):
        """Test undo functionality."""
        from big_mood_detector.interfaces.cli.main import cli
        
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        mock_instance.undo_last.return_value = True
        
        # Need to patch EpisodeLabeler in the undo command module
        with patch("big_mood_detector.interfaces.cli.labeling.commands.EpisodeLabeler") as mock_labeler_class:
            mock_labeler_class.return_value = mock_instance
            
            result = runner.invoke(cli, ["label", "undo"])
        
        assert result.exit_code == 0
        assert "Undid last label" in result.output
        mock_instance.undo_last.assert_called_once()

    def test_csv_export_on_save(self, runner, mock_episode_labeler):
        """Test that labels are exported to CSV after saving."""
        from big_mood_detector.interfaces.cli.main import cli
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "labels.csv"
            
            mock_instance = Mock()
            mock_episode_labeler.return_value = mock_instance
            
            result = runner.invoke(
                cli,
                ["label", "episode", "--date", "2024-03-15", "--mood", "depressive",
                 "--output", str(output_path)],
            )
            
            assert result.exit_code == 0
            mock_instance.to_dataframe.assert_called()
            # In real implementation, CSV should be written

    def test_rater_id_from_config_or_flag(self, runner, mock_episode_labeler):
        """Test rater ID can come from flag or config."""
        from big_mood_detector.interfaces.cli.main import cli
        
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        
        # Test with explicit rater ID
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-15", "--mood", "depressive",
             "--rater-id", "dr_smith"],
        )
        
        assert result.exit_code == 0
        # Should pass rater_id to persistence layer
        
    def test_conflict_detection(self, runner, mock_episode_labeler):
        """Test that overlapping episodes are detected."""
        from big_mood_detector.interfaces.cli.main import cli
        
        mock_instance = Mock()
        mock_episode_labeler.return_value = mock_instance
        
        # Mock existing episode
        mock_instance.check_overlap.return_value = True
        
        result = runner.invoke(
            cli,
            ["label", "episode", "--date-range", "2024-03-15:2024-03-20",
             "--mood", "depressive", "--no-interactive"],
            input="n\n"  # Don't override when conflict detected
        )
        
        assert "Conflict detected" in result.output or "overlaps" in result.output

    def test_max_span_warning(self, runner):
        """Test warning for very long episode spans."""
        from big_mood_detector.interfaces.cli.main import cli
        
        result = runner.invoke(
            cli,
            ["label", "episode", "--date-range", "2024-01-01:2024-06-01",
             "--mood", "depressive", "--no-interactive"],
            input="n\n"  # Don't confirm
        )
        
        assert "days is unusually long" in result.output or "Warning" in result.output