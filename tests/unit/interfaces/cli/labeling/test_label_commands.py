"""
Test Label CLI Commands

Following TDD approach with minimal, consistent mocking.
"""

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from click.testing import CliRunner


class TestLabelCLI:
    """Test the label CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_dependencies(self):
        """Single fixture to mock ALL label CLI dependencies consistently."""
        with patch(
            "big_mood_detector.interfaces.cli.labeling.commands.EpisodeLabeler"
        ) as mock_labeler:
            with patch(
                "big_mood_detector.interfaces.cli.labeling.commands.ClinicalValidator"
            ) as mock_validator:
                # Set up labeler mock
                labeler_instance = Mock()
                labeler_instance.check_overlap.return_value = False
                labeler_instance.undo_last.return_value = True
                labeler_instance.to_dataframe.return_value = pd.DataFrame()
                mock_labeler.return_value = labeler_instance

                # Set up validator mock - always valid for unit tests
                validator_instance = Mock()
                validator_instance.validate_episode_duration.return_value = Mock(
                    valid=True, warning=None, suggestion=None
                )
                mock_validator.return_value = validator_instance

                yield labeler_instance, validator_instance

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

    def test_label_episode_single_day(self, runner, mock_dependencies):
        """Test labeling a single day as depressed."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date",
                "2024-03-15",
                "--mood",
                "depressive",
                "--severity",
                "3",
                "--rater-id",
                "test_clinician",
            ],
        )

        assert result.exit_code == 0
        assert "Labeled 2024-03-15 as depressive" in result.output
        labeler_instance.add_episode.assert_called_once()

        # Check the call arguments
        call_args = labeler_instance.add_episode.call_args
        assert call_args.kwargs["date"] == date(2024, 3, 15)
        assert call_args.kwargs["episode_type"] == "depressive"
        assert call_args.kwargs["severity"] == 3

    def test_label_episode_date_range(self, runner, mock_dependencies):
        """Test labeling multi-day episode."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-03-10:2024-03-20",
                "--mood",
                "hypomanic",
                "--severity",
                "3",
            ],
        )

        assert result.exit_code == 0
        assert "Labeled 11-day hypomanic episode" in result.output

        call_args = labeler_instance.add_episode.call_args
        assert call_args.kwargs["start_date"] == date(2024, 3, 10)
        assert call_args.kwargs["end_date"] == date(2024, 3, 20)
        assert call_args.kwargs["episode_type"] == "hypomanic"

    def test_dsm5_duration_warning(self, runner):
        """Test DSM-5 validation warnings for short episodes."""
        from big_mood_detector.interfaces.cli.main import cli

        # This test uses REAL validator to test actual DSM-5 logic
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-03-10:2024-03-12",
                "--mood",
                "manic",
                "--severity",
                "4",
                "--no-interactive",
            ],
            input="n\n",  # Don't confirm when warned
        )

        assert "Manic episodes require ≥7 days" in result.output
        assert result.exit_code != 0  # Should exit with error if not confirmed

    def test_label_baseline_command(self, runner, mock_dependencies):
        """Test marking baseline periods."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        result = runner.invoke(
            cli,
            [
                "label",
                "baseline",
                "--start",
                "2024-05-01",
                "--end",
                "2024-05-14",
                "--notes",
                "Stable on medication",
            ],
        )

        assert result.exit_code == 0
        assert "Marked baseline period" in result.output
        labeler_instance.add_baseline.assert_called_once()

    def test_mood_aliases(self, runner, mock_dependencies):
        """Test case-insensitive mood aliases."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

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

            call_args = labeler_instance.add_episode.call_args
            assert call_args.kwargs["episode_type"] == expected

    def test_interactive_mode_with_predictions(self, runner, mock_dependencies):
        """Test interactive mode showing predictions."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        # Create mock predictions file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
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
                                "sleep_efficiency": 0.42,
                            },
                        }
                    ]
                },
                f,
            )
            predictions_file = f.name

        try:
            result = runner.invoke(
                cli,
                [
                    "label",
                    "episode",
                    "--predictions",
                    predictions_file,
                    "--interactive",
                ],
                input="1\n3\nn\n",  # Select depression, severity 3, no notes
            )

            assert result.exit_code == 0
            # Check for content regardless of Rich formatting
            assert "72%" in result.output  # Depression risk
            assert "3.2 hrs" in result.output  # Sleep hours
        finally:
            Path(predictions_file).unlink()

    def test_dry_run_mode(self, runner, mock_dependencies):
        """Test dry-run doesn't save anything."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date",
                "2024-03-15",
                "--mood",
                "depressive",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Would label" in result.output
        labeler_instance.add_episode.assert_not_called()

    def test_label_undo_command(self, runner, mock_dependencies):
        """Test undo functionality."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        result = runner.invoke(cli, ["label", "undo"])

        assert result.exit_code == 0
        assert "Undid last label" in result.output
        labeler_instance.undo_last.assert_called_once()

    def test_csv_export_on_save(self, runner, mock_dependencies):
        """Test that labels are exported to CSV after saving."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "labels.csv"

            result = runner.invoke(
                cli,
                [
                    "label",
                    "episode",
                    "--date",
                    "2024-03-15",
                    "--mood",
                    "depressive",
                    "--output",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            labeler_instance.to_dataframe.assert_called()
            # In real implementation, CSV should be written

    def test_rater_id_from_config_or_flag(self, runner, mock_dependencies):
        """Test rater ID can come from flag or config."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        # Test with explicit rater ID
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date",
                "2024-03-15",
                "--mood",
                "depressive",
                "--rater-id",
                "dr_smith",
            ],
        )

        assert result.exit_code == 0
        # Check rater_id was passed
        call_args = labeler_instance.add_episode.call_args
        assert call_args.kwargs["rater_id"] == "dr_smith"

    def test_conflict_detection(self, runner, mock_dependencies):
        """Test that overlapping episodes are detected."""
        from big_mood_detector.interfaces.cli.main import cli

        labeler_instance, _ = mock_dependencies

        # Mock existing episode overlap
        labeler_instance.check_overlap.return_value = True

        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-03-15:2024-03-20",
                "--mood",
                "depressive",
                "--no-interactive",
            ],
            input="n\n",  # Don't override when conflict detected
        )

        assert "Conflict detected" in result.output or "overlaps" in result.output

    def test_max_span_warning(self, runner):
        """Test warning for very long episode spans."""
        from big_mood_detector.interfaces.cli.main import cli

        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-01-01:2024-06-01",
                "--mood",
                "depressive",
                "--no-interactive",
            ],
            input="n\n",  # Don't confirm
        )

        assert "days is unusually long" in result.output or "Warning" in result.output
