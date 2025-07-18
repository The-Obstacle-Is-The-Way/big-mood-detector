"""
Test Progress Tracking for Label CLI

Tests for progress bars and status updates during labeling operations.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from big_mood_detector.interfaces.cli.main import cli


class TestProgressTracking:
    """Test progress tracking features in label CLI."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_progress(self):
        """Mock rich progress bar."""
        with patch("big_mood_detector.interfaces.cli.labeling.commands.Progress") as mock:
            yield mock

    @pytest.fixture
    def mock_console(self):
        """Mock rich console."""
        with patch("big_mood_detector.interfaces.cli.utils.console") as mock:
            yield mock

    def test_episode_labeling_shows_progress(self, runner, mock_progress):
        """Test that episode labeling shows progress for date ranges."""
        # Given: A date range to label
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-01-01:2024-01-31",
                "--mood",
                "depressive",
                "--severity",
                "3",
                "--no-interactive",
                "--progress",
            ],
        )

        # Then: Progress bar should be created
        mock_progress.assert_called_once()
        
        # And: Command should succeed
        assert result.exit_code == 0

    def test_batch_import_shows_progress(self, runner, mock_progress, tmp_path):
        """Test that batch import shows progress for multiple episodes."""
        # Given: A CSV file with multiple episodes
        csv_file = tmp_path / "episodes.csv"
        csv_content = """start_date,end_date,mood,severity,notes
2024-01-01,2024-01-14,depressive,3,First episode
2024-02-01,2024-02-07,manic,4,Second episode
2024-03-01,2024-03-14,depressive,2,Third episode
"""
        csv_file.write_text(csv_content)

        # When: Importing with progress flag
        result = runner.invoke(
            cli,
            [
                "label",
                "import",
                str(csv_file),
                "--progress",
            ],
        )

        # Then: Progress should be shown
        mock_progress.assert_called()
        
        # And: Import should succeed
        assert result.exit_code == 0

    def test_export_shows_progress_for_large_datasets(self, runner, mock_progress, tmp_path):
        """Test that export shows progress when processing many records."""
        # Given: A database with many episodes
        db_path = tmp_path / "test.db"
        
        # Create test data
        from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )
        
        labeler = EpisodeLabeler()
        # Add 100 episodes
        for i in range(100):
            labeler.add_episode(
                date=f"2024-01-{i+1:02d}",
                episode_type="depressive" if i % 2 == 0 else "manic",
                severity=(i % 5) + 1,
            )
        
        repo = SQLiteEpisodeRepository(db_path)
        repo.save_labeler(labeler)

        # When: Exporting with progress
        output_file = tmp_path / "export.csv"
        result = runner.invoke(
            cli,
            [
                "label",
                "export",
                "--db",
                str(db_path),
                "--output",
                str(output_file),
                "--progress",
            ],
        )

        # Then: Progress should be shown
        mock_progress.assert_called()
        
        # And: Export should succeed
        assert result.exit_code == 0
        assert output_file.exists()

    def test_statistics_calculation_shows_progress(self, runner, mock_console, tmp_path):
        """Test that statistics calculation shows progress spinner."""
        # Given: A database with episodes
        db_path = tmp_path / "test.db"
        
        # When: Getting statistics with progress
        with patch("big_mood_detector.interfaces.cli.labeling.stats_command.Spinner") as mock_spinner:
            result = runner.invoke(
                cli,
                [
                    "label",
                    "stats",
                    "--db",
                    str(db_path),
                    "--detailed",
                    "--progress",
                ],
            )

            # Then: Spinner should be shown
            mock_spinner.assert_called()

    def test_live_status_updates_during_processing(self, runner, mock_console):
        """Test that live status updates are shown during processing."""
        # Given: A long-running operation
        with patch("big_mood_detector.interfaces.cli.labeling.commands.Status") as mock_status:
            result = runner.invoke(
                cli,
                [
                    "label",
                    "process",
                    "--predictions",
                    "predictions.csv",
                    "--live-status",
                ],
            )

            # Then: Status should be created
            mock_status.assert_called()
            
            # And: Updates should be shown
            mock_status.return_value.update.assert_called()

    def test_progress_disabled_by_default(self, runner, mock_progress):
        """Test that progress is not shown without explicit flag."""
        # When: Running command without progress flag
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date",
                "2024-01-01",
                "--mood",
                "manic",
                "--severity",
                "3",
            ],
        )

        # Then: Progress should not be shown
        mock_progress.assert_not_called()

    def test_progress_respects_quiet_mode(self, runner, mock_progress, mock_console):
        """Test that progress is suppressed in quiet mode."""
        # When: Running with both progress and quiet flags
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-01-01:2024-01-07",
                "--mood",
                "mixed",
                "--severity",
                "3",
                "--progress",
                "--quiet",
            ],
        )

        # Then: Progress should not be shown
        mock_progress.assert_not_called()
        
        # And: Console output should be minimal
        mock_console.print.assert_not_called()

    def test_progress_callback_integration(self, runner):
        """Test that progress callbacks are properly integrated."""
        # Given: A custom progress callback
        progress_calls = []
        
        def progress_callback(current: int, total: int, message: str):
            progress_calls.append((current, total, message))
        
        with patch("big_mood_detector.interfaces.cli.labeling.commands.get_progress_callback") as mock_get:
            mock_get.return_value = progress_callback
            
            # When: Running a command with progress
            result = runner.invoke(
                cli,
                [
                    "label",
                    "validate",
                    "--db",
                    "test.db",
                    "--progress",
                ],
            )
            
            # Then: Progress callback should be retrieved
            mock_get.assert_called_once()

    def test_progress_with_multiple_raters(self, runner, mock_progress):
        """Test progress tracking when processing multiple raters."""
        # Given: Multiple raters to process
        with patch("big_mood_detector.interfaces.cli.labeling.commands.track") as mock_track:
            result = runner.invoke(
                cli,
                [
                    "label",
                    "merge",
                    "--raters",
                    "rater1,rater2,rater3",
                    "--progress",
                ],
            )
            
            # Then: Track should be used for iteration
            mock_track.assert_called()

    def test_progress_error_handling(self, runner, mock_console):
        """Test that progress handles errors gracefully."""
        # Given: An operation that fails
        with patch("big_mood_detector.interfaces.cli.labeling.commands.Progress") as mock_progress:
            # Make the operation fail
            mock_progress.side_effect = Exception("Progress error")
            
            # When: Running with progress
            result = runner.invoke(
                cli,
                [
                    "label",
                    "episode",
                    "--date",
                    "2024-01-01",
                    "--mood",
                    "manic",
                    "--progress",
                ],
            )
            
            # Then: Command should still work without progress
            # (graceful degradation)
            assert "Progress error" not in result.output