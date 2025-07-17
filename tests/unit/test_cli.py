"""
Test CLI commands for Big Mood Detector

TDD for CLI implementation
"""

import tempfile
from unittest.mock import Mock, patch

from click.testing import CliRunner


class TestCLI:
    """Test CLI commands."""

    def test_cli_imports(self):
        """Test that CLI module can be imported."""
        from big_mood_detector import main_cli  # noqa: F401

    def test_main_command_exists(self):
        """Test that main CLI group exists."""
        from big_mood_detector.main_cli import cli

        assert cli is not None
        assert hasattr(cli, "command")

    def test_cli_help(self):
        """Test CLI help command."""
        from big_mood_detector.main_cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Big Mood Detector - Clinical mood prediction" in result.output

    def test_process_command_exists(self):
        """Test that process command exists."""
        from big_mood_detector.main_cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process health data" in result.output

    @patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline")
    def test_process_command_with_directory(self, mock_pipeline):
        """Test process command with directory input."""
        from big_mood_detector.main_cli import cli

        # Setup mock
        mock_instance = Mock()
        # Mock process_health_export to return a DataFrame
        import pandas as pd

        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_instance.process_health_export.return_value = mock_df
        mock_pipeline.return_value = mock_instance

        # Create test directory
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["process", tmpdir, "--output", "test_output.csv"],
            )

            # Debug output
            if result.exit_code != 0:
                print(f"Exit code: {result.exit_code}")
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")

            # Test should pass if the command runs without error
            # The actual implementation now just processes and saves to CSV
            assert result.exit_code == 0

            # Verify pipeline was called
            mock_instance.process_health_export.assert_called_once()

    def test_serve_command_exists(self):
        """Test that serve command exists."""
        from big_mood_detector.main_cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the API server" in result.output

    @patch("big_mood_detector.interfaces.cli.server.uvicorn")
    def test_serve_command(self, mock_uvicorn):
        """Test serve command starts uvicorn."""
        from big_mood_detector.main_cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--port", "8001"])

        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            "big_mood_detector.interfaces.api.main:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
            workers=1,
        )

    def test_watch_command_exists(self):
        """Test that watch command exists."""
        from big_mood_detector.main_cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch directory" in result.output

    @patch("big_mood_detector.infrastructure.monitoring.file_watcher.FileWatcher")
    def test_watch_command_with_options(self, mock_file_watcher):
        """Test watch command accepts all options."""
        from big_mood_detector.main_cli import cli

        # Mock the file watcher to prevent actual watching
        mock_instance = Mock()
        mock_file_watcher.return_value = mock_instance
        mock_instance.watch.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "watch",
                    tmpdir,
                    "--poll-interval",
                    "30",
                    "--patterns",
                    "*.json",
                    "--no-recursive",
                ],
            )

            # Should show setup messages
            assert "Watching" in result.output
            assert "health data files" in result.output
            assert "Poll interval: 30" in result.output
            assert "Stopping file watcher" in result.output

    def test_process_command_error_handling(self):
        """Test process command handles errors gracefully."""
        from big_mood_detector.main_cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["process", "/nonexistent/path"])

        assert result.exit_code == 2  # Click error code for bad path
        assert "does not exist" in result.output

    @patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline")
    def test_process_command_pipeline_error(self, mock_pipeline):
        """Test process command handles pipeline errors."""
        from big_mood_detector.main_cli import cli

        # Setup mock to raise error
        mock_instance = Mock()
        mock_instance.process_health_export.side_effect = Exception("Processing failed")
        mock_pipeline.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(cli, ["process", tmpdir])

            assert result.exit_code == 1
            assert "Error" in result.output
            assert "Processing failed" in result.output
