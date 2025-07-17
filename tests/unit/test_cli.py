"""
Test CLI Commands

Ensures CLI functionality works correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner


class TestCLI:
    """Test command line interface."""

    def test_cli_imports(self) -> None:
        """Test that CLI imports work."""
        from big_mood_detector.main_cli import cli  # type: ignore

        assert cli is not None

    def test_main_command_exists(self) -> None:
        """Test that main command exists."""
        from big_mood_detector.main_cli import cli  # type: ignore

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_cli_help(self) -> None:
        """Test CLI help shows expected commands."""
        from big_mood_detector.main_cli import cli  # type: ignore

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "process" in result.output
        assert "serve" in result.output
        assert "watch" in result.output

    def test_process_command_exists(self) -> None:
        """Test that process command exists."""
        from big_mood_detector.main_cli import cli  # type: ignore

        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process health data" in result.output

    @patch(
        "big_mood_detector.application.services.data_parsing_service.DataParsingService"
    )
    def test_process_command_with_directory(self, mock_pipeline: Mock) -> None:
        """Test process command with directory input."""
        from big_mood_detector.main_cli import cli  # type: ignore

        # Setup mock
        mock_instance = Mock()
        # Mock process_health_export to return a DataFrame
        import pandas as pd

        mock_df = pd.DataFrame({"test": [1, 2, 3]})
        mock_instance.process_health_export.return_value = mock_df
        mock_pipeline.return_value = mock_instance

        # Create test directory with valid health data files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock JSON file that would pass validation
            test_file = Path(tmpdir) / "Sleep Analysis.json"
            test_file.write_text('{"data": []}')  # Valid JSON health file

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

    def test_serve_command_exists(self) -> None:
        """Test that serve command exists."""
        from big_mood_detector.main_cli import cli  # type: ignore

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the API server" in result.output

    @patch("big_mood_detector.interfaces.cli.server.uvicorn")
    def test_serve_command(self, mock_uvicorn: Mock) -> None:
        """Test serve command starts server."""
        from big_mood_detector.main_cli import cli  # type: ignore

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--port", "8001"])
        assert result.exit_code == 0

        # Verify uvicorn was called
        mock_uvicorn.run.assert_called_once()

    def test_watch_command_exists(self) -> None:
        """Test that watch command exists."""
        from big_mood_detector.main_cli import cli  # type: ignore

        runner = CliRunner()
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch directory" in result.output

    @patch("big_mood_detector.infrastructure.monitoring.file_watcher.FileWatcher")
    def test_watch_command_with_options(self, mock_file_watcher: Mock) -> None:
        """Test watch command accepts all options."""
        from big_mood_detector.main_cli import cli  # type: ignore

        # Mock the file watcher to prevent actual watching
        mock_instance = Mock()
        mock_file_watcher.return_value = mock_instance

        # Mock watch to complete normally instead of raising KeyboardInterrupt
        def mock_watch() -> None:
            return  # Just return normally for testing

        mock_instance.watch = mock_watch

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
                catch_exceptions=False,  # Better error reporting in tests
            )

            # Should show setup messages
            assert result.exit_code == 0
            assert "Watching" in result.output
            assert "health data files" in result.output
            assert "Poll interval: 30" in result.output

    @patch(
        "big_mood_detector.application.services.data_parsing_service.DataParsingService"
    )
    def test_process_command_error_handling(self, mock_pipeline: Mock) -> None:
        """Test process command error handling."""
        from big_mood_detector.main_cli import cli  # type: ignore

        # Setup mock to raise error
        mock_instance = Mock()
        mock_instance.process_health_export.side_effect = Exception("Processing failed")
        mock_pipeline.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid health data file so validation passes
            test_file = Path(tmpdir) / "Heart Rate.json"
            test_file.write_text('{"data": []}')

            runner = CliRunner()
            result = runner.invoke(cli, ["process", tmpdir])

            assert result.exit_code == 1
            assert "Error" in result.output
            assert "Processing failed" in result.output
