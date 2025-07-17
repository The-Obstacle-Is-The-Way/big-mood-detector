"""
Test CLI commands for Big Mood Detector

TDD for CLI implementation
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner


class TestCLI:
    """Test CLI commands."""

    def test_cli_imports(self):
        """Test that CLI module can be imported."""
        from big_mood_detector import cli  # noqa: F401

    def test_main_command_exists(self):
        """Test that main CLI group exists."""
        from big_mood_detector.cli import main

        assert main is not None
        assert hasattr(main, "command")

    def test_cli_help(self):
        """Test CLI help command."""
        from big_mood_detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Big Mood Detector CLI" in result.output

    def test_process_command_exists(self):
        """Test that process command exists."""
        from big_mood_detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process health data" in result.output

    @patch("big_mood_detector.cli.MoodPredictionPipeline")
    def test_process_command_with_directory(self, mock_pipeline):
        """Test process command with directory input."""
        from big_mood_detector.cli import main

        # Setup mock
        mock_instance = Mock()
        mock_result = Mock()
        mock_result.overall_summary = {
            "avg_depression_risk": 0.41,
            "avg_hypomanic_risk": 0.005,
            "avg_manic_risk": 0.001,
            "days_analyzed": 7,
        }
        mock_result.confidence_score = 0.85
        mock_result.daily_predictions = {}
        mock_result.records_processed = 100
        mock_result.warnings = []
        mock_instance.process_apple_health_file.return_value = mock_result
        mock_instance.export_results.return_value = None
        mock_pipeline.return_value = mock_instance

        # Create test directory
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["process", tmpdir, "--output", "test_output.csv", "--no-report"],
            )

            # Debug output
            if result.exit_code != 0:
                print(f"Exit code: {result.exit_code}")
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")

            assert result.exit_code == 0
            assert "Analysis Complete" in result.output
            assert "Depression Risk: 41.0%" in result.output
            assert "Hypomanic Risk: 0.5%" in result.output
            assert "Manic Risk: 0.1%" in result.output

            # Verify pipeline was called
            mock_instance.process_apple_health_file.assert_called_once()
            mock_instance.export_results.assert_called_once()

    def test_serve_command_exists(self):
        """Test that serve command exists."""
        from big_mood_detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the API server" in result.output

    @patch("big_mood_detector.cli.uvicorn")
    def test_serve_command(self, mock_uvicorn):
        """Test serve command starts uvicorn."""
        from big_mood_detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--port", "8001"])

        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            "big_mood_detector.interfaces.api.main:app",
            host="0.0.0.0",
            port=8001,
            reload=True,
        )

    def test_watch_command_exists(self):
        """Test that watch command exists."""
        from big_mood_detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch directory" in result.output

    @patch("big_mood_detector.infrastructure.monitoring.file_watcher.FileWatcher")
    def test_watch_command_with_options(self, mock_file_watcher):
        """Test watch command accepts all options."""
        from big_mood_detector.cli import main

        # Mock the file watcher to prevent actual watching
        mock_instance = Mock()
        mock_file_watcher.return_value = mock_instance
        mock_instance.watch.side_effect = KeyboardInterrupt()  # Simulate Ctrl+C

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(
                main,
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

    @patch("big_mood_detector.cli.MoodPredictionPipeline")
    def test_process_command_with_report(self, mock_pipeline):
        """Test process command generates clinical report."""
        from big_mood_detector.cli import main

        # Setup mock
        mock_instance = Mock()
        mock_result = Mock()
        mock_result.overall_summary = {
            "avg_depression_risk": 0.41,
            "avg_hypomanic_risk": 0.005,
            "avg_manic_risk": 0.001,
            "days_analyzed": 7,
        }
        mock_result.daily_predictions = {
            "2024-01-01": {
                "depression_risk": 0.4,
                "hypomanic_risk": 0.005,
                "manic_risk": 0.001,
                "confidence": 0.8,
            }
        }
        mock_result.confidence_score = 0.85
        mock_result.records_processed = 100
        mock_result.warnings = []
        mock_instance.process_apple_health_file.return_value = mock_result
        mock_instance.export_results.return_value = None
        mock_pipeline.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.csv"
            runner = CliRunner()
            result = runner.invoke(
                main, ["process", tmpdir, "--output", str(output_path), "--report"]
            )

            assert result.exit_code == 0
            assert "Clinical report generated" in result.output

    def test_process_command_error_handling(self):
        """Test process command handles errors gracefully."""
        from big_mood_detector.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["process", "/nonexistent/path"])

        assert result.exit_code == 2  # Click error code for bad path
        assert "does not exist" in result.output

    @patch("big_mood_detector.cli.MoodPredictionPipeline")
    def test_process_command_pipeline_error(self, mock_pipeline):
        """Test process command handles pipeline errors."""
        from big_mood_detector.cli import main

        # Setup mock to raise error
        mock_instance = Mock()
        mock_instance.process_apple_health_file.side_effect = Exception(
            "Processing failed"
        )
        mock_pipeline.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            result = runner.invoke(main, ["process", tmpdir])

            assert result.exit_code == 1
            assert "Error processing health data" in result.output
            assert "Processing failed" in result.output
