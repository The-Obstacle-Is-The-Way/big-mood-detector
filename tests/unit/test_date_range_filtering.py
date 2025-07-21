"""
Test date range filtering for XML processing.

Tests for Issue #33: Add date range filtering for XML processing to handle large files
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

class TestDateRangeFiltering:
    """Test date range filtering functionality."""

    def test_days_back_parameter_added_to_cli(self):
        """Test that --days-back parameter is available."""
        from big_mood_detector.interfaces.cli.commands import process_command

        runner = CliRunner()
        result = runner.invoke(process_command, ["--help"])
        assert result.exit_code == 0
        assert "--days-back" in result.output
        assert "Process only the last N days" in result.output

    def test_date_range_parameter_added_to_cli(self):
        """Test that --date-range parameter is available."""
        from big_mood_detector.interfaces.cli.commands import process_command

        runner = CliRunner()
        result = runner.invoke(process_command, ["--help"])
        assert result.exit_code == 0
        assert "--date-range" in result.output
        assert "Date range in format" in result.output

    @patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline")
    def test_days_back_filters_data(self, mock_pipeline_class):
        """Test that --days-back correctly calculates date range."""
        from big_mood_detector.interfaces.cli.commands import process_command

        import pandas as pd

        runner = CliRunner()
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        # Return a simple DataFrame instead of Mock
        mock_pipeline.process_health_export.return_value = pd.DataFrame({"test": [1, 2, 3]})

        with runner.isolated_filesystem():
            # Create a dummy file
            test_file = Path("test.xml")
            test_file.write_text("<HealthData/>")

            # Run with --days-back 90
            result = runner.invoke(
                process_command,
                ["test.xml", "--days-back", "90"]
            )

            if result.exit_code != 0:
                print(f"Exit code: {result.exit_code}")
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")
            assert result.exit_code == 0

            # Verify the pipeline was called with correct date range
            mock_pipeline.process_health_export.assert_called_once()
            call_args = mock_pipeline.process_health_export.call_args

            # Check that end_date is today and start_date is 90 days ago
            start_date = call_args.kwargs.get('start_date')
            end_date = call_args.kwargs.get('end_date')

            assert end_date is not None
            assert start_date is not None
            assert (end_date - start_date).days == 90

    @patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline")
    def test_date_range_parsing(self, mock_pipeline_class):
        """Test that --date-range correctly parses dates."""
        from big_mood_detector.interfaces.cli.commands import process_command

        import pandas as pd

        runner = CliRunner()
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.process_health_export.return_value = pd.DataFrame({"test": [1, 2, 3]})

        with runner.isolated_filesystem():
            # Create a dummy file
            test_file = Path("test.xml")
            test_file.write_text("<HealthData/>")

            # Run with --date-range
            result = runner.invoke(
                process_command,
                ["test.xml", "--date-range", "2024-01-01:2024-03-31"]
            )

            assert result.exit_code == 0

            # Verify the pipeline was called with correct dates
            mock_pipeline.process_health_export.assert_called_once()
            call_args = mock_pipeline.process_health_export.call_args

            start_date = call_args.kwargs.get('start_date')
            end_date = call_args.kwargs.get('end_date')

            assert start_date == datetime(2024, 1, 1).date()
            assert end_date == datetime(2024, 3, 31).date()

    def test_invalid_date_range_format(self):
        """Test that invalid date range format shows error."""
        from big_mood_detector.interfaces.cli.commands import process_command

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a dummy file
            test_file = Path("test.xml")
            test_file.write_text("<HealthData/>")

            # Run with invalid date range
            result = runner.invoke(
                process_command,
                ["test.xml", "--date-range", "invalid-format"]
            )

            assert result.exit_code != 0
            assert "Invalid date range format" in result.output

    def test_days_back_and_date_range_conflict(self):
        """Test that using both --days-back and --date-range shows error."""
        from big_mood_detector.interfaces.cli.commands import process_command

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a dummy file
            test_file = Path("test.xml")
            test_file.write_text("<HealthData/>")

            # Run with both parameters
            result = runner.invoke(
                process_command,
                ["test.xml", "--days-back", "90", "--date-range", "2024-01-01:2024-03-31"]
            )

            assert result.exit_code != 0
            assert "Cannot use both --days-back and --date-range" in result.output

    @patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline")
    def test_no_date_filtering_by_default(self, mock_pipeline_class):
        """Test that no date filtering is applied by default."""
        from big_mood_detector.interfaces.cli.commands import process_command

        import pandas as pd

        runner = CliRunner()
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.process_health_export.return_value = pd.DataFrame({"test": [1, 2, 3]})

        with runner.isolated_filesystem():
            # Create a dummy file
            test_file = Path("test.xml")
            test_file.write_text("<HealthData/>")

            # Run without date parameters
            result = runner.invoke(process_command, ["test.xml"])

            assert result.exit_code == 0

            # Verify no date filtering was applied
            mock_pipeline.process_health_export.assert_called_once()
            call_args = mock_pipeline.process_health_export.call_args

            assert call_args.kwargs.get('start_date') is None
            assert call_args.kwargs.get('end_date') is None

    def test_date_range_validation_start_after_end(self):
        """Test that start date after end date shows error."""
        from big_mood_detector.interfaces.cli.commands import process_command

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a dummy file
            test_file = Path("test.xml")
            test_file.write_text("<HealthData/>")

            # Run with invalid date range (start > end)
            result = runner.invoke(
                process_command,
                ["test.xml", "--date-range", "2024-03-31:2024-01-01"]
            )

            assert result.exit_code != 0
            assert "must be before end date" in result.output

    @patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline")
    def test_large_file_warning_with_suggestion(self, mock_pipeline_class):
        """Test that large files show warning suggesting date filtering."""
        from big_mood_detector.interfaces.cli.commands import process_command

        import pandas as pd

        runner = CliRunner()
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.process_health_export.return_value = pd.DataFrame({"test": [1, 2, 3]})

        with runner.isolated_filesystem():
            # Create a large dummy file (simulate 600MB)
            test_file = Path("test.xml")
            test_file.write_text("<HealthData/>" + " " * 600_000_000)

            # Run without date filtering
            result = runner.invoke(process_command, ["test.xml"])

            # Should show warning about large file
            assert "Very large file" in result.output
            assert "Consider using --days-back or --date-range" in result.output

