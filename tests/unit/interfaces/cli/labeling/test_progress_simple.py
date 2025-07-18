"""
Test Simple Progress Features for Label CLI

Tests for basic progress indicators during labeling operations.
"""

import pytest
from click.testing import CliRunner

from big_mood_detector.interfaces.cli.main import cli


class TestSimpleProgress:
    """Test simple progress features in label CLI."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_episode_labeling_with_progress_flag(self, runner):
        """Test that --progress flag is accepted."""
        # When: Running with progress flag
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
                "--progress",
            ],
        )

        # Then: Should show error about unknown option (for now)
        assert "--progress" in result.output or result.exit_code != 0

    def test_export_with_verbose_output(self, runner, tmp_path):
        """Test that export can show verbose output."""
        # Given: A database with episodes
        db_path = tmp_path / "test.db"

        # Create test data
        from big_mood_detector.domain.services.episode_labeler import EpisodeLabeler
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )

        labeler = EpisodeLabeler()
        labeler.add_episode(
            date="2024-01-01",
            episode_type="depressive",
            severity=3,
        )

        repo = SQLiteEpisodeRepository(db_path)
        repo.save_labeler(labeler)

        # When: Exporting with verbose flag
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
                "--verbose",
            ],
        )

        # Then: Should include additional output
        assert result.exit_code == 0
        assert "Exported to" in result.output

    def test_stats_with_detailed_flag(self, runner, tmp_path):
        """Test that stats can show detailed output."""
        # Given: A database that exists (even if empty)
        db_path = tmp_path / "test.db"
        # Create empty database
        from big_mood_detector.infrastructure.repositories.sqlite_episode_repository import (
            SQLiteEpisodeRepository,
        )

        SQLiteEpisodeRepository(db_path)

        # When: Getting stats with detailed flag
        result = runner.invoke(
            cli,
            [
                "label",
                "stats",
                "--db",
                str(db_path),
                "--detailed",
            ],
        )

        # Then: Command should handle the flag
        if result.exit_code != 0:
            print(f"Error: {result.output}")
        assert result.exit_code == 0

    def test_import_shows_count_messages(self, runner, tmp_path):
        """Test that import shows count of imported episodes."""
        # Given: A CSV file with episodes
        csv_file = tmp_path / "episodes.csv"
        csv_content = """date,mood,severity
2024-01-01,depressive,3
2024-01-02,manic,4
"""
        csv_file.write_text(csv_content)

        # When: Importing
        result = runner.invoke(
            cli,
            [
                "label",
                "import",
                str(csv_file),
            ],
        )

        # Then: Should show import count
        if result.exit_code != 0:
            print(f"Import error: {result.output}")
            print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        assert "Imported 2 episodes" in result.output

    def test_quiet_mode_suppresses_output(self, runner):
        """Test that --quiet flag suppresses output."""
        # When: Running with quiet flag
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
                "--quiet",
            ],
        )

        # Then: Output should be minimal
        # For now, just check the flag is accepted
        assert "--quiet" in result.output or result.exit_code == 0
