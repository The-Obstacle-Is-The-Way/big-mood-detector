"""
Test Label CLI Persistence

Tests for SQLite persistence integration with label CLI.
"""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner


class TestLabelPersistence:
    """Test label CLI with SQLite persistence."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        yield db_path
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    def test_label_with_database_saves_episodes(self, runner, temp_db):
        """Test that labeling with --db flag saves to database."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Label an episode with database
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-15", "--mood", "depressive",
             "--severity", "3", "--db", str(temp_db)],
        )
        
        assert result.exit_code == 0
        assert "Labeled 2024-03-15 as depressive" in result.output
        assert f"Saved to database: {temp_db}" in result.output

    def test_label_loads_from_database(self, runner, temp_db):
        """Test that labeling loads existing data from database."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # First, add an episode
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-15", "--mood", "depressive",
             "--severity", "3", "--db", str(temp_db)],
        )
        assert result.exit_code == 0
        
        # Then add another episode - should load the first one
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-16", "--mood", "manic",
             "--severity", "4", "--db", str(temp_db)],
        )
        
        assert result.exit_code == 0
        # Should mention loading existing data
        assert "Loaded" in result.output or "existing" in result.output

    def test_label_stats_with_database(self, runner, temp_db):
        """Test stats command works with database."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Add some episodes
        for i, mood in enumerate(["depressive", "manic", "hypomanic"]):
            runner.invoke(
                cli,
                ["label", "episode", "--date", f"2024-03-{i+15}", 
                 "--mood", mood, "--db", str(temp_db)],
            )
        
        # Get stats
        result = runner.invoke(cli, ["label", "stats", "--db", str(temp_db)])
        
        assert result.exit_code == 0
        assert "Total Episodes: 3" in result.output
        assert "depressive" in result.output

    def test_export_with_database(self, runner, temp_db):
        """Test export command with database."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Add episodes
        runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-15", "--mood", "depressive",
             "--db", str(temp_db)],
        )
        
        # Export to CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)
        
        try:
            result = runner.invoke(
                cli,
                ["label", "export", "--db", str(temp_db), "--output", str(csv_path)]
            )
            
            assert result.exit_code == 0
            assert csv_path.exists()
            assert f"Exported to {csv_path}" in result.output
            
            # Verify CSV content
            content = csv_path.read_text()
            assert "date,label,severity" in content
            assert "2024-03-15,depressive,3" in content
            
        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_persistence_across_sessions(self, runner, temp_db):
        """Test that data persists across CLI sessions."""
        from big_mood_detector.interfaces.cli.main import cli
        
        # Session 1: Add episodes
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-15", "--mood", "depressive",
             "--severity", "3", "--rater-id", "session1", "--db", str(temp_db)],
        )
        assert result.exit_code == 0
        
        # Session 2: Different rater adds episode
        result = runner.invoke(
            cli,
            ["label", "episode", "--date", "2024-03-16", "--mood", "manic",
             "--severity", "4", "--rater-id", "session2", "--db", str(temp_db)],
        )
        assert result.exit_code == 0
        
        # Session 3: Check stats shows both
        result = runner.invoke(cli, ["label", "stats", "--db", str(temp_db)])
        
        assert result.exit_code == 0
        assert "Total Episodes: 2" in result.output
        assert "Raters: 2" in result.output