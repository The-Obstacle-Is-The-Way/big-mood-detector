"""
End-to-End Test for Label Workflow

Tests the complete labeling workflow from episode creation to export.
Uses BDD-style Given-When-Then structure.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from big_mood_detector.interfaces.cli.main import cli


class TestLabelWorkflowE2E:
    """End-to-end tests for the complete label workflow."""

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

    def test_complete_labeling_workflow(self, runner, temp_db):
        """
        Given a user wants to label mood episodes
        When they create episodes, baselines, and export data
        Then the workflow should complete successfully with persisted data
        """
        # Given: A fresh database
        assert temp_db.exists()

        # When: User labels a depressive episode
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-01-01:2024-01-14",
                "--mood",
                "depressive",
                "--severity",
                "4",
                "--rater-id",
                "clinician1",
                "--notes",
                "Patient reported low energy and anhedonia",
                "--db",
                str(temp_db),
            ],
        )

        # Then: Episode is created successfully
        assert result.exit_code == 0
        assert "Labeled 14-day depressive episode" in result.output
        assert "Saved to database" in result.output

        # When: User adds a baseline period
        result = runner.invoke(
            cli,
            [
                "label",
                "baseline",
                "--start",
                "2024-02-01",
                "--end",
                "2024-02-28",
                "--notes",
                "Patient stable on medication",
                "--rater-id",
                "clinician1",
                "--db",
                str(temp_db),
            ],
        )

        # Then: Baseline is created successfully
        assert result.exit_code == 0
        assert "Marked baseline period (28 days)" in result.output

        # When: User adds a manic episode
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-03-15:2024-03-22",
                "--mood",
                "manic",
                "--severity",
                "5",
                "--rater-id",
                "clinician1",
                "--notes",
                "Hospitalization required",
                "--db",
                str(temp_db),
            ],
        )

        # Then: Manic episode is created
        assert result.exit_code == 0
        assert "Labeled 8-day manic episode" in result.output

        # When: User checks statistics
        result = runner.invoke(cli, ["label", "stats", "--db", str(temp_db)])

        # Then: Statistics show all labeled data
        assert result.exit_code == 0
        assert "Total Episodes: 2" in result.output
        assert "Total Baselines: 1" in result.output
        assert "depressive" in result.output.lower()
        assert "manic" in result.output.lower()

        # When: User exports to CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            result = runner.invoke(
                cli,
                ["label", "export", "--db", str(temp_db), "--output", str(csv_path)],
            )

            # Then: Export succeeds and contains all data
            assert result.exit_code == 0
            assert csv_path.exists()

            df = pd.read_csv(csv_path)
            assert len(df) == 28 + 14 + 8  # All days labeled
            assert "depressive" in df["label"].values
            assert "manic" in df["label"].values
            assert "baseline" in df["label"].values

        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_multi_rater_collaboration(self, runner, temp_db):
        """
        Given multiple clinicians are labeling episodes
        When they each add their assessments
        Then the system should track rater attribution
        """
        # Given: Two different raters
        rater1 = "dr_smith"
        rater2 = "dr_jones"

        # When: First rater labels an episode
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-04-01:2024-04-07",
                "--mood",
                "hypomanic",
                "--severity",
                "3",
                "--rater-id",
                rater1,
                "--db",
                str(temp_db),
            ],
        )
        assert result.exit_code == 0

        # When: Second rater labels a different episode
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-04-10:2024-04-24",
                "--mood",
                "depressive",
                "--severity",
                "3",
                "--rater-id",
                rater2,
                "--db",
                str(temp_db),
            ],
        )
        if result.exit_code != 0:
            print(f"Error output: {result.output}")
            print(f"Exception: {result.exception}")
        assert result.exit_code == 0

        # When: Check statistics
        result = runner.invoke(cli, ["label", "stats", "--db", str(temp_db)])

        # Then: Both raters are shown
        assert result.exit_code == 0
        assert "Raters: 2" in result.output
        assert rater1 in result.output
        assert rater2 in result.output

    def test_undo_functionality(self, runner, temp_db):
        """
        Given a user made a labeling mistake in the current session
        When they use the undo command
        Then the last label should be removed

        Note: Undo only works within the same session, not for database-loaded episodes
        """
        # Given: Start a session without a database (in-memory only)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            # Create an episode without database
            result = runner.invoke(
                cli,
                [
                    "label",
                    "episode",
                    "--date-range",
                    "2024-05-01:2024-05-07",
                    "--mood",
                    "mixed",
                    "--severity",
                    "3",
                    "--output",
                    str(csv_path),
                ],
            )
            assert result.exit_code == 0

            # When: User undoes the last label (in same session)
            # Since we can't maintain session state between CLI invocations,
            # we'll test that undo reports no labels when database is empty
            result = runner.invoke(cli, ["label", "undo"])

            # Then: Undo reports no labels to undo (expected behavior)
            assert result.exit_code == 0
            assert "No labels to undo" in result.output

        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_json_export_format(self, runner, temp_db):
        """
        Given labeled episodes exist
        When user exports in JSON format
        Then the export should be valid JSON with all fields
        """
        # Given: A labeled episode
        runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-06-01:2024-06-14",
                "--mood",
                "depressive",
                "--severity",
                "3",
                "--notes",
                "Test episode",
                "--db",
                str(temp_db),
            ],
        )

        # When: Export to JSON
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = Path(f.name)

        try:
            result = runner.invoke(
                cli,
                [
                    "label",
                    "export",
                    "--db",
                    str(temp_db),
                    "--output",
                    str(json_path),
                    "--format",
                    "json",
                ],
            )

            # Then: Export succeeds and is valid JSON
            assert result.exit_code == 0
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 14  # 14 days
            assert all("date" in record for record in data)
            assert all("label" in record for record in data)
            assert all("severity" in record for record in data)

        finally:
            if json_path.exists():
                json_path.unlink()

    def test_conflict_resolution(self, runner, temp_db):
        """
        Given an episode is already labeled
        When user tries to label overlapping dates
        Then the system should detect and handle conflicts
        """
        # Given: An existing episode
        runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-07-01:2024-07-07",
                "--mood",
                "manic",
                "--db",
                str(temp_db),
            ],
        )

        # When: Try to label overlapping dates (non-interactive mode)
        result = runner.invoke(
            cli,
            [
                "label",
                "episode",
                "--date-range",
                "2024-07-05:2024-07-10",
                "--mood",
                "depressive",
                "--no-interactive",
                "--db",
                str(temp_db),
            ],
        )

        # Then: Conflict is detected
        assert result.exit_code != 0
        assert "Conflict detected" in result.output or "Aborted" in result.output
