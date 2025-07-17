"""
Integration tests for the label CLI.

Tests the full CLI integration including the unified label group.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from big_mood_detector.interfaces.cli.main import cli


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestLabelCLIIntegration:
    """Test the integrated label CLI commands."""
    
    def test_label_help(self, runner):
        """Test that label help shows both episode and management commands."""
        result = runner.invoke(cli, ["label", "--help"])
        
        assert result.exit_code == 0
        assert "Episode Labeling:" in result.output
        assert "episode" in result.output
        assert "baseline" in result.output
        assert "Label Management:" in result.output
        assert "manage" in result.output
        
    def test_label_manage_list(self, runner):
        """Test the label manage list command through the main CLI."""
        result = runner.invoke(cli, ["label", "manage", "list"])
        
        assert result.exit_code == 0
        # Should show default labels from in-memory repository
        assert "Depression" in result.output
        assert "Mania" in result.output
        assert "Sleep Disruption" in result.output
        
    def test_label_manage_create_and_list(self, runner):
        """Test creating a label and then listing it."""
        # Create a new label
        create_result = runner.invoke(cli, [
            "label", "manage", "create",
            "--name", "Test Anxiety",
            "--description", "Test anxiety indicators",
            "--category", "mood",
            "--color", "#9B59B6"
        ])
        
        assert create_result.exit_code == 0
        assert "Label 'Test Anxiety' created successfully" in create_result.output
        
        # List labels to verify it was created
        list_result = runner.invoke(cli, ["label", "manage", "list"])
        
        assert list_result.exit_code == 0
        assert "Test Anxiety" in list_result.output
        
    def test_label_manage_search(self, runner):
        """Test searching for labels."""
        result = runner.invoke(cli, ["label", "manage", "search", "sleep"])
        
        assert result.exit_code == 0
        assert "Sleep Disruption" in result.output
        assert "Circadian Rhythm Shift" in result.output
        
    def test_label_manage_json_output(self, runner):
        """Test JSON output format."""
        result = runner.invoke(cli, ["label", "manage", "list", "--format", "json"])
        
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0
        assert all("name" in item for item in data)
        
    def test_label_episode_help(self, runner):
        """Test episode labeling help."""
        result = runner.invoke(cli, ["label", "episode", "--help"])
        
        assert result.exit_code == 0
        assert "Label mood episodes" in result.output
        assert "--mood" in result.output
        assert "--date-range" in result.output
        
    def test_label_manage_import_export_workflow(self, runner, tmp_path):
        """Test importing labels from a file."""
        # Create a test import file
        import_file = tmp_path / "test_labels.json"
        labels_data = [
            {
                "name": "Integration Test Label",
                "description": "Label for integration testing",
                "category": "test",
                "color": "#123456"
            }
        ]
        import_file.write_text(json.dumps(labels_data))
        
        # Import the labels
        result = runner.invoke(cli, ["label", "manage", "import", str(import_file)])
        
        assert result.exit_code == 0
        assert "Successfully imported 1 labels" in result.output
        
        # Verify the label was imported
        list_result = runner.invoke(cli, ["label", "manage", "list", "--category", "test"])
        
        assert list_result.exit_code == 0
        assert "Integration Test Label" in list_result.output