"""
Test the label management CLI commands.

Tests follow the patterns from reference CLI frameworks:
- Typer's CliRunner for testing
- Rich components for beautiful output
- Interactive prompts and confirmations
"""

import json
from typing import List
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from big_mood_detector.domain.entities.label import Label
from big_mood_detector.interfaces.cli.label_commands import app


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_label_service():
    """Mock the label service for testing."""
    # Mock at the module level where it's actually used
    with patch("big_mood_detector.interfaces.cli.label_commands.label_service") as mock_service:
        with patch("big_mood_detector.interfaces.cli.label_commands.label_repository") as mock_repo:
            yield mock_service


@pytest.fixture
def sample_labels() -> List[Label]:
    """Create sample labels for testing."""
    return [
        Label(
            id="label-1",
            name="Depression",
            description="Major depressive episode indicators",
            color="#5B6C8F",
            category="mood",
            metadata={"dsm5_code": "296.2x"}
        ),
        Label(
            id="label-2", 
            name="Mania",
            description="Manic episode indicators",
            color="#FF6B6B",
            category="mood",
            metadata={"dsm5_code": "296.4x"}
        ),
        Label(
            id="label-3",
            name="Sleep Disruption",
            description="Significant sleep pattern disruption",
            color="#4ECDC4",
            category="sleep",
            metadata={"threshold": "3.5h"}
        )
    ]


class TestLabelListCommand:
    """Test the 'label list' command."""
    
    def test_list_all_labels_as_table(self, runner, mock_label_service, sample_labels):
        """Test listing all labels in a beautiful table format."""
        # Arrange
        mock_label_service.list_labels.return_value = sample_labels
        
        # Act
        result = runner.invoke(app, ["list"])
        
        # Assert
        assert result.exit_code == 0
        assert "Depression" in result.output
        assert "Mania" in result.output
        assert "Sleep Disruption" in result.output
        assert "Major depressive episode" in result.output
        assert "#5B6C8F" in result.output
        
    def test_list_labels_by_category(self, runner, mock_label_service, sample_labels):
        """Test filtering labels by category."""
        # Arrange
        mood_labels = [l for l in sample_labels if l.category == "mood"]
        mock_label_service.list_labels.return_value = mood_labels
        
        # Act
        result = runner.invoke(app, ["list", "--category", "mood"])
        
        # Assert
        assert result.exit_code == 0
        assert "Depression" in result.output
        assert "Mania" in result.output
        assert "Sleep Disruption" not in result.output
        mock_label_service.list_labels.assert_called_once_with(category="mood")
        
    def test_list_labels_as_json(self, runner, mock_label_service, sample_labels):
        """Test JSON output format."""
        # Arrange
        mock_label_service.list_labels.return_value = sample_labels
        
        # Act
        result = runner.invoke(app, ["list", "--format", "json"])
        
        # Assert
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 3
        assert output_data[0]["name"] == "Depression"
        
    def test_empty_label_list(self, runner, mock_label_service):
        """Test graceful handling of empty label list."""
        # Arrange
        mock_label_service.list_labels.return_value = []
        
        # Act
        result = runner.invoke(app, ["list"])
        
        # Assert
        assert result.exit_code == 0
        assert "No labels found" in result.output


class TestLabelCreateCommand:
    """Test the 'label create' command."""
    
    def test_create_label_interactive(self, runner, mock_label_service):
        """Test creating a label with interactive prompts."""
        # Arrange
        created_label = Label(
            id="new-label",
            name="Hypomania",
            description="Hypomanic episode indicators",
            color="#FFA500",
            category="mood",
            metadata={}
        )
        mock_label_service.create_label.return_value = created_label
        
        # Act - simulate user input
        result = runner.invoke(
            app, 
            ["create"],
            input="Hypomania\nHypomanic episode indicators\nmood\n#FFA500\n"
        )
        
        # Assert
        assert result.exit_code == 0
        assert "Label 'Hypomania' created successfully" in result.output
        assert "#FFA500" in result.output
        
    def test_create_label_with_arguments(self, runner, mock_label_service):
        """Test creating a label with command line arguments."""
        # Arrange
        created_label = Label(
            id="new-label",
            name="Anxiety",
            description="Anxiety indicators",
            color="#9B59B6",
            category="mood",
            metadata={}
        )
        mock_label_service.create_label.return_value = created_label
        
        # Act
        result = runner.invoke(app, [
            "create",
            "--name", "Anxiety",
            "--description", "Anxiety indicators",
            "--category", "mood",
            "--color", "#9B59B6"
        ])
        
        # Assert
        assert result.exit_code == 0
        assert "Label 'Anxiety' created successfully" in result.output
        
    def test_create_label_duplicate_name(self, runner, mock_label_service):
        """Test error handling for duplicate label names."""
        # Arrange
        mock_label_service.create_label.side_effect = ValueError("Label with name 'Depression' already exists")
        
        # Act
        result = runner.invoke(app, [
            "create",
            "--name", "Depression",
            "--description", "Test",
            "--category", "mood",
            "--color", "#5B6C8F"
        ])
        
        # Assert
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "already exists" in result.output


class TestLabelSearchCommand:
    """Test the 'label search' command."""
    
    def test_search_labels_by_name(self, runner, mock_label_service, sample_labels):
        """Test searching labels by name."""
        # Arrange
        search_results = [l for l in sample_labels if "sleep" in l.name.lower()]
        mock_label_service.search_labels.return_value = search_results
        
        # Act
        result = runner.invoke(app, ["search", "sleep"])
        
        # Assert
        assert result.exit_code == 0
        assert "Sleep Disruption" in result.output
        assert "Depression" not in result.output
        # Should highlight the search term
        assert "sleep" in result.output.lower()
        
    def test_search_no_results(self, runner, mock_label_service):
        """Test search with no results."""
        # Arrange
        mock_label_service.search_labels.return_value = []
        
        # Act
        result = runner.invoke(app, ["search", "nonexistent"])
        
        # Assert
        assert result.exit_code == 0
        assert "No labels found matching 'nonexistent'" in result.output


class TestLabelDeleteCommand:
    """Test the 'label delete' command."""
    
    def test_delete_label_with_confirmation(self, runner, mock_label_service, sample_labels):
        """Test deleting a label with confirmation prompt."""
        # Arrange
        label_to_delete = sample_labels[0]
        mock_label_service.get_label.return_value = label_to_delete
        mock_label_service.delete_label.return_value = True
        
        # Act - simulate user confirming deletion
        result = runner.invoke(
            app,
            ["delete", "label-1"],
            input="y\n"
        )
        
        # Assert
        assert result.exit_code == 0
        assert "Are you sure you want to delete" in result.output
        assert "Depression" in result.output
        assert "deleted successfully" in result.output
        
    def test_delete_label_cancelled(self, runner, mock_label_service, sample_labels):
        """Test cancelling label deletion."""
        # Arrange
        label = sample_labels[0]
        mock_label_service.get_label.return_value = label
        
        # Act - simulate user cancelling
        result = runner.invoke(
            app,
            ["delete", "label-1"],
            input="n\n"
        )
        
        # Assert
        assert result.exit_code == 0
        assert "Deletion cancelled" in result.output
        mock_label_service.delete_label.assert_not_called()
        
    def test_delete_nonexistent_label(self, runner, mock_label_service):
        """Test deleting a label that doesn't exist."""
        # Arrange
        mock_label_service.get_label.return_value = None
        
        # Act
        result = runner.invoke(app, ["delete", "nonexistent"])
        
        # Assert
        assert result.exit_code == 1
        assert "Label 'nonexistent' not found" in result.output


class TestLabelUpdateCommand:
    """Test the 'label update' command."""
    
    def test_update_label_description(self, runner, mock_label_service, sample_labels):
        """Test updating a label's description."""
        # Arrange
        original_label = sample_labels[0]
        updated_label = Label(
            id=original_label.id,
            name=original_label.name,
            description="Updated description",
            color=original_label.color,
            category=original_label.category,
            metadata=original_label.metadata
        )
        mock_label_service.get_label.return_value = original_label
        mock_label_service.update_label.return_value = updated_label
        
        # Act
        result = runner.invoke(app, [
            "update", "label-1",
            "--description", "Updated description"
        ])
        
        # Assert
        assert result.exit_code == 0
        assert "Label 'Depression' updated successfully" in result.output
        assert "Updated description" in result.output


class TestLabelBatchOperations:
    """Test batch operations with progress bars."""
    
    def test_batch_import_labels(self, runner, mock_label_service, tmp_path, monkeypatch):
        """Test importing multiple labels from a file with progress bar."""
        # Arrange
        import_file = tmp_path / "labels.json"
        labels_data = [
            {
                "name": "Label1",
                "description": "Description1",
                "category": "mood",
                "color": "#FF0000"
            },
            {
                "name": "Label2", 
                "description": "Description2",
                "category": "sleep",
                "color": "#00FF00"
            }
        ]
        import_file.write_text(json.dumps(labels_data))
        
        # Mock the label creation
        created_labels = []
        def mock_create(name, description, category, color, metadata):
            label = Label(
                id=f"id-{len(created_labels)}",
                name=name,
                description=description,
                category=category,
                color=color,
                metadata=metadata
            )
            created_labels.append(label)
            return label
        
        mock_label_service.create_label.side_effect = mock_create
        
        # Act
        result = runner.invoke(app, ["import", str(import_file)])
        
        # Assert
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            print(f"Exception: {result.exception}")
            if result.exception:
                import traceback
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        
        assert result.exit_code == 0
        assert "Found 2 labels to import" in result.output
        assert "Successfully imported 2 labels" in result.output