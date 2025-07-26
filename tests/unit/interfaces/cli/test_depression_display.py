"""
Test Depression Score Display in CLI

Following TDD principles - RED phase.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    PipelineResult,
)
from big_mood_detector.interfaces.cli.commands import predict_command


class TestDepressionScoreDisplay:
    """Test that depression scores are displayed in CLI output."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_pipeline_result(self):
        """Create a mock pipeline result with depression scores."""
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    "pat_depression_probability": 0.75,  # NEW: PAT depression score
                    "pat_confidence": 0.85,  # NEW: PAT confidence
                }
            },
            overall_summary={
                "avg_depression_risk": 0.3,
                "avg_hypomanic_risk": 0.2,
                "avg_manic_risk": 0.1,
                "days_analyzed": 1,
                "avg_pat_depression_probability": 0.75,  # NEW: Average PAT score
                "avg_pat_confidence": 0.85,  # NEW: Average PAT confidence
            },
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        return result
    
    def test_displays_pat_depression_score_in_summary(self, runner, mock_pipeline_result):
        """Should display PAT depression probability in summary output."""
        with patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline") as mock_pipeline:
            # Setup mock
            mock_instance = MagicMock()
            mock_instance.process_apple_health_file.return_value = mock_pipeline_result
            mock_pipeline.return_value = mock_instance
            
            # Create test directory with dummy file
            with runner.isolated_filesystem():
                test_dir = Path("test_data")
                test_dir.mkdir()
                test_file = test_dir / "Sleep Analysis.json"
                test_file.write_text("{}")
                
                # Run command
                result = runner.invoke(
                    predict_command,
                    ["test_data", "--ensemble"]
                )
                
                # Should succeed
                assert result.exit_code == 0
                
                # Check output contains PAT depression score
                assert "PAT Depression Risk:" in result.output
                assert "75.0%" in result.output or "0.75" in result.output
    
    def test_displays_pat_confidence_with_verbose(self, runner, mock_pipeline_result):
        """Should display PAT confidence when verbose flag is used."""
        with patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline") as mock_pipeline:
            # Setup mock
            mock_instance = MagicMock()
            mock_instance.process_apple_health_file.return_value = mock_pipeline_result
            mock_pipeline.return_value = mock_instance
            
            # Create test directory with dummy file
            with runner.isolated_filesystem():
                test_dir = Path("test_data")
                test_dir.mkdir()
                test_file = test_dir / "Sleep Analysis.json"
                test_file.write_text("{}")
                
                # Run command with verbose
                result = runner.invoke(
                    predict_command,
                    ["test_data", "--ensemble", "--verbose"]
                )
                
                # Should succeed
                assert result.exit_code == 0
                
                # Check output contains PAT confidence
                assert "PAT Confidence:" in result.output
                assert "85.0%" in result.output
    
    def test_json_output_includes_pat_scores(self):
        """JSON output should include PAT depression scores."""
        from big_mood_detector.interfaces.cli.commands import save_json_output
        
        # Create a pipeline result with PAT scores
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    "pat_depression_probability": 0.75,
                    "pat_confidence": 0.85,
                }
            },
            overall_summary={
                "avg_depression_risk": 0.3,
                "avg_hypomanic_risk": 0.2,
                "avg_manic_risk": 0.1,
                "days_analyzed": 1,
                "avg_pat_depression_probability": 0.75,
            },
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        
        # Save to JSON
        with CliRunner().isolated_filesystem():
            output_path = Path("output.json")
            save_json_output(result, output_path)
            
            # Check JSON file contains PAT scores
            output_data = json.loads(output_path.read_text())
            
            # Check summary includes average PAT score
            assert "avg_pat_depression_probability" in output_data["summary"]
            assert output_data["summary"]["avg_pat_depression_probability"] == 0.75
            
            # Check daily predictions include PAT scores
            daily_data = output_data["daily_predictions"]["2025-07-25"]
            assert "pat_depression_probability" in daily_data
            assert daily_data["pat_depression_probability"] == 0.75
            assert "pat_confidence" in daily_data
            assert daily_data["pat_confidence"] == 0.85
    
    def test_csv_output_includes_pat_columns(self):
        """CSV output should include PAT depression columns."""
        from big_mood_detector.interfaces.cli.commands import save_csv_output
        
        # Create a pipeline result with PAT scores
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    "pat_depression_probability": 0.75,
                    "pat_confidence": 0.85,
                }
            },
            overall_summary={
                "avg_depression_risk": 0.3,
                "avg_hypomanic_risk": 0.2,
                "avg_manic_risk": 0.1,
                "days_analyzed": 1,
                "avg_pat_depression_probability": 0.75,
            },
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        
        # Save to CSV
        with CliRunner().isolated_filesystem():
            output_path = Path("output.csv")
            save_csv_output(result, output_path)
            
            # Check CSV file contains PAT columns
            csv_content = output_path.read_text()
            lines = csv_content.strip().split("\n")
            
            # Check header
            header = lines[0]
            assert "pat_depression_probability" in header
            assert "pat_confidence" in header
            
            # Check data row
            data_row = lines[1]
            assert "0.75" in data_row
            assert "0.85" in data_row
    
    def test_handles_missing_pat_scores_gracefully(self, runner):
        """Should handle pipeline results without PAT scores gracefully."""
        # Create result without PAT scores
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    # No PAT scores
                }
            },
            overall_summary={
                "avg_depression_risk": 0.3,
                "avg_hypomanic_risk": 0.2, 
                "avg_manic_risk": 0.1,
                "days_analyzed": 1,
                # No average PAT score
            },
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        
        with patch("big_mood_detector.interfaces.cli.commands.MoodPredictionPipeline") as mock_pipeline:
            # Setup mock
            mock_instance = MagicMock()
            mock_instance.process_apple_health_file.return_value = result
            mock_pipeline.return_value = mock_instance
            
            # Create test directory with dummy file
            with runner.isolated_filesystem():
                test_dir = Path("test_data")
                test_dir.mkdir()
                test_file = test_dir / "Sleep Analysis.json"
                test_file.write_text("{}")
                
                # Run command
                result = runner.invoke(
                    predict_command,
                    ["test_data"]
                )
                
                # Should succeed
                assert result.exit_code == 0
                
                # Should not show PAT scores
                assert "PAT Depression Risk:" not in result.output
    
    def test_clinical_report_includes_pat_assessment(self):
        """Clinical report should include PAT depression assessment."""
        from big_mood_detector.interfaces.cli.commands import generate_clinical_report
        
        # Create a pipeline result with PAT scores
        result = PipelineResult(
            daily_predictions={
                date(2025, 7, 25): {
                    "depression_risk": 0.3,
                    "hypomanic_risk": 0.2,
                    "manic_risk": 0.1,
                    "confidence": 0.8,
                    "pat_depression_probability": 0.75,
                    "pat_confidence": 0.85,
                }
            },
            overall_summary={
                "avg_depression_risk": 0.3,
                "avg_hypomanic_risk": 0.2,
                "avg_manic_risk": 0.1,
                "days_analyzed": 1,
                "avg_pat_depression_probability": 0.75,
            },
            confidence_score=0.8,
            processing_time_seconds=2.5,
            records_processed=100,
            features_extracted=36,
        )
        
        # Generate report
        with CliRunner().isolated_filesystem():
            report_path = Path("clinical_report.txt")
            generate_clinical_report(result, report_path)
            
            # Check report contains PAT assessment
            report_content = report_path.read_text()
            assert "PAT Depression Assessment" in report_content
            assert "75.0%" in report_content
            assert "PHQ-9 ≥ 10" in report_content  # Clinical threshold reference