"""
Test Prediction CLI Command

TDD for mood prediction command line interface.
"""

import argparse
import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from big_mood_detector.application.use_cases.process_health_data_use_case import (
    PipelineResult,
)


class TestPredictCLI:
    """Test prediction command line interface."""

    def test_predict_command_exists(self):
        """Test that predict command can be imported."""
        from big_mood_detector.cli import predict

        assert predict is not None
        assert hasattr(predict, "main")

    def test_predict_command_help(self):
        """Test predict command shows help."""
        from big_mood_detector.cli.predict import main

        with patch("sys.argv", ["predict", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_predict_from_xml_file(self, tmp_path, mock_pipeline_result):
        """Test prediction from XML export."""
        from big_mood_detector.cli.predict import main

        # Create dummy XML file
        xml_file = tmp_path / "export.xml"
        xml_file.write_text("<HealthData></HealthData>")

        # Mock the pipeline
        with patch("sys.argv", ["predict", "--input", str(xml_file)]):
            with patch(
                "big_mood_detector.cli.predict.MoodPredictionPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.process_apple_health_file.return_value = (
                    mock_pipeline_result
                )

                # Run command
                main()

                # Verify pipeline was called
                mock_pipeline.process_apple_health_file.assert_called_once()
                call_kwargs = mock_pipeline.process_apple_health_file.call_args.kwargs
                assert str(call_kwargs["file_path"]) == str(xml_file)

    def test_predict_with_date_range(self, tmp_path, mock_pipeline_result):
        """Test prediction with date filtering."""
        from big_mood_detector.cli.predict import main

        xml_file = tmp_path / "export.xml"
        xml_file.write_text("<HealthData></HealthData>")

        with patch(
            "sys.argv",
            [
                "predict",
                "--input",
                str(xml_file),
                "--start-date",
                "2024-01-01",
                "--end-date",
                "2024-01-31",
            ],
        ):
            with patch(
                "big_mood_detector.cli.predict.MoodPredictionPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.process_apple_health_file.return_value = (
                    mock_pipeline_result
                )

                main()

                # Verify dates were passed
                _, kwargs = mock_pipeline.process_apple_health_file.call_args
                assert kwargs["start_date"] == date(2024, 1, 1)
                assert kwargs["end_date"] == date(2024, 1, 31)

    def test_predict_output_formats(self, tmp_path, mock_pipeline_result):
        """Test different output formats."""
        from big_mood_detector.cli.predict import main

        xml_file = tmp_path / "export.xml"
        xml_file.write_text("<HealthData></HealthData>")
        output_file = tmp_path / "predictions.json"

        # Test JSON output
        with patch(
            "sys.argv",
            [
                "predict",
                "--input",
                str(xml_file),
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        ):
            with patch(
                "big_mood_detector.cli.predict.MoodPredictionPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.process_apple_health_file.return_value = (
                    mock_pipeline_result
                )

                main()

                # Check output file was created
                assert output_file.exists()
                data = json.loads(output_file.read_text())
                assert "daily_predictions" in data
                assert "overall_summary" in data

    def test_predict_with_ensemble(self, tmp_path, mock_pipeline_result):
        """Test prediction with ensemble models."""
        from big_mood_detector.cli.predict import main

        xml_file = tmp_path / "export.xml"
        xml_file.write_text("<HealthData></HealthData>")

        with patch("sys.argv", ["predict", "--input", str(xml_file), "--ensemble"]):
            with patch(
                "big_mood_detector.cli.predict.MoodPredictionPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.process_apple_health_file.return_value = (
                    mock_pipeline_result
                )

                main()

                # Should create pipeline (ensemble will be handled differently)
                mock_pipeline_class.assert_called_once()

    def test_predict_with_personal_model(self, tmp_path, mock_pipeline_result):
        """Test prediction with personalized model."""
        from big_mood_detector.cli.predict import main

        xml_file = tmp_path / "export.xml"
        xml_file.write_text("<HealthData></HealthData>")

        # Create fake personal model
        personal_model_dir = tmp_path / "models" / "users" / "test_user"
        personal_model_dir.mkdir(parents=True)
        (personal_model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "user_id": "test_user",
                    "model_type": "xgboost",
                    "baseline": {"mean_sleep_duration": 420},
                }
            )
        )

        with patch(
            "sys.argv",
            [
                "predict",
                "--input",
                str(xml_file),
                "--user-id",
                "test_user",
                "--model-dir",
                str(tmp_path / "models"),
            ],
        ):
            with patch(
                "big_mood_detector.cli.predict.MoodPredictionPipeline"
            ) as mock_pipeline_class:
                with patch(
                    "big_mood_detector.cli.predict.PersonalCalibrator"
                ) as mock_calibrator_class:
                    mock_pipeline = Mock()
                    mock_pipeline_class.return_value = mock_pipeline
                    mock_pipeline.process_apple_health_file.return_value = (
                        mock_pipeline_result
                    )

                    main()

                    # Should load personal calibrator
                    mock_calibrator_class.load.assert_called_once_with(
                        user_id="test_user", model_dir=tmp_path / "models"
                    )

    def test_predict_summary_output(self, tmp_path, capsys):
        """Test prediction summary printed to console."""
        from big_mood_detector.cli.predict import main

        xml_file = tmp_path / "export.xml"
        xml_file.write_text("<HealthData></HealthData>")

        # Create result with predictions
        result = PipelineResult(
            daily_predictions={
                date(2024, 1, 1): {
                    "depression_risk": 0.75,
                    "hypomanic_risk": 0.10,
                    "manic_risk": 0.05,
                    "confidence": 0.85,
                },
                date(2024, 1, 2): {
                    "depression_risk": 0.60,
                    "hypomanic_risk": 0.15,
                    "manic_risk": 0.08,
                    "confidence": 0.85,
                },
            },
            overall_summary={
                "avg_depression_risk": 0.675,
                "avg_hypomanic_risk": 0.125,
                "avg_manic_risk": 0.065,
                "days_analyzed": 2,
            },
            confidence_score=0.85,
            processing_time_seconds=1.23,
            records_processed=1000,
            features_extracted=36,
        )

        with patch("sys.argv", ["predict", "--input", str(xml_file)]):
            with patch(
                "big_mood_detector.cli.predict.MoodPredictionPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.process_apple_health_file.return_value = result

                main()

                # Check output
                captured = capsys.readouterr()
                assert "MOOD PREDICTION RESULTS" in captured.out
                assert "2024-01-01" in captured.out
                assert "Depression: 75.0%" in captured.out
                assert "Overall Risk Summary" in captured.out
                assert "Confidence: 85.0%" in captured.out

    def test_predict_clinical_report(self, tmp_path):
        """Test generation of clinical report."""
        from big_mood_detector.cli.predict import main

        xml_file = tmp_path / "export.xml"
        xml_file.write_text("<HealthData></HealthData>")
        report_file = tmp_path / "clinical_report.md"

        with patch(
            "sys.argv",
            [
                "predict",
                "--input",
                str(xml_file),
                "--clinical-report",
                str(report_file),
            ],
        ):
            with patch(
                "big_mood_detector.cli.predict.MoodPredictionPipeline"
            ) as mock_pipeline_class:
                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.process_apple_health_file.return_value = PipelineResult(
                    daily_predictions={
                        date(2024, 1, 1): {
                            "depression_risk": 0.75,
                            "hypomanic_risk": 0.10,
                            "manic_risk": 0.05,
                            "confidence": 0.85,
                        }
                    },
                    overall_summary={
                        "avg_depression_risk": 0.75,
                        "avg_hypomanic_risk": 0.10,
                        "avg_manic_risk": 0.05,
                        "days_analyzed": 1,
                    },
                    confidence_score=0.85,
                    processing_time_seconds=1.0,
                )

                main()

                # Check report was created
                assert report_file.exists()
                content = report_file.read_text()
                assert "Clinical Assessment Report" in content
                assert "Depression Risk: HIGH" in content

    @pytest.fixture
    def mock_pipeline_result(self):
        """Create mock pipeline result."""
        return PipelineResult(
            daily_predictions={
                date(2024, 1, 1): {
                    "depression_risk": 0.25,
                    "hypomanic_risk": 0.10,
                    "manic_risk": 0.05,
                    "confidence": 0.85,
                }
            },
            overall_summary={
                "avg_depression_risk": 0.25,
                "avg_hypomanic_risk": 0.10,
                "avg_manic_risk": 0.05,
                "days_analyzed": 1,
            },
            confidence_score=0.85,
            processing_time_seconds=1.0,
            records_processed=100,
            features_extracted=36,
        )
