"""
Integration tests for the ensemble prediction system.

Tests both CLI and API interfaces with various scenarios.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from big_mood_detector.domain.entities.activity_record import ActivityRecord


class TestEnsemblePredictions:
    """Test ensemble predictions across different scenarios."""

    @pytest.fixture
    def mock_activity_data(self):
        """Generate mock 7-day activity data."""
        records = []
        base_date = date.today() - timedelta(days=7)

        for day in range(7):
            current_date = base_date + timedelta(days=day)
            for hour in range(24):
                # Simulate circadian rhythm
                if 23 <= hour or hour <= 6:
                    # Night time - low activity
                    intensity = np.random.uniform(0, 10)
                elif 12 <= hour <= 14:
                    # Lunch time - moderate activity
                    intensity = np.random.uniform(20, 40)
                else:
                    # Day time - higher activity
                    intensity = np.random.uniform(30, 80)

                for minute in range(60):
                    records.append(
                        ActivityRecord(
                            start_date=datetime.combine(
                                current_date, datetime.min.time()
                            )
                            + timedelta(hours=hour, minutes=minute),
                            activity_type="movement",
                            intensity=intensity + np.random.normal(0, 5),
                            duration_minutes=1.0,
                        )
                    )

        return records

    @pytest.fixture
    def mock_features(self):
        """Generate mock feature array."""
        return np.array(
            [
                7.5,  # sleep_duration_hours
                0.85,  # sleep_efficiency
                23.5,  # sleep_onset_hour
                7.0,  # wake_time_hour
                0.15,  # sleep_fragmentation
                85.0,  # sleep_regularity_index
                0.1,  # short_sleep_window_pct
                0.05,  # long_sleep_window_pct
                0.5,  # sleep_onset_variance
                0.3,  # wake_time_variance
                0.75,  # interdaily_stability
                0.45,  # intradaily_variability
                0.82,  # relative_amplitude
                10.5,  # l5_value
                85.2,  # m10_value
                2.5,  # l5_onset_hour
                14.5,  # m10_onset_hour
                22.0,  # dlmo_hour
                8500,  # total_steps
                150.0,  # activity_variance
                8.0,  # sedentary_hours
                0.25,  # activity_fragmentation
                45.0,  # sedentary_bout_mean
                2.5,  # activity_intensity_ratio
                65.0,  # avg_resting_hr
                35.0,  # hrv_sdnn
                20.0,  # hr_circadian_range
                4.5,  # hr_minimum_hour
                0.0,  # circadian_phase_advance
                0.5,  # circadian_phase_delay
                0.9,  # dlmo_confidence
                14.0,  # pat_hour
                0.1,  # sleep_duration_zscore
                -0.2,  # activity_zscore
                0.3,  # circadian_phase_zscore
                0.0,  # sleep_efficiency_zscore
            ],
            dtype=np.float32,
        )

    def test_ensemble_with_pat_available(self, mock_features, mock_activity_data):
        """Test ensemble prediction when PAT is available."""
        from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE

        if not PAT_AVAILABLE:
            pytest.skip("TensorFlow not installed")

        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleConfig,
            EnsembleOrchestrator,
        )
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostMoodPredictor,
        )

        # Initialize models
        xgboost_predictor = XGBoostMoodPredictor()
        assert xgboost_predictor.load_models(Path("model_weights/xgboost/pretrained"))

        pat_model = PATModel(model_size="medium")
        assert pat_model.load_pretrained_weights()

        # Create orchestrator
        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor,
            pat_model=pat_model,
            config=EnsembleConfig(),
        )

        # Make prediction
        result = orchestrator.predict(
            statistical_features=mock_features,
            activity_records=mock_activity_data,
            prediction_date=None,
        )

        # Verify results
        assert result.xgboost_prediction is not None
        assert result.pat_enhanced_prediction is not None
        assert result.ensemble_prediction is not None
        assert len(result.models_used) >= 2
        assert "xgboost" in result.models_used
        assert "pat_enhanced" in result.models_used

        # Check predictions are in valid range
        for pred in [
            result.xgboost_prediction,
            result.pat_enhanced_prediction,
            result.ensemble_prediction,
        ]:
            assert 0 <= pred.depression_risk <= 1
            assert 0 <= pred.hypomanic_risk <= 1
            assert 0 <= pred.manic_risk <= 1
            assert 0 <= pred.confidence <= 1

    @pytest.mark.xfail(
        reason="Issue #40: XGBoost Booster objects loaded from JSON lack predict_proba method",
        strict=True
    )
    def test_ensemble_without_pat(self, mock_features):
        """Test ensemble prediction when PAT is not available."""
        with patch("big_mood_detector.infrastructure.ml_models.PAT_AVAILABLE", False):
            from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
                EnsembleConfig,
                EnsembleOrchestrator,
            )
            from big_mood_detector.infrastructure.ml_models.xgboost_models import (
                XGBoostMoodPredictor,
            )

            # Initialize only XGBoost
            xgboost_predictor = XGBoostMoodPredictor()
            assert xgboost_predictor.load_models(
                Path("model_weights/xgboost/pretrained")
            )

            # Create orchestrator without PAT
            orchestrator = EnsembleOrchestrator(
                xgboost_predictor=xgboost_predictor,
                pat_model=None,
                config=EnsembleConfig(),
            )

            # Make prediction
            result = orchestrator.predict(
                statistical_features=mock_features,
                activity_records=None,
                prediction_date=None,
            )

            # Verify results
            assert result.xgboost_prediction is not None
            assert result.pat_enhanced_prediction is None
            assert result.ensemble_prediction is not None
            assert result.models_used == ["xgboost"]

            # Ensemble should equal XGBoost when PAT unavailable
            assert (
                result.ensemble_prediction.depression_risk
                == result.xgboost_prediction.depression_risk
            )

    @pytest.mark.asyncio
    async def test_api_ensemble_endpoint(self, mock_features):
        """Test the API ensemble endpoint."""
        from fastapi.testclient import TestClient

        from big_mood_detector.interfaces.api.main import app

        client = TestClient(app)

        # Prepare test data
        test_features = {
            "sleep_duration": 7.5,
            "sleep_efficiency": 0.85,
            "sleep_timing_variance": 0.3,
            "daily_steps": 8500,
            "activity_variance": 150.0,
            "sedentary_hours": 8.0,
            "interdaily_stability": 0.75,
            "intradaily_variability": 0.45,
            "relative_amplitude": 0.82,
            "resting_hr": 65.0,
            "hrv_rmssd": 35.0,
        }

        # Test status endpoint
        response = client.get("/api/v1/predictions/status")
        assert response.status_code == 200
        status = response.json()
        assert "xgboost_available" in status
        assert "pat_available" in status
        assert "ensemble_available" in status

        # Test ensemble prediction
        response = client.post(
            "/api/v1/predictions/predict/ensemble",
            json=test_features,
        )

        # Check response based on PAT availability
        from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE

        if PAT_AVAILABLE:
            assert response.status_code == 200
            result = response.json()
            assert "ensemble_prediction" in result
            assert "models_used" in result
            assert "clinical_summary" in result
            assert "recommendations" in result
        else:
            # Should return 501 if TensorFlow not installed
            assert response.status_code == 501
            assert "TensorFlow" in response.json()["detail"]

    def test_cli_predict_ensemble(self, tmp_path):
        """Test CLI predict command with ensemble flag."""
        from big_mood_detector.interfaces.cli.commands import predict_command

        runner = CliRunner()

        # Create dummy export file
        export_file = tmp_path / "export.xml"
        export_file.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<HealthData locale="en_US">\n'
            "</HealthData>"
        )

        # Run predict command with ensemble
        result = runner.invoke(
            predict_command,
            [
                str(export_file),
                "--ensemble",
                "--format",
                "json",
                "-o",
                str(tmp_path / "predictions.json"),
            ],
        )

        # Check result based on PAT availability
        from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE

        if PAT_AVAILABLE:
            # Should succeed but may have warnings about empty data
            assert result.exit_code == 0 or "No valid data" in result.output
        else:
            # Should still work, just without PAT
            assert result.exit_code == 0 or "No valid data" in result.output

    def test_ensemble_timeout_handling(self, mock_features):
        """Test that ensemble handles timeouts gracefully."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleConfig,
            EnsembleOrchestrator,
        )
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostMoodPredictor,
        )

        # Create config with very short timeout
        config = EnsembleConfig(
            xgboost_timeout=0.001,  # 1ms - will timeout
            pat_timeout=0.001,
            fallback_to_single_model=True,
        )

        xgboost_predictor = XGBoostMoodPredictor()
        assert xgboost_predictor.load_models(Path("model_weights/xgboost/pretrained"))

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor,
            pat_model=None,
            config=config,
        )

        # Should still return a result despite timeout
        with patch("time.sleep", side_effect=lambda x: None):
            result = orchestrator.predict(
                statistical_features=mock_features,
                activity_records=None,
                prediction_date=None,
            )

        assert result.ensemble_prediction is not None

    @pytest.mark.xfail(
        reason="Issue #40: XGBoost Booster objects loaded from JSON lack predict_proba method",
        strict=True
    )
    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_pat_model_sizes(self, model_size):
        """Test different PAT model sizes."""
        from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE

        if not PAT_AVAILABLE:
            pytest.skip("TensorFlow not installed")

        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        pat_model = PATModel(model_size=model_size)

        # Check if weights exist
        weights_path = Path(
            f"model_weights/pat/pretrained/PAT-{model_size[0].upper()}_29k_weights.h5"
        )

        if weights_path.exists():
            assert pat_model.load_pretrained_weights()
            assert pat_model.is_loaded

            # Verify model info
            info = pat_model.get_model_info()
            assert info["model_size"] == model_size
            assert info["is_loaded"] is True
