"""
Test PAT model environment variable loading
"""

from unittest.mock import Mock, patch

import pytest

from big_mood_detector.infrastructure.ml_models.pat_model import PATModel


class TestPATModelEnvironmentLoading:
    """Test PAT model can load weights from environment variable."""

    def test_pat_model_loads_from_env(self, tmp_path, monkeypatch):
        """Test PAT model loads weights from BIG_MOOD_PAT_WEIGHTS_DIR."""
        # Create fake weights file
        weights_dir = tmp_path / "custom_weights"
        weights_dir.mkdir()
        weights_file = weights_dir / "PAT-M_29k_weights.h5"
        weights_file.touch()  # Create empty file

        # Set environment variable
        monkeypatch.setenv("BIG_MOOD_PAT_WEIGHTS_DIR", str(weights_dir))

        # Mock the actual loading to avoid needing real weights
        with patch.object(PATModel, "_load_weights_file") as mock_load:
            mock_load.return_value = True

            model = PATModel()

            # Should try to load from environment path
            assert model.load_pretrained_weights() is True
            mock_load.assert_called_once()

            # Check it tried the right path
            called_path = mock_load.call_args[0][0]
            assert str(weights_dir) in str(called_path)

    def test_pat_model_falls_back_to_default(self, monkeypatch):
        """Test PAT model falls back to default path when env not set."""
        # Ensure env var is not set
        monkeypatch.delenv("BIG_MOOD_PAT_WEIGHTS_DIR", raising=False)

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.object(PATModel, "_load_weights_file") as mock_load,
        ):
            mock_exists.return_value = True
            mock_load.return_value = False  # Simulate weights load failure

            model = PATModel()
            result = model.load_pretrained_weights()

            # Should return False when weights load fails
            assert result is False

            # Should have tried default path
            called_path = mock_load.call_args[0][0]
            assert "model_weights/pat/pretrained" in str(called_path)

    def test_pat_model_handles_invalid_env_path(self, monkeypatch):
        """Test PAT model handles invalid environment path gracefully."""
        # Set invalid path
        monkeypatch.setenv("BIG_MOOD_PAT_WEIGHTS_DIR", "/nonexistent/path")

        model = PATModel()

        # Should not crash
        result = model.load_pretrained_weights()
        assert result is False
        assert not model.is_loaded

    def test_ensemble_degradation_without_pat(self, tmp_path, monkeypatch):
        """Test ensemble can degrade gracefully without PAT weights."""
        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleOrchestrator,
        )
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostMoodPredictor,
        )

        # Mock XGBoost predictor
        xgboost_predictor = Mock(spec=XGBoostMoodPredictor)
        xgboost_predictor.is_loaded = True
        xgboost_predictor.predict.return_value = Mock(
            depression_risk=0.3, hypomanic_risk=0.1, manic_risk=0.05, confidence=0.8
        )

        # Create PAT model that fails to load
        monkeypatch.setenv("BIG_MOOD_PAT_WEIGHTS_DIR", "/invalid/path")
        pat_model = PATModel()

        # Create ensemble
        ensemble = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor, pat_model=pat_model
        )

        # Should still work with XGBoost only
        import numpy as np

        result = ensemble.predict(
            statistical_features=np.zeros(36), activity_records=[]
        )

        assert result is not None
        assert result.xgboost_prediction is not None
        assert result.ensemble_prediction is not None
        assert "xgboost" in result.models_used

    @pytest.mark.parametrize(
        "env_value,expected_path",
        [
            ("/custom/path", "/custom/path/PAT-M_29k_weights.h5"),
            ("./relative/path", "./relative/path/PAT-M_29k_weights.h5"),
            (
                "",
                "model_weights/pat/pretrained/PAT-M_29k_weights.h5",
            ),  # Empty = default
        ],
    )
    def test_pat_model_path_resolution(self, env_value, expected_path, monkeypatch):
        """Test PAT model resolves paths correctly from environment."""
        if env_value:
            monkeypatch.setenv("BIG_MOOD_PAT_WEIGHTS_DIR", env_value)
        else:
            monkeypatch.delenv("BIG_MOOD_PAT_WEIGHTS_DIR", raising=False)

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.object(PATModel, "_load_weights_file") as mock_load,
        ):
            mock_exists.return_value = True
            mock_load.return_value = True

            model = PATModel()
            model.load_pretrained_weights()

            # Verify the constructed path
            called_path = mock_load.call_args[0][0]
            # For relative paths, Path normalizes away the leading ./
            if expected_path.startswith("./"):
                assert expected_path[2:] in str(called_path)
            else:
                assert expected_path in str(called_path)
