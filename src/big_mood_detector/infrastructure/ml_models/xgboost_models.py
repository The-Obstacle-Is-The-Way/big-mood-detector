"""
XGBoost Model Infrastructure

Provides infrastructure for loading and using XGBoost models for mood prediction.
Implements the domain MoodPredictor interface with concrete XGBoost models.

Based on the Seoul National University and Harvard/Fitbit studies that use
36 engineered features for mood episode prediction.
"""

import logging
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np

from big_mood_detector.domain.services.mood_predictor import MoodPrediction

logger = logging.getLogger(__name__)


class XGBoostModelLoader:
    """
    Loads and manages XGBoost models for mood prediction.

    Handles three models:
    - Depression risk (PHQ-9 >= 10)
    - Hypomanic episode risk
    - Manic episode risk
    """

    # Expected feature names based on the papers
    FEATURE_NAMES = [
        # Sleep percentage features (mean, std, z-score)
        "sleep_percentage_MN",
        "sleep_percentage_SD",
        "sleep_percentage_Z",
        # Sleep amplitude features
        "sleep_amplitude_MN",
        "sleep_amplitude_SD",
        "sleep_amplitude_Z",
        # Long sleep window features
        "long_num_MN",
        "long_num_SD",
        "long_num_Z",
        "long_len_MN",
        "long_len_SD",
        "long_len_Z",
        "long_ST_MN",
        "long_ST_SD",
        "long_ST_Z",
        "long_WT_MN",
        "long_WT_SD",
        "long_WT_Z",
        # Short sleep window features
        "short_num_MN",
        "short_num_SD",
        "short_num_Z",
        "short_len_MN",
        "short_len_SD",
        "short_len_Z",
        "short_ST_MN",
        "short_ST_SD",
        "short_ST_Z",
        "short_WT_MN",
        "short_WT_SD",
        "short_WT_Z",
        # Circadian rhythm features
        "circadian_amplitude_MN",
        "circadian_amplitude_SD",
        "circadian_amplitude_Z",
        "circadian_phase_MN",
        "circadian_phase_SD",
        "circadian_phase_Z",
    ]

    def __init__(self) -> None:
        """Initialize the model loader."""
        self.models: dict[str, Any] = {}
        self.feature_names = self.FEATURE_NAMES
        self.is_loaded = False

        logger.info("Initialized XGBoost model loader")

    def load_model(self, model_type: str, model_path: Path) -> bool:
        """
        Load a single XGBoost model.

        Args:
            model_type: One of "depression", "hypomanic", or "manic"
            model_path: Path to the model file (JSON or PKL)

        Returns:
            True if successful, False otherwise
        """
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        try:
            # Handle JSON format (preferred)
            if model_path.suffix == ".json":
                import xgboost as xgb

                model = xgb.Booster()
                model.load_model(str(model_path))
                self.models[model_type] = model
                logger.info(
                    f"Successfully loaded {model_type} model from {model_path} (JSON format)"
                )
            # Handle PKL format (legacy)
            else:
                model = joblib.load(model_path)
                self.models[model_type] = model
                logger.info(
                    f"Successfully loaded {model_type} model from {model_path} (PKL format)"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return False

    def load_all_models(self, model_dir: Path) -> dict[str, bool]:
        """
        Load all three mood prediction models.

        Args:
            model_dir: Directory containing model files

        Returns:
            Dictionary of load results for each model
        """
        results = {}

        # Expected model files - using actual JSON filenames
        model_files = {
            "depression": "XGBoost_DE.json",
            "hypomanic": "XGBoost_HME.json",
            "manic": "XGBoost_ME.json",
        }

        for model_type, filename in model_files.items():
            model_path = model_dir / filename
            results[model_type] = self.load_model(model_type, model_path)

        # Update loaded status
        self.is_loaded = all(results.values())

        if self.is_loaded:
            logger.info("All models loaded successfully")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"Failed to load models: {failed}")

        return results

    def predict(self, features: np.ndarray) -> MoodPrediction:
        """
        Make mood predictions using all loaded models.

        Args:
            features: Feature vector of shape (36,)

        Returns:
            MoodPrediction with risk scores
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_all_models first.")

        # Validate features
        self._validate_features(features)

        # Reshape for sklearn
        features_2d = features.reshape(1, -1)

        # Get predictions from each model
        depression_proba = self.models["depression"].predict_proba(features_2d)[0]
        hypomanic_proba = self.models["hypomanic"].predict_proba(features_2d)[0]
        manic_proba = self.models["manic"].predict_proba(features_2d)[0]

        # Extract positive class probabilities (usually index 1)
        depression_risk = (
            depression_proba[1] if len(depression_proba) > 1 else depression_proba[0]
        )
        hypomanic_risk = (
            hypomanic_proba[1] if len(hypomanic_proba) > 1 else hypomanic_proba[0]
        )
        manic_risk = manic_proba[1] if len(manic_proba) > 1 else manic_proba[0]

        # Calculate confidence based on prediction strength
        max_risk = max(depression_risk, hypomanic_risk, manic_risk)
        confidence = self._calculate_confidence(max_risk)

        return MoodPrediction(
            depression_risk=float(depression_risk),
            hypomanic_risk=float(hypomanic_risk),
            manic_risk=float(manic_risk),
            confidence=float(confidence),
        )

    def predict_batch(self, features_batch: np.ndarray) -> list[MoodPrediction]:
        """
        Make predictions for multiple samples.

        Args:
            features_batch: Feature matrix of shape (n_samples, 36)

        Returns:
            List of MoodPrediction objects
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_all_models first.")

        predictions = []

        # Get batch predictions from each model
        depression_probas = self.models["depression"].predict_proba(features_batch)
        hypomanic_probas = self.models["hypomanic"].predict_proba(features_batch)
        manic_probas = self.models["manic"].predict_proba(features_batch)

        # Process each sample
        for i in range(len(features_batch)):
            depression_risk = (
                depression_probas[i][1]
                if depression_probas.shape[1] > 1
                else depression_probas[i][0]
            )
            hypomanic_risk = (
                hypomanic_probas[i][1]
                if hypomanic_probas.shape[1] > 1
                else hypomanic_probas[i][0]
            )
            manic_risk = (
                manic_probas[i][1] if manic_probas.shape[1] > 1 else manic_probas[i][0]
            )

            max_risk = max(depression_risk, hypomanic_risk, manic_risk)
            confidence = self._calculate_confidence(max_risk)

            predictions.append(
                MoodPrediction(
                    depression_risk=float(depression_risk),
                    hypomanic_risk=float(hypomanic_risk),
                    manic_risk=float(manic_risk),
                    confidence=float(confidence),
                )
            )

        return predictions

    def _validate_features(self, features: np.ndarray) -> None:
        """
        Validate feature vector.

        Args:
            features: Feature vector to validate

        Raises:
            ValueError if features are invalid
        """
        if len(features) != 36:
            raise ValueError(f"Expected 36 features, got {len(features)}")

    def _calculate_confidence(self, max_risk: float) -> float:
        """
        Calculate confidence score based on prediction strength.

        Args:
            max_risk: Maximum risk score among all mood types

        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence based on how far from 0.5 the prediction is
        # Could be enhanced with model uncertainty estimates
        return abs(max_risk - 0.5) * 2

    def dict_to_array(self, feature_dict: dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to array in correct order.

        Args:
            feature_dict: Dictionary with feature names as keys

        Returns:
            Feature array in correct order
        """
        return np.array([feature_dict[name] for name in self.feature_names])

    def get_model_info(self) -> dict:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        return {
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "models_loaded": list(self.models.keys()),
            "is_loaded": self.is_loaded,
        }


class XGBoostMoodPredictor:
    """
    Concrete implementation of the domain MoodPredictor interface.

    This class bridges the domain service with the infrastructure XGBoost models.
    """

    def __init__(self) -> None:
        """Initialize the mood predictor."""
        self.model_loader = XGBoostModelLoader()

    def load_models(self, model_dir: Path) -> dict[str, bool]:
        """
        Load mood prediction models.

        Args:
            model_dir: Directory containing model files

        Returns:
            Dictionary of load results
        """
        return self.model_loader.load_all_models(model_dir)

    def predict(self, features: np.ndarray | dict[str, float]) -> MoodPrediction:
        """
        Predict mood episode risks.

        Args:
            features: Either a numpy array of 36 features or a feature dictionary

        Returns:
            MoodPrediction with risk scores
        """
        # Convert dict to array if needed
        if isinstance(features, dict):
            features = self.model_loader.dict_to_array(features)

        return self.model_loader.predict(features)

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self.model_loader.is_loaded

    def get_model_info(self) -> dict:
        """Get model information."""
        return self.model_loader.get_model_info()
