"""
Mood Predictor Service

Loads pre-trained XGBoost models and makes mood episode predictions.
Based on Seoul National University study models.

Design Principles:
- Immutable prediction results
- Model loading with validation
- Clear error handling
"""

import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

# Suppress XGBoost warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


@dataclass(frozen=True)
class MoodPrediction:
    """
    Immutable mood prediction result.

    Contains risk probabilities for different mood episode types.
    """

    depression_risk: float  # 0-1 probability of depression episode
    hypomanic_risk: float  # 0-1 probability of hypomanic episode
    manic_risk: float  # 0-1 probability of manic episode
    confidence: float  # Overall confidence in predictions

    @property
    def highest_risk_type(self) -> str:
        """Return the mood type with highest risk."""
        risks = {
            "depression": self.depression_risk,
            "hypomanic": self.hypomanic_risk,
            "manic": self.manic_risk,
        }
        return max(risks, key=lambda k: risks[k])

    @property
    def highest_risk_value(self) -> float:
        """Return the highest risk value."""
        return max(self.depression_risk, self.hypomanic_risk, self.manic_risk)

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for serialization."""
        return {
            "depression_risk": float(round(self.depression_risk, 4)),
            "hypomanic_risk": float(round(self.hypomanic_risk, 4)),
            "manic_risk": float(round(self.manic_risk, 4)),
            "confidence": float(round(self.confidence, 4)),
            "highest_risk_type": self.highest_risk_type,
            "highest_risk_value": float(round(self.highest_risk_value, 4)),
        }


class MoodPredictor:
    """
    Predicts mood episodes using pre-trained XGBoost models.

    Loads the Seoul National University models for:
    - Depression Episode (DE)
    - Hypomanic Episode (HME)
    - Manic Episode (ME)
    """

    def __init__(self, model_dir: Path | None = None):
        """
        Initialize predictor with model directory.

        Args:
            model_dir: Directory containing XGBoost .pkl files
                      Defaults to model_weights/xgboost/pretrained
        """
        if model_dir is None:
            # Try to use settings if available, otherwise fall back to environment variable
            try:
                from big_mood_detector.infrastructure.settings.config import (
                    get_settings,
                )

                settings = get_settings()
                # Check in the root model_weights directory first, then in data directory
                model_dir = Path("model_weights/xgboost/converted")
                if not model_dir.exists():
                    model_dir = (
                        settings.DATA_DIR / "model_weights" / "xgboost" / "converted"
                    )
            except ImportError:
                # Fallback for tests or when settings module is not available
                model_path = os.environ.get(
                    "XGBOOST_MODEL_PATH", "model_weights/xgboost/converted"
                )
                if os.path.isabs(model_path):
                    model_dir = Path(model_path)
                else:
                    base_path = Path(
                        os.path.dirname(__file__)
                    ).parent.parent.parent.parent
                    model_dir = base_path / model_path

        self.model_dir = Path(model_dir)
        self.models: dict[str, Any] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load XGBoost models - prefer JSON format to avoid warnings."""
        # Check for JSON models first to avoid pickle warnings
        # If model_dir is already 'converted', use it; otherwise look for converted dir
        if self.model_dir.name == "converted":
            converted_dir = self.model_dir
        else:
            converted_dir = self.model_dir.parent / "converted"
        json_models = {
            "depression": "XGBoost_DE.json",
            "hypomanic": "XGBoost_HME.json",
            "manic": "XGBoost_ME.json",
        }
        pkl_models = {
            "depression": "depression_model.pkl",
            "hypomanic": "hypomanic_model.pkl",
            "manic": "manic_model.pkl",
        }

        for mood_type in ["depression", "hypomanic", "manic"]:
            # Try JSON format first (no warnings)
            json_path = converted_dir / json_models[mood_type]
            pkl_path = self.model_dir / pkl_models[mood_type]

            if json_path.exists():
                try:
                    self.models[mood_type] = xgb.Booster()
                    self.models[mood_type].load_model(str(json_path))
                    print(
                        f"Loaded {mood_type} model from {json_models[mood_type]} (JSON format)"
                    )
                    continue
                except Exception as e:
                    print(f"Failed to load JSON model: {e}")

            # Fall back to pickle format
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        self.models[mood_type] = pickle.load(f)
                        print(f"Loaded {mood_type} model from {pkl_models[mood_type]}")
                except Exception as e:
                    print(f"Error loading {mood_type} model: {e}")
            else:
                print(f"Warning: No model found for {mood_type}")

    def predict(self, features: np.ndarray) -> MoodPrediction:
        """
        Predict mood episode risks from features.

        Args:
            features: 36-element feature vector from AdvancedFeatures

        Returns:
            MoodPrediction with risk probabilities

        Raises:
            ValueError: If features are invalid or models not loaded
        """
        # Ensure features is a numpy array
        features = (
            np.array(features) if not isinstance(features, np.ndarray) else features
        )

        if features.shape != (36,):
            raise ValueError(f"Expected 36 features, got {features.shape}")

        if not self.models:
            raise ValueError("No models loaded")

        # Reshape for XGBoost (needs 2D array)
        features_2d = features.reshape(1, -1)

        # Get predictions from each model
        predictions = {}
        for mood_type, model in self.models.items():
            try:
                # Handle different model types
                if isinstance(model, xgb.Booster):
                    # JSON-loaded Booster - use DMatrix with feature names
                    feature_names = [
                        "ST_long_MN",
                        "ST_long_SD",
                        "ST_long_Zscore",
                        "ST_short_MN",
                        "ST_short_SD",
                        "ST_short_Zscore",
                        "WT_long_MN",
                        "WT_long_SD",
                        "WT_long_Zscore",
                        "WT_short_MN",
                        "WT_short_SD",
                        "WT_short_Zscore",
                        "LongSleepWindow_length_MN",
                        "LongSleepWindow_length_SD",
                        "LongSleepWindow_length_Zscore",
                        "LongSleepWindow_number_MN",
                        "LongSleepWindow_number_SD",
                        "LongSleepWindow_number_Zscore",
                        "ShortSleepWindow_length_MN",
                        "ShortSleepWindow_length_SD",
                        "ShortSleepWindow_length_Zscore",
                        "ShortSleepWindow_number_MN",
                        "ShortSleepWindow_number_SD",
                        "ShortSleepWindow_number_Zscore",
                        "Sleep_percentage_MN",
                        "Sleep_percentage_SD",
                        "Sleep_percentage_Zscore",
                        "Sleep_amplitude_MN",
                        "Sleep_amplitude_SD",
                        "Sleep_amplitude_Zscore",
                        "Circadian_phase_MN",
                        "Circadian_phase_SD",
                        "Circadian_phase_Zscore",
                        "Circadian_amplitude_MN",
                        "Circadian_amplitude_SD",
                        "Circadian_amplitude_Zscore",
                    ]
                    dmatrix = xgb.DMatrix(features_2d, feature_names=feature_names)
                    # Booster.predict returns raw scores (probabilities for binary classification)
                    predictions[mood_type] = float(model.predict(dmatrix)[0])
                elif hasattr(model, "predict_proba"):
                    # Scikit-learn style model (XGBClassifier)
                    proba = model.predict_proba(features_2d)[0]
                    # XGBoost returns [prob_negative, prob_positive]
                    predictions[mood_type] = proba[1] if len(proba) > 1 else proba[0]
                else:
                    # Fallback for other model types
                    predictions[mood_type] = float(model.predict(features_2d)[0])
            except Exception as e:
                print(f"Error predicting {mood_type}: {e}")
                predictions[mood_type] = 0.0

        # Calculate confidence based on feature quality
        # (In practice, this would consider data completeness, recency, etc.)
        confidence = self._calculate_confidence(features)

        return MoodPrediction(
            depression_risk=predictions.get("depression", 0.0),
            hypomanic_risk=predictions.get("hypomanic", 0.0),
            manic_risk=predictions.get("manic", 0.0),
            confidence=confidence,
        )

    def predict_batch(self, feature_list: list[np.ndarray]) -> list[MoodPrediction]:
        """
        Predict mood risks for multiple days.

        Args:
            feature_list: List of 36-element feature vectors

        Returns:
            List of MoodPrediction objects
        """
        return [self.predict(features) for features in feature_list]

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """
        Calculate prediction confidence based on feature quality.

        Simple heuristic based on non-zero features and reasonable ranges.
        """
        # Count non-zero features
        non_zero_count = np.count_nonzero(features)

        # Check for reasonable ranges (z-scores should be mostly -3 to 3)
        z_score_indices = [
            i for i in range(len(features)) if i % 3 == 2
        ]  # Every 3rd feature is z-score
        z_scores = features[z_score_indices]
        reasonable_z_scores = np.sum(np.abs(z_scores) < 3)

        # Simple confidence calculation
        feature_quality = non_zero_count / 36.0
        z_score_quality = reasonable_z_scores / len(z_score_indices)

        confidence = (feature_quality + z_score_quality) / 2.0

        return float(min(max(confidence, 0.0), 1.0))

    @property
    def is_loaded(self) -> bool:
        """Check if models are successfully loaded."""
        return len(self.models) > 0

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models."""
        info = {}
        for mood_type, model in self.models.items():
            info[mood_type] = {
                "type": type(model).__name__,
                "n_features": getattr(model, "n_features_", 36),
                "loaded": True,
            }
        return info
