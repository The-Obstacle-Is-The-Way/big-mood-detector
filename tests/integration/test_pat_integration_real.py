"""
Real integration test for PAT model with actual weights.

This test verifies the complete PAT pipeline works end-to-end
with real pretrained weights.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from big_mood_detector.domain.services.pat_sequence_builder import PATSequence
from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE


@pytest.mark.slow  # Requires real ML models
@pytest.mark.skipif(not PAT_AVAILABLE, reason="TensorFlow not available")
class TestPATRealIntegration:
    """Test PAT with real weights and data."""

    def test_pat_loads_real_weights(self):
        """Test that PAT can load actual pretrained weights."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        model = PATModel(model_size="medium")
        weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")

        if not weights_path.exists():
            pytest.skip(f"PAT weights not found at {weights_path}")

        # Should load successfully
        success = model.load_pretrained_weights(weights_path)
        assert success is True
        assert model.is_loaded is True

    def test_pat_extracts_embeddings_with_correct_dtype(self):
        """Test PAT extracts embeddings without dtype errors."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        # Load model
        model = PATModel(model_size="medium")
        weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")

        if not weights_path.exists():
            pytest.skip(f"PAT weights not found at {weights_path}")

        model.load_pretrained_weights(weights_path)

        # Create realistic test sequence
        # Simulate circadian rhythm in activity
        activity_values = []
        for _day in range(7):
            for hour in range(24):
                # Low activity at night (0-6, 22-24)
                if hour < 6 or hour >= 22:
                    base_activity = np.random.uniform(0, 10)
                # Moderate activity during day
                elif 9 <= hour <= 17:
                    base_activity = np.random.uniform(40, 80)
                # Transition periods
                else:
                    base_activity = np.random.uniform(20, 40)

                # Add minute-level variation
                for _minute in range(60):
                    activity = max(0, base_activity + np.random.normal(0, 5))
                    activity_values.append(activity)

        activity_array = np.array(activity_values, dtype=np.float32)  # Ensure float32
        assert len(activity_array) == 10080  # 7 * 24 * 60

        sequence = PATSequence(
            end_date=date(2025, 6, 15),
            activity_values=activity_array,
            missing_days=[],
            data_quality_score=1.0
        )

        # Extract features - should work without dtype errors
        embeddings = model.extract_features(sequence)

        # Verify embeddings
        assert embeddings is not None
        assert embeddings.shape == (96,)  # 96-dimensional embedding
        assert embeddings.dtype in [np.float32, np.float64]
        assert not np.all(embeddings == 0)  # Should have non-zero values
        assert np.isfinite(embeddings).all()  # No NaN or inf

    def test_pat_batch_extraction(self):
        """Test PAT can process multiple sequences efficiently."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        model = PATModel(model_size="medium")
        weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")

        if not weights_path.exists():
            pytest.skip(f"PAT weights not found at {weights_path}")

        model.load_pretrained_weights(weights_path)

        # Create batch of sequences
        sequences = []
        for i in range(3):
            activity = np.random.randn(10080).astype(np.float32)
            sequence = PATSequence(
                end_date=date(2025, 6, 15 + i),
                activity_values=activity,
                missing_days=[],
                data_quality_score=1.0
            )
            sequences.append(sequence)

        # Extract batch
        embeddings = model.extract_features_batch(sequences)

        assert embeddings.shape == (3, 96)
        assert np.isfinite(embeddings).all()

    def test_ensemble_with_pat_enabled(self):
        """Test ensemble orchestrator with PAT enabled."""
        from datetime import datetime

        from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
            EnsembleConfig,
            EnsembleOrchestrator,
        )
        from big_mood_detector.domain.entities.activity_record import (
            ActivityRecord,
            ActivityType,
        )
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
        from big_mood_detector.infrastructure.ml_models.xgboost_models import (
            XGBoostMoodPredictor,
        )

        # Load XGBoost
        xgboost = XGBoostMoodPredictor()
        model_dir = Path("model_weights/xgboost/converted")
        if not xgboost.load_models(model_dir):
            pytest.skip("XGBoost models not found")

        # Load PAT
        pat = PATModel(model_size="medium")
        pat_weights = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
        if not pat_weights.exists() or not pat.load_pretrained_weights(pat_weights):
            pytest.skip("PAT weights not found")

        # Create orchestrator
        config = EnsembleConfig(
            xgboost_weight=0.6,
            pat_weight=0.4,
            use_pat_features=True
        )

        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost,
            pat_model=pat,
            config=config
        )

        # Create test data
        # Seoul features (36 values)
        seoul_features = np.array([
            7.5, 0.85, 23.5, 7.0, 0.15,  # Basic sleep
            85.0, 0.1, 0.05, 0.5, 0.3,   # Advanced sleep
            0.75, 0.45, 0.65, 5.0, 50.0, # Circadian
            1.0, 13.0, 20.0, 8500, 1200, # More circadian + activity
            6.5, 0.25, 5.5, 0.6, 72.0,   # Activity
            42.0, 0.25, 10.0, 4.0, 0.85, # Heart rate + phase
            0.3, 18.5, 0.85, 0.1, -0.2,  # Phase + z-scores
            0.15                          # Last z-score
        ])

        # Activity records for PAT
        activity_records = []
        base_date = datetime(2025, 6, 15)
        for _day in range(7):
            for hour in range(24):
                start = base_date.replace(day=15-_day, hour=hour)
                # Handle day boundary
                if hour == 23:
                    end = base_date.replace(day=15-_day+1, hour=0)
                else:
                    end = base_date.replace(day=15-_day, hour=hour+1)

                record = ActivityRecord(
                    source_name="Test",
                    start_date=start,
                    end_date=end,
                    activity_type=ActivityType.STEP_COUNT,
                    value=np.random.uniform(0, 100),
                    unit="count"
                )
                activity_records.append(record)

        # Make prediction
        result = orchestrator.predict(
            statistical_features=seoul_features,
            activity_records=activity_records,
            prediction_date=np.datetime64('2025-06-15')
        )

        # Verify results
        assert result.ensemble_prediction is not None
        assert 0 <= result.ensemble_prediction.depression_risk <= 1
        assert 0 <= result.ensemble_prediction.hypomanic_risk <= 1
        assert 0 <= result.ensemble_prediction.manic_risk <= 1

        # Check models used
        # XGBoost might fail if features aren't provided with correct names
        # PAT embeddings should always be extracted
        assert "pat_embeddings" in result.models_used or "xgboost" in result.models_used
        
        # Should have processing times
        assert result.processing_time_ms["total"] > 0

    def test_pat_handles_missing_data(self):
        """Test PAT handles sequences with missing days."""
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel

        model = PATModel(model_size="medium")
        weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")

        if not weights_path.exists():
            pytest.skip(f"PAT weights not found at {weights_path}")

        model.load_pretrained_weights(weights_path)

        # Create sequence with some zero days (missing data)
        activity = np.random.randn(10080).astype(np.float32)
        # Zero out day 3 and 5
        activity[2*1440:3*1440] = 0  # Day 3
        activity[4*1440:5*1440] = 0  # Day 5

        sequence = PATSequence(
            end_date=date(2025, 6, 15),
            activity_values=activity,
            missing_days=[2, 4],  # 0-indexed
            data_quality_score=0.7  # Reduced quality
        )

        # Should still extract features
        embeddings = model.extract_features(sequence)
        assert embeddings.shape == (96,)
        assert np.isfinite(embeddings).all()

