"""
Test PAT Predictor Interface

Tests for the domain interface that PAT predictors must implement.
This ensures we can swap implementations without changing domain logic.
"""

import numpy as np
import pytest

from big_mood_detector.domain.services.pat_predictor import PATBinaryPredictions


class TestPATPredictorInterface:
    """Test the PAT predictor interface contract."""

    def test_pat_predictor_interface_exists(self):
        """PAT predictor interface should be defined in domain layer."""
        from big_mood_detector.domain.services.pat_predictor import (
            PATPredictorInterface,
        )

        assert hasattr(PATPredictorInterface, 'predict_from_embeddings')

    def test_pat_predictor_returns_binary_predictions(self):
        """PAT predictor should return standard MoodPrediction object."""
        from big_mood_detector.domain.services.pat_predictor import (
            PATPredictorInterface,
        )

        # Create a concrete implementation for testing
        class ConcretePATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> PATBinaryPredictions:
                # Simple implementation for testing
                return PATBinaryPredictions(
                    depression_probability=0.3,
                    benzodiazepine_probability=0.2,
                    confidence=0.8
                )
            
            def predict_depression(self, embeddings: np.ndarray) -> float:
                return 0.3
            
            def predict_medication_proxy(self, embeddings: np.ndarray) -> float:
                return 0.2

        predictor = ConcretePATPredictor()
        embeddings = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(embeddings)

        assert isinstance(prediction, PATBinaryPredictions)
        assert 0 <= prediction.depression_probability <= 1
        assert 0 <= prediction.benzodiazepine_probability <= 1
        assert 0 <= prediction.confidence <= 1

    def test_pat_predictor_validates_embedding_dimension(self):
        """PAT predictor should validate embedding dimensions."""
        from big_mood_detector.domain.services.pat_predictor import (
            PATPredictorInterface,
        )

        from big_mood_detector.domain.services.pat_predictor import PATBinaryPredictions
        
        class StrictPATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> PATBinaryPredictions:
                if embeddings.shape[0] != 96:
                    raise ValueError(f"Expected 96-dim embeddings, got {embeddings.shape[0]}")
                return PATBinaryPredictions(0.5, 0.5, 0.5)
            
            def predict_depression(self, embeddings: np.ndarray) -> float:
                if embeddings.shape[0] != 96:
                    raise ValueError(f"Expected 96-dim embeddings, got {embeddings.shape[0]}")
                return 0.5
            
            def predict_medication_proxy(self, embeddings: np.ndarray) -> float:
                return 0.5

        predictor = StrictPATPredictor()

        # Should work with correct dimensions
        valid_embeddings = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(valid_embeddings)
        assert isinstance(prediction, PATBinaryPredictions)

        # Should fail with wrong dimensions
        invalid_embeddings = np.random.rand(50)
        with pytest.raises(ValueError, match="Expected 96-dim embeddings"):
            predictor.predict_from_embeddings(invalid_embeddings)

    def test_pat_predictor_handles_batch_predictions(self):
        """PAT predictor should handle batch predictions efficiently."""
        from big_mood_detector.domain.services.pat_predictor import (
            PATPredictorInterface,
            PATBinaryPredictions,
        )

        class BatchPATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> PATBinaryPredictions:
                # Handle single embedding
                if embeddings.ndim == 1:
                    return PATBinaryPredictions(0.4, 0.3, 0.7)
                # Batch predictions not required by interface but useful
                raise NotImplementedError("Batch predictions not implemented")
            
            def predict_depression(self, embeddings: np.ndarray) -> float:
                return 0.4
            
            def predict_medication_proxy(self, embeddings: np.ndarray) -> float:
                return 0.3

        predictor = BatchPATPredictor()
        single_embedding = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(single_embedding)
        assert isinstance(prediction, PATBinaryPredictions)

    def test_binary_predictions_are_independent(self):
        """Binary predictions don't need to sum to 1 (they're independent)."""
        from big_mood_detector.domain.services.pat_predictor import (
            PATPredictorInterface,
            PATBinaryPredictions,
        )

        class IndependentPATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> PATBinaryPredictions:
                # Independent binary predictions
                return PATBinaryPredictions(
                    depression_probability=0.8,  # High depression
                    benzodiazepine_probability=0.1,  # Low benzo use
                    confidence=0.85
                )
            
            def predict_depression(self, embeddings: np.ndarray) -> float:
                return 0.8
            
            def predict_medication_proxy(self, embeddings: np.ndarray) -> float:
                return 0.1

        predictor = IndependentPATPredictor()
        embeddings = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(embeddings)

        # Binary predictions are independent - they don't sum to 1
        assert prediction.depression_probability == 0.8
        assert prediction.benzodiazepine_probability == 0.1
        assert prediction.confidence == 0.85
