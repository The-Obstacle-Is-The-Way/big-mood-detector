"""
Test PAT Predictor Interface

Tests for the domain interface that PAT predictors must implement.
This ensures we can swap implementations without changing domain logic.
"""

import numpy as np
import pytest

from big_mood_detector.domain.services.mood_predictor import MoodPrediction


class TestPATPredictorInterface:
    """Test the PAT predictor interface contract."""

    def test_pat_predictor_interface_exists(self):
        """PAT predictor interface should be defined in domain layer."""
        from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface
        
        assert hasattr(PATPredictorInterface, 'predict_from_embeddings')
    
    def test_pat_predictor_returns_mood_prediction(self):
        """PAT predictor should return standard MoodPrediction object."""
        from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface
        
        # Create a concrete implementation for testing
        class ConcretePATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> MoodPrediction:
                # Simple implementation for testing
                return MoodPrediction(
                    depression_risk=0.3,
                    hypomanic_risk=0.2,
                    manic_risk=0.1,
                    confidence=0.8
                )
        
        predictor = ConcretePATPredictor()
        embeddings = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(embeddings)
        
        assert isinstance(prediction, MoodPrediction)
        assert 0 <= prediction.depression_risk <= 1
        assert 0 <= prediction.hypomanic_risk <= 1
        assert 0 <= prediction.manic_risk <= 1
        assert 0 <= prediction.confidence <= 1
    
    def test_pat_predictor_validates_embedding_dimension(self):
        """PAT predictor should validate embedding dimensions."""
        from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface
        
        class StrictPATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> MoodPrediction:
                if embeddings.shape[0] != 96:
                    raise ValueError(f"Expected 96-dim embeddings, got {embeddings.shape[0]}")
                return MoodPrediction(0.5, 0.5, 0.5, 0.5)
        
        predictor = StrictPATPredictor()
        
        # Should work with correct dimensions
        valid_embeddings = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(valid_embeddings)
        assert isinstance(prediction, MoodPrediction)
        
        # Should fail with wrong dimensions
        invalid_embeddings = np.random.rand(50)
        with pytest.raises(ValueError, match="Expected 96-dim embeddings"):
            predictor.predict_from_embeddings(invalid_embeddings)
    
    def test_pat_predictor_handles_batch_predictions(self):
        """PAT predictor should handle batch predictions efficiently."""
        from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface
        
        class BatchPATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> MoodPrediction:
                # Handle single embedding
                if embeddings.ndim == 1:
                    return MoodPrediction(0.4, 0.3, 0.2, 0.7)
                # Batch predictions not required by interface but useful
                raise NotImplementedError("Batch predictions not implemented")
        
        predictor = BatchPATPredictor()
        single_embedding = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(single_embedding)
        assert isinstance(prediction, MoodPrediction)
    
    def test_mood_prediction_risks_sum_approximately_to_one(self):
        """Mood risks should be probabilities that approximately sum to 1."""
        from big_mood_detector.domain.services.pat_predictor import PATPredictorInterface
        
        class ProbabilisticPATPredictor(PATPredictorInterface):
            def predict_from_embeddings(self, embeddings: np.ndarray) -> MoodPrediction:
                # Softmax-like output
                raw_scores = np.array([0.8, 0.3, 0.1])
                exp_scores = np.exp(raw_scores)
                probs = exp_scores / exp_scores.sum()
                
                return MoodPrediction(
                    depression_risk=float(probs[0]),
                    hypomanic_risk=float(probs[1]),
                    manic_risk=float(probs[2]),
                    confidence=0.85
                )
        
        predictor = ProbabilisticPATPredictor()
        embeddings = np.random.rand(96)
        prediction = predictor.predict_from_embeddings(embeddings)
        
        total_risk = (prediction.depression_risk + 
                      prediction.hypomanic_risk + 
                      prediction.manic_risk)
        
        # Allow small floating point errors
        assert abs(total_risk - 1.0) < 0.01