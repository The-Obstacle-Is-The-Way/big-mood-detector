"""
Ensemble Model Orchestrator

Coordinates multiple ML models (PAT + XGBoost) for enhanced mood predictions.
Implements parallel processing, confidence weighting, and fallback strategies.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from big_mood_detector.domain.entities.activity_record import ActivityRecord
from big_mood_detector.domain.services.mood_predictor import MoodPrediction
from big_mood_detector.domain.services.pat_sequence_builder import PATSequenceBuilder
from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
from big_mood_detector.infrastructure.ml_models.xgboost_models import (
    XGBoostMoodPredictor,
)

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model orchestration."""
    
    # Model weights for ensemble averaging
    xgboost_weight: float = 0.6
    pat_weight: float = 0.4
    
    # Timeout settings (seconds)
    pat_timeout: float = 10.0
    xgboost_timeout: float = 5.0
    
    # Feature engineering
    use_pat_features: bool = True
    pat_feature_dim: int = 16  # How many PAT features to use
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.7
    fallback_to_single_model: bool = True


@dataclass
class EnsemblePrediction:
    """Enhanced prediction with model contributions."""
    
    # Individual model predictions
    xgboost_prediction: Optional[MoodPrediction]
    pat_enhanced_prediction: Optional[MoodPrediction]
    
    # Ensemble results
    ensemble_prediction: MoodPrediction
    
    # Metadata
    models_used: List[str]
    confidence_scores: Dict[str, float]
    processing_time_ms: Dict[str, float]


class EnsembleOrchestrator:
    """
    Orchestrates multiple ML models for robust mood prediction.
    
    Features:
    - Parallel model execution
    - Graceful degradation on model failures
    - Confidence-based weighting
    - Performance monitoring
    """
    
    def __init__(
        self,
        xgboost_predictor: XGBoostMoodPredictor,
        pat_model: Optional[PATModel] = None,
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize the orchestrator with models.
        
        Args:
            xgboost_predictor: Primary XGBoost predictor
            pat_model: Optional PAT model for enhanced features
            config: Ensemble configuration
        """
        self.xgboost_predictor = xgboost_predictor
        self.pat_model = pat_model
        self.pat_builder = PATSequenceBuilder() if pat_model else None
        self.config = config or EnsembleConfig()
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info(
            f"Initialized ensemble orchestrator with: "
            f"XGBoost={xgboost_predictor.is_loaded}, "
            f"PAT={pat_model is not None and pat_model.is_loaded}"
        )
    
    def predict(
        self,
        statistical_features: np.ndarray,
        activity_records: Optional[List[ActivityRecord]] = None,
        prediction_date: Optional[np.datetime64] = None
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction using all available models.
        
        Args:
            statistical_features: 36-feature vector from standard pipeline
            activity_records: Optional activity data for PAT features
            prediction_date: Date for prediction (for PAT sequence)
            
        Returns:
            EnsemblePrediction with detailed results
        """
        import time
        start_time = time.time()
        
        # Track timing
        timing = {}
        models_used = []
        predictions = {}
        
        # Submit parallel tasks
        futures = {}
        
        # 1. Standard XGBoost prediction
        futures['xgboost'] = self.executor.submit(
            self._predict_xgboost,
            statistical_features
        )
        
        # 2. PAT-enhanced prediction (if available)
        if self.config.use_pat_features and self.pat_model and activity_records:
            futures['pat_enhanced'] = self.executor.submit(
                self._predict_with_pat,
                statistical_features,
                activity_records,
                prediction_date
            )
        
        # Collect results with timeouts
        for name, future in futures.items():
            try:
                timeout = (
                    self.config.xgboost_timeout 
                    if name == 'xgboost' 
                    else self.config.pat_timeout
                )
                
                result_start = time.time()
                predictions[name] = future.result(timeout=timeout)
                timing[name] = (time.time() - result_start) * 1000  # ms
                models_used.append(name)
                
            except TimeoutError:
                logger.warning(f"{name} prediction timed out after {timeout}s")
                predictions[name] = None
                
            except Exception as e:
                logger.error(f"{name} prediction failed: {e}")
                predictions[name] = None
        
        # Calculate ensemble prediction
        ensemble_pred = self._calculate_ensemble(
            predictions.get('xgboost'),
            predictions.get('pat_enhanced')
        )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(predictions)
        
        # Total processing time
        timing['total'] = (time.time() - start_time) * 1000
        
        return EnsemblePrediction(
            xgboost_prediction=predictions.get('xgboost'),
            pat_enhanced_prediction=predictions.get('pat_enhanced'),
            ensemble_prediction=ensemble_pred,
            models_used=models_used,
            confidence_scores=confidence_scores,
            processing_time_ms=timing
        )
    
    def _predict_xgboost(self, features: np.ndarray) -> MoodPrediction:
        """Run standard XGBoost prediction."""
        return self.xgboost_predictor.predict(features)
    
    def _predict_with_pat(
        self,
        statistical_features: np.ndarray,
        activity_records: List[ActivityRecord],
        prediction_date: Optional[np.datetime64]
    ) -> MoodPrediction:
        """Run prediction with PAT-enhanced features."""
        # Build PAT sequence
        if prediction_date:
            sequence = self.pat_builder.build_sequence(
                activity_records,
                end_date=prediction_date
            )
        else:
            # Use last available date
            dates = [r.start_date.date() for r in activity_records]
            sequence = self.pat_builder.build_sequence(
                activity_records,
                end_date=max(dates) if dates else None
            )
        
        # Extract PAT features
        pat_features = self.pat_model.extract_features(sequence)
        
        # Combine with statistical features
        # Take first 20 statistical + 16 PAT features
        enhanced_features = np.concatenate([
            statistical_features[:20],
            pat_features[:self.config.pat_feature_dim]
        ])
        
        # Pad to 36 features if needed
        if len(enhanced_features) < 36:
            enhanced_features = np.pad(
                enhanced_features,
                (0, 36 - len(enhanced_features)),
                mode='constant'
            )
        
        return self.xgboost_predictor.predict(enhanced_features)
    
    def _calculate_ensemble(
        self,
        xgboost_pred: Optional[MoodPrediction],
        pat_pred: Optional[MoodPrediction]
    ) -> MoodPrediction:
        """Calculate weighted ensemble prediction."""
        # Handle missing predictions
        if xgboost_pred is None and pat_pred is None:
            # Return neutral prediction if both failed
            return MoodPrediction(
                depression_risk=0.5,
                hypomanic_risk=0.5,
                manic_risk=0.5,
                confidence=0.0
            )
        
        if xgboost_pred is None:
            return pat_pred
        
        if pat_pred is None:
            return xgboost_pred
        
        # Weighted average
        w1, w2 = self.config.xgboost_weight, self.config.pat_weight
        
        depression = w1 * xgboost_pred.depression_risk + w2 * pat_pred.depression_risk
        hypomanic = w1 * xgboost_pred.hypomanic_risk + w2 * pat_pred.hypomanic_risk
        manic = w1 * xgboost_pred.manic_risk + w2 * pat_pred.manic_risk
        
        # Average confidence
        confidence = (xgboost_pred.confidence + pat_pred.confidence) / 2
        
        # Determine highest risk
        risks = {
            "depression": depression,
            "hypomanic": hypomanic,
            "manic": manic
        }
        highest_risk_type = max(risks, key=risks.get)
        
        return MoodPrediction(
            depression_risk=depression,
            hypomanic_risk=hypomanic,
            manic_risk=manic,
            confidence=confidence
        )
    
    def _calculate_confidence(
        self,
        predictions: Dict[str, Optional[MoodPrediction]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for each model."""
        scores = {}
        
        for name, pred in predictions.items():
            if pred is not None:
                scores[name] = pred.confidence
            else:
                scores[name] = 0.0
        
        # Overall ensemble confidence
        valid_scores = [s for s in scores.values() if s > 0]
        scores['ensemble'] = np.mean(valid_scores) if valid_scores else 0.0
        
        return scores
    
    def shutdown(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)