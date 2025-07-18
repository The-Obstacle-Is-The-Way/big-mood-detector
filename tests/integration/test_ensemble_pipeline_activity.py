"""
End-to-end integration test for ensemble pipeline with activity data.

Tests that activity data flows correctly through the entire pipeline
and that predictions are consistent whether called via API or directly.
"""

import pytest
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path

from big_mood_detector.application.use_cases.predict_mood_ensemble_use_case import (
    EnsembleConfig,
    EnsembleOrchestrator,
)
from big_mood_detector.application.use_cases.process_health_data_use_case import (
    MoodPredictionPipeline,
)
from big_mood_detector.domain.entities.activity_record import ActivityRecord, ActivityType
from big_mood_detector.domain.entities.sleep_record import SleepRecord
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
)
from big_mood_detector.infrastructure.ml_models.xgboost_models import XGBoostMoodPredictor
from big_mood_detector.infrastructure.ml_models import PAT_AVAILABLE


class TestEnsemblePipelineActivityFlow:
    """Test activity data flow through ensemble pipeline."""

    @pytest.fixture
    def sample_records(self):
        """Generate sample health records with activity data."""
        base_date = date.today() - timedelta(days=14)
        
        sleep_records = []
        activity_records = []
        
        # Generate 14 days of data
        for day in range(14):
            current_date = base_date + timedelta(days=day)
            
            # Sleep record
            sleep_records.append(
                SleepRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=23),
                    end_date=datetime.combine(current_date + timedelta(days=1), datetime.min.time()) + timedelta(hours=7),
                    is_main_sleep=True,
                    duration_hours=8.0,
                )
            )
            
            # Activity records throughout the day
            # Morning activity
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=7),
                    end_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=9),
                    activity_type=ActivityType.STEP_COUNT,
                    value=2000.0,
                    unit="count",
                )
            )
            
            # Midday activity
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=12),
                    end_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=13),
                    activity_type=ActivityType.STEP_COUNT,
                    value=1000.0,
                    unit="count",
                )
            )
            
            # Evening activity - varying to create patterns
            evening_steps = 5000.0 + (2000.0 * np.sin(day * np.pi / 7))
            activity_records.append(
                ActivityRecord(
                    source_name="Apple Watch",
                    start_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=17),
                    end_date=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=19),
                    activity_type=ActivityType.STEP_COUNT,
                    value=evening_steps,
                    unit="count",
                )
            )
        
        return {
            "sleep": sleep_records,
            "activity": activity_records,
            "heart_rate": []
        }

    @pytest.fixture
    def xgboost_predictor(self):
        """Create XGBoost predictor."""
        predictor = XGBoostMoodPredictor()
        # Try to load models, skip test if not available
        model_path = Path("model_weights/xgboost/pretrained")
        if not model_path.exists() or not predictor.load_models(model_path):
            pytest.skip("XGBoost models not available")
        return predictor

    @pytest.fixture
    def pat_model(self):
        """Create PAT model if available."""
        if not PAT_AVAILABLE:
            return None
            
        from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
        model = PATModel()
        if not model.load_pretrained_weights():
            return None
        return model

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_direct_ensemble_with_activity(self, sample_records, xgboost_predictor, pat_model):
        """Test ensemble prediction with activity data via direct call."""
        # Extract clinical features
        extractor = ClinicalFeatureExtractor()
        
        feature_set = extractor.extract_clinical_features(
            sleep_records=sample_records["sleep"],
            activity_records=sample_records["activity"],
            heart_records=sample_records["heart_rate"],
            target_date=date.today(),
        )
        
        # Verify activity features are extracted
        assert feature_set.total_steps > 0
        assert feature_set.activity_variance > 0
        
        # Create ensemble orchestrator
        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor,
            pat_model=pat_model,
            config=EnsembleConfig(),
        )
        
        # Get prediction with full features
        feature_vector = np.array(feature_set.to_xgboost_features(), dtype=np.float32)
        
        # Filter activity records for PAT
        target_date = date.today()
        relevant_activity = [
            r for r in sample_records["activity"]
            if (target_date - r.start_date.date()).days <= 7
        ]
        
        result = orchestrator.predict(
            statistical_features=feature_vector,
            activity_records=relevant_activity if pat_model else None,
            prediction_date=None,
        )
        
        # Verify prediction
        assert result.ensemble_prediction is not None
        assert 0 <= result.ensemble_prediction.depression_risk <= 1
        assert 0 <= result.ensemble_prediction.confidence <= 1
        
        # If PAT is available, both models should be used
        if pat_model:
            assert "pat_enhanced" in result.models_used
            assert len(result.models_used) == 2

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_pipeline_process_with_activity(self, sample_records, tmp_path):
        """Test full pipeline processing with activity data."""
        # Create pipeline
        pipeline = MoodPredictionPipeline()
        
        # Process data
        result = pipeline.process_health_data(
            sleep_records=sample_records["sleep"],
            activity_records=sample_records["activity"],
            heart_records=sample_records["heart_rate"],
            output_path=tmp_path / "results.json",
        )
        
        # Verify processing
        assert result.records_processed > 0
        assert len(result.daily_predictions) > 0
        
        # Check that activity was included in processing
        latest_prediction = result.daily_predictions[-1]
        assert latest_prediction["confidence"] > 0
        
        # If we have metadata about models used
        if "models_used" in result.metadata:
            assert "xgboost" in result.metadata["models_used"]

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_api_vs_direct_consistency(self, sample_records, xgboost_predictor, pat_model):
        """Test that API and direct calls produce consistent results."""
        from fastapi.testclient import TestClient
        from big_mood_detector.interfaces.api.main import app
        
        client = TestClient(app)
        
        # First, get prediction via direct call
        extractor = ClinicalFeatureExtractor()
        feature_set = extractor.extract_clinical_features(
            sleep_records=sample_records["sleep"],
            activity_records=sample_records["activity"],
            heart_records=sample_records["heart_rate"],
            target_date=date.today(),
        )
        
        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor,
            pat_model=pat_model,
            config=EnsembleConfig(),
        )
        
        direct_result = orchestrator.predict(
            statistical_features=np.array(feature_set.to_xgboost_features(), dtype=np.float32),
            activity_records=sample_records["activity"][-168:],  # Last 7 days
            prediction_date=None,
        )
        
        # Now via API (once activity features are exposed)
        # Create a mock feature input that includes activity
        features_dict = feature_set.__dict__.copy()
        features_dict.pop("date")  # Remove non-feature fields
        
        response = client.post(
            "/api/v1/predictions/predict/ensemble",
            json=features_dict,
        )
        
        assert response.status_code == 200
        api_result = response.json()
        
        # Results should be very close
        api_depression = api_result["ensemble_prediction"]["depression_risk"]
        direct_depression = direct_result.ensemble_prediction.depression_risk
        
        assert abs(api_depression - direct_depression) < 0.01

    @pytest.mark.skip(reason="awaiting implementation - activity features not exposed in API")
    def test_activity_improves_prediction_confidence(self, sample_records, xgboost_predictor):
        """Test that including activity data improves prediction confidence."""
        extractor = ClinicalFeatureExtractor()
        
        # Features without activity
        features_no_activity = extractor.extract_clinical_features(
            sleep_records=sample_records["sleep"],
            activity_records=[],  # No activity data
            heart_records=sample_records["heart_rate"],
            target_date=date.today(),
        )
        
        # Features with activity
        features_with_activity = extractor.extract_clinical_features(
            sleep_records=sample_records["sleep"],
            activity_records=sample_records["activity"],
            heart_records=sample_records["heart_rate"],
            target_date=date.today(),
        )
        
        # Verify activity features differ
        assert features_with_activity.total_steps > features_no_activity.total_steps
        assert features_with_activity.activity_variance > 0
        
        # Create predictions for both
        orchestrator = EnsembleOrchestrator(
            xgboost_predictor=xgboost_predictor,
            pat_model=None,  # Test without PAT to isolate activity effect
            config=EnsembleConfig(),
        )
        
        result_no_activity = orchestrator.predict(
            statistical_features=np.array(features_no_activity.to_xgboost_features(), dtype=np.float32),
            activity_records=None,
            prediction_date=None,
        )
        
        result_with_activity = orchestrator.predict(
            statistical_features=np.array(features_with_activity.to_xgboost_features(), dtype=np.float32),
            activity_records=None,
            prediction_date=None,
        )
        
        # Activity data should affect the prediction
        # (Can't guarantee higher confidence, but predictions should differ)
        assert (
            result_no_activity.ensemble_prediction.depression_risk != 
            result_with_activity.ensemble_prediction.depression_risk
        )