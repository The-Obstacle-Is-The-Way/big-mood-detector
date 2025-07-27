"""
Process Health Data with Independent Pipelines Use Case

Uses the new independent PAT and XGBoost pipelines to process health data
according to each model's specific requirements and temporal windows.
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

from big_mood_detector.application.pipelines.pat_pipeline import PatPipeline
from big_mood_detector.application.pipelines.xgboost_pipeline import XGBoostPipeline
from big_mood_detector.application.services.data_parsing_service import (
    DataParsingService,
)
from big_mood_detector.application.validators.pipeline_validators import (
    PATValidator,
    XGBoostValidator,
)
from big_mood_detector.domain.services.clinical_feature_extractor import (
    ClinicalFeatureExtractor,
)
from big_mood_detector.infrastructure.ml_models.xgboost_models import (
    XGBoostMoodPredictor,
)

logger = logging.getLogger(__name__)


@dataclass
class IndependentPipelineResult:
    """Results from running independent pipelines."""

    pat_available: bool
    pat_result: Any  # Optional[PATResult]
    pat_message: str
    
    xgboost_available: bool
    xgboost_result: Any  # Optional[XGBoostResult]
    xgboost_message: str
    
    temporal_ensemble: dict[str, Any]  # Combined interpretation
    data_summary: dict[str, Any]


class ProcessWithIndependentPipelinesUseCase:
    """
    Process health data using independent PAT and XGBoost pipelines.
    
    This use case:
    1. Parses health data from XML/JSON
    2. Validates data for each pipeline independently
    3. Runs PAT for current depression assessment (7 consecutive days)
    4. Runs XGBoost for tomorrow's mood prediction (30-60 sparse days)
    5. Combines results into temporal ensemble interpretation
    """
    
    def __init__(
        self,
        data_parsing_service: Optional[DataParsingService] = None,
        pat_pipeline: Optional[PatPipeline] = None,
        xgboost_pipeline: Optional[XGBoostPipeline] = None,
    ):
        """Initialize with services."""
        self.data_parsing_service = data_parsing_service or DataParsingService()
        
        # Initialize pipelines if not provided
        self.pat_pipeline: Optional[PatPipeline] = pat_pipeline
        if not self.pat_pipeline:
            try:
                from big_mood_detector.infrastructure.ml_models.pat_production_loader import (
                    ProductionPATLoader,
                )
                pat_loader = ProductionPATLoader()
                self.pat_pipeline = PatPipeline(
                    pat_loader=pat_loader,
                    validator=PATValidator(),
                )
            except Exception as e:
                logger.warning(f"Could not initialize PAT pipeline: {e}")
                self.pat_pipeline = None
            
        self.xgboost_pipeline: Optional[XGBoostPipeline] = xgboost_pipeline
        if not self.xgboost_pipeline:
            try:
                predictor = XGBoostMoodPredictor()
                predictor.load_models(Path("model_weights/xgboost/converted"))
                
                feature_extractor = ClinicalFeatureExtractor()
                
                self.xgboost_pipeline = XGBoostPipeline(
                    feature_extractor=feature_extractor,
                    predictor=predictor,
                    validator=XGBoostValidator(),
                )
            except Exception as e:
                logger.warning(f"Could not initialize XGBoost pipeline: {e}")
                self.xgboost_pipeline = None
    
    def execute(
        self,
        file_path: Path,
        target_date: Optional[date] = None,
    ) -> IndependentPipelineResult:
        """
        Execute the use case.
        
        Args:
            file_path: Path to health data export
            target_date: Date to assess (defaults to today)
            
        Returns:
            IndependentPipelineResult with results from both pipelines
        """
        # Default to today if not specified
        if not target_date:
            target_date = date.today()
            
        # Parse health data
        logger.info(f"Parsing health data from {file_path}")
        parsed_data = self.data_parsing_service.parse_health_data(
            file_path=file_path,
            continue_on_error=True,
        )
        
        # Extract records
        sleep_records = parsed_data.get("sleep_records", [])
        activity_records = parsed_data.get("activity_records", [])
        heart_records = parsed_data.get("heart_rate_records", [])
        
        # Get data summary
        unique_sleep_days = len({r.start_date.date() for r in sleep_records})
        unique_activity_days = len({r.start_date.date() for r in activity_records})
        unique_heart_days = len({r.timestamp.date() for r in heart_records})
        
        data_summary = {
            "sleep_days": unique_sleep_days,
            "activity_days": unique_activity_days,
            "heart_days": unique_heart_days,
            "total_records": len(sleep_records) + len(activity_records) + len(heart_records),
        }
        
        # Initialize results
        pat_result = None
        pat_message = ""
        pat_available = False
        
        xgboost_result = None
        xgboost_message = ""
        xgboost_available = False
        
        # Try PAT pipeline
        if self.pat_pipeline:
            logger.info("Checking PAT pipeline availability")
            
            # Get date range for validation
            if activity_records:
                pat_dates = sorted({r.start_date.date() for r in activity_records})
                start_date = min(pat_dates)
                end_date = max(pat_dates)
                
                validation = self.pat_pipeline.can_run(
                    activity_records=activity_records,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                if validation.can_run:
                    logger.info("Running PAT pipeline")
                    pat_result = self.pat_pipeline.process(
                        activity_records=activity_records,
                        target_date=target_date,
                    )
                    if pat_result:
                        pat_available = True
                        pat_message = f"PAT assessed {pat_result.window_start_date} to {pat_result.window_end_date}"
                    else:
                        pat_message = "PAT processing failed"
                else:
                    pat_message = validation.message
                    logger.info(f"PAT not available: {pat_message}")
            else:
                pat_message = "No activity records found"
        else:
            pat_message = "PAT pipeline not initialized"
            
        # Try XGBoost pipeline
        if self.xgboost_pipeline:
            logger.info("Checking XGBoost pipeline availability")
            
            # Get date range for validation
            xgboost_dates: set[date] = set()
            for r in sleep_records:
                xgboost_dates.add(r.start_date.date())
            for r in activity_records:
                xgboost_dates.add(r.start_date.date())
            for r in heart_records:
                xgboost_dates.add(r.timestamp.date())
                
            if xgboost_dates:
                sorted_dates = sorted(xgboost_dates)
                start_date = min(sorted_dates)
                end_date = max(sorted_dates)
                
                validation = self.xgboost_pipeline.can_run(
                    sleep_records=sleep_records,
                    activity_records=activity_records,
                    heart_records=heart_records,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                if validation.can_run:
                    logger.info("Running XGBoost pipeline")
                    xgboost_result = self.xgboost_pipeline.process(
                        sleep_records=sleep_records,
                        activity_records=activity_records,
                        heart_records=heart_records,
                        target_date=target_date,
                    )
                    if xgboost_result:
                        xgboost_available = True
                        xgboost_message = f"XGBoost used {xgboost_result.data_days_used} days of data"
                    else:
                        xgboost_message = "XGBoost processing failed"
                else:
                    xgboost_message = validation.message
                    logger.info(f"XGBoost not available: {xgboost_message}")
            else:
                xgboost_message = "No health records found"
        else:
            xgboost_message = "XGBoost pipeline not initialized"
            
        # Create temporal ensemble interpretation
        temporal_ensemble = self._create_temporal_ensemble(
            pat_result=pat_result,
            xgboost_result=xgboost_result,
            target_date=target_date,
        )
        
        return IndependentPipelineResult(
            pat_available=pat_available,
            pat_result=pat_result,
            pat_message=pat_message,
            xgboost_available=xgboost_available,
            xgboost_result=xgboost_result,
            xgboost_message=xgboost_message,
            temporal_ensemble=temporal_ensemble,
            data_summary=data_summary,
        )
    
    def _create_temporal_ensemble(
        self,
        pat_result: Any,  # Optional[PATResult]
        xgboost_result: Any,  # Optional[XGBoostResult]
        target_date: date,
    ) -> dict[str, Any]:
        """
        Create temporal ensemble interpretation.
        
        PAT: Assesses current state (past 7 days)
        XGBoost: Predicts future risk (next 24 hours)
        """
        ensemble: dict[str, Any] = {
            "assessment_date": str(target_date),
            "temporal_windows": {},
            "clinical_summary": "",
            "recommendations": list[str](),
        }
        
        # Add PAT assessment (current state)
        if pat_result:
            ensemble["temporal_windows"]["current_state"] = {
                "model": "PAT",
                "window": f"{pat_result.window_start_date} to {pat_result.window_end_date}",
                "depression_risk": pat_result.depression_risk_score,
                "confidence": pat_result.confidence,
                "interpretation": pat_result.clinical_interpretation,
            }
        
        # Add XGBoost prediction (future risk)
        if xgboost_result:
            ensemble["temporal_windows"]["future_risk"] = {
                "model": "XGBoost",
                "window": xgboost_result.prediction_window,
                "depression_risk": xgboost_result.depression_probability,
                "mania_risk": xgboost_result.mania_probability,
                "hypomania_risk": xgboost_result.hypomania_probability,
                "highest_risk": xgboost_result.highest_risk_episode,
                "confidence": xgboost_result.confidence_level,
                "interpretation": xgboost_result.clinical_interpretation,
            }
        
        # Create clinical summary
        recommendations: list[str] = ensemble["recommendations"]
        
        if pat_result and xgboost_result:
            # Both models available - full temporal assessment
            current_depression = pat_result.depression_risk_score
            future_depression = xgboost_result.depression_probability
            future_mania = xgboost_result.mania_probability
            future_hypomania = xgboost_result.hypomania_probability
            
            # Determine overall state
            if current_depression > 0.5 and future_depression > 0.5:
                ensemble["clinical_summary"] = (
                    "Currently experiencing elevated depression symptoms with "
                    "continued risk in the next 24 hours. Immediate clinical "
                    "intervention recommended."
                )
                recommendations.extend([
                    "Contact mental health provider immediately",
                    "Maintain regular sleep schedule",
                    "Engage in light physical activity if possible",
                    "Avoid alcohol and stimulants",
                ])
            elif current_depression > 0.5 and future_depression <= 0.5:
                ensemble["clinical_summary"] = (
                    "Currently experiencing depression symptoms but risk may "
                    "decrease in next 24 hours. Continue monitoring closely."
                )
                recommendations.extend([
                    "Schedule follow-up with provider",
                    "Continue current treatment plan",
                    "Track mood changes closely",
                ])
            elif current_depression <= 0.5 and future_depression > 0.5:
                ensemble["clinical_summary"] = (
                    "Currently stable but increasing depression risk detected "
                    "for next 24 hours. Take preventive measures."
                )
                recommendations.extend([
                    "Prioritize sleep hygiene tonight",
                    "Plan stress-reducing activities",
                    "Consider reaching out to support network",
                ])
            elif max(future_mania, future_hypomania) > 0.5:
                ensemble["clinical_summary"] = (
                    f"Elevated risk for {xgboost_result.highest_risk_episode} "
                    "in next 24 hours. Monitor for early warning signs."
                )
                recommendations.extend([
                    "Monitor for decreased sleep need",
                    "Track activity levels and spending",
                    "Inform trusted contacts about risk",
                    "Have emergency plan ready",
                ])
            else:
                ensemble["clinical_summary"] = (
                    "Currently stable with low risk for mood episodes in "
                    "next 24 hours. Continue healthy routines."
                )
                recommendations.extend([
                    "Maintain consistent sleep schedule",
                    "Continue regular activities",
                    "Stay connected with support network",
                ])
        
        elif pat_result:
            # Only PAT available
            if pat_result.depression_risk_score > 0.5:
                ensemble["clinical_summary"] = (
                    "Current depression symptoms detected. Unable to predict "
                    "future risk without more historical data."
                )
            else:
                ensemble["clinical_summary"] = (
                    "Currently stable. Continue collecting data for future "
                    "risk prediction."
                )
            recommendations.append(
                "Collect at least 30 days of data for predictive capabilities"
            )
            
        elif xgboost_result:
            # Only XGBoost available
            if xgboost_result.highest_risk_episode != "stable":
                ensemble["clinical_summary"] = (
                    f"Elevated risk for {xgboost_result.highest_risk_episode} "
                    "predicted. Current state unknown without recent activity data."
                )
            else:
                ensemble["clinical_summary"] = (
                    "Low risk predicted for next 24 hours. Current state "
                    "unknown without recent activity data."
                )
            recommendations.append(
                "Ensure consistent activity tracking for current state assessment"
            )
        
        else:
            # Neither model available
            ensemble["clinical_summary"] = (
                "Insufficient data for mood assessment. Need at least 7 "
                "consecutive days of activity data and 30 days of health data."
            )
            recommendations.extend([
                "Wear activity tracker consistently",
                "Enable all health data permissions",
                "Check back after collecting more data",
            ])
        
        return ensemble