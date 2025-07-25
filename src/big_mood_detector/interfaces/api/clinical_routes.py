"""
Clinical API Routes

Exposes clinical interpretation endpoints for mood prediction results.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from big_mood_detector.domain.services.clinical_interpreter import (
    ClinicalInterpreter,
    EpisodeType,
)

router = APIRouter(prefix="/api/v1/clinical", tags=["clinical"])
interpreter = ClinicalInterpreter()


class DepressionAssessmentRequest(BaseModel):
    """Request model for depression assessment."""

    phq_score: float = Field(..., ge=0, le=27, description="PHQ-8/9 score (0-27)")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep hours")
    activity_steps: int = Field(..., ge=0, description="Daily step count")
    suicidal_ideation: bool = Field(False, description="Presence of suicidal thoughts")


class ManiaAssessmentRequest(BaseModel):
    """Request model for mania assessment."""

    asrm_score: float = Field(..., ge=0, description="ASRM score")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep hours")
    activity_steps: int = Field(..., ge=0, description="Daily step count")
    psychotic_features: bool = Field(
        False, description="Presence of psychotic features"
    )


class MixedStateAssessmentRequest(BaseModel):
    """Request model for mixed state assessment."""

    phq_score: float = Field(..., ge=0, le=27, description="PHQ-8/9 score")
    asrm_score: float = Field(..., ge=0, description="ASRM score")
    sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep hours")
    activity_steps: int = Field(..., ge=0, description="Daily step count")
    racing_thoughts: bool = Field(False, description="Racing thoughts present")
    increased_energy: bool = Field(False, description="Increased energy present")
    depressed_mood: bool = Field(False, description="Depressed mood present")
    anhedonia: bool = Field(False, description="Loss of interest/pleasure")
    guilt: bool = Field(False, description="Excessive guilt present")


class EpisodeDurationRequest(BaseModel):
    """Request model for episode duration evaluation."""

    episode_type: str = Field(
        ..., description="Episode type (manic, hypomanic, depressive)"
    )
    symptom_days: int = Field(..., ge=0, description="Number of days with symptoms")
    hospitalization: bool = Field(False, description="Whether hospitalization occurred")


class DigitalBiomarkersRequest(BaseModel):
    """Request model for digital biomarker interpretation."""

    sleep_duration: float = Field(
        ..., ge=0, le=24, description="Sleep duration in hours"
    )
    sleep_efficiency: float = Field(
        ..., ge=0, le=1, description="Sleep efficiency (0-1)"
    )
    sleep_timing_variance: float = Field(
        ..., ge=0, description="Variance in sleep timing (hours)"
    )
    daily_steps: int | None = Field(None, ge=0, description="Daily step count")
    sedentary_hours: float | None = Field(
        None, ge=0, le=24, description="Sedentary hours per day"
    )
    activity_variance: float | None = Field(
        None, ge=0, description="Activity level variance"
    )
    circadian_phase_advance: float | None = Field(
        None, description="Circadian phase advance in hours"
    )
    interdaily_stability: float | None = Field(
        None, ge=0, le=1, description="Interdaily stability (0-1)"
    )
    intradaily_variability: float | None = Field(
        None, ge=0, description="Intradaily variability"
    )


class ClinicalInterpretationResponse(BaseModel):
    """Response model for clinical interpretation."""

    risk_level: str = Field(
        ..., description="Risk level (low, moderate, high, critical)"
    )
    episode_type: str = Field(..., description="Episode type detected")
    confidence: float = Field(
        ..., ge=0, le=1, description="Prediction confidence (0-1)"
    )
    clinical_summary: str = Field(..., description="Human-readable clinical summary")
    dsm5_criteria_met: bool = Field(..., description="Whether DSM-5 criteria are met")
    recommendations: list[dict] = Field(..., description="Treatment recommendations")
    clinical_features: dict[str, Any]= Field(..., description="Key clinical features")


class DSM5EvaluationResponse(BaseModel):
    """Response model for DSM-5 evaluation."""

    meets_dsm5_criteria: bool = Field(..., description="Whether criteria are met")
    clinical_note: str = Field(..., description="Clinical explanation")
    duration_met: bool = Field(..., description="Whether duration requirement is met")


class BiomarkerInterpretationResponse(BaseModel):
    """Response model for biomarker interpretation."""

    mania_risk_factors: int = Field(..., description="Number of mania risk factors")
    depression_risk_factors: int = Field(
        ..., description="Number of depression risk factors"
    )
    mood_instability_risk: str = Field(..., description="Overall mood instability risk")
    clinical_notes: list[str] = Field(..., description="Clinical observations")
    recommendation_priority: str = Field(..., description="Urgency of intervention")


@router.post("/interpret/depression", response_model=ClinicalInterpretationResponse)
async def interpret_depression(
    request: DepressionAssessmentRequest,
) -> ClinicalInterpretationResponse:
    """
    Interpret depression assessment scores and biomarkers.

    Returns clinical risk level, episode type, and treatment recommendations
    based on PHQ-8/9 scores and digital biomarkers.
    """
    try:
        result = interpreter.interpret_depression_score(
            phq_score=request.phq_score,
            sleep_hours=request.sleep_hours,
            activity_steps=request.activity_steps,
            suicidal_ideation=request.suicidal_ideation,
        )

        return ClinicalInterpretationResponse(
            risk_level=result.risk_level.value,
            episode_type=result.episode_type.value,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            dsm5_criteria_met=result.dsm5_criteria_met,
            recommendations=[
                {
                    "medication": rec.medication,
                    "evidence_level": rec.evidence_level,
                    "description": rec.description,
                }
                for rec in result.recommendations
            ],
            clinical_features=result.clinical_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/interpret/mania", response_model=ClinicalInterpretationResponse)
async def interpret_mania(
    request: ManiaAssessmentRequest,
) -> ClinicalInterpretationResponse:
    """
    Interpret mania/hypomania assessment scores and biomarkers.

    Returns clinical risk level, episode type, and treatment recommendations
    based on ASRM scores and digital biomarkers.
    """
    try:
        result = interpreter.interpret_mania_score(
            asrm_score=request.asrm_score,
            sleep_hours=request.sleep_hours,
            activity_steps=request.activity_steps,
            psychotic_features=request.psychotic_features,
        )

        return ClinicalInterpretationResponse(
            risk_level=result.risk_level.value,
            episode_type=result.episode_type.value,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            dsm5_criteria_met=result.dsm5_criteria_met,
            recommendations=[
                {
                    "medication": rec.medication,
                    "evidence_level": rec.evidence_level,
                    "description": rec.description,
                }
                for rec in result.recommendations
            ],
            clinical_features=result.clinical_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/interpret/mixed", response_model=ClinicalInterpretationResponse)
async def interpret_mixed_state(
    request: MixedStateAssessmentRequest,
) -> ClinicalInterpretationResponse:
    """
    Interpret mixed state features based on DSM-5 criteria.

    Detects depression with mixed features or mania with mixed features
    when criteria from both poles are present.
    """
    try:
        result = interpreter.interpret_mixed_state(
            phq_score=request.phq_score,
            asrm_score=request.asrm_score,
            sleep_hours=request.sleep_hours,
            activity_steps=request.activity_steps,
            racing_thoughts=request.racing_thoughts,
            increased_energy=request.increased_energy,
            depressed_mood=request.depressed_mood,
            anhedonia=request.anhedonia,
            guilt=request.guilt,
        )

        return ClinicalInterpretationResponse(
            risk_level=result.risk_level.value,
            episode_type=result.episode_type.value,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            dsm5_criteria_met=result.dsm5_criteria_met,
            recommendations=[
                {
                    "medication": rec.medication,
                    "evidence_level": rec.evidence_level,
                    "description": rec.description,
                }
                for rec in result.recommendations
            ],
            clinical_features=result.clinical_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/evaluate/duration", response_model=DSM5EvaluationResponse)
async def evaluate_episode_duration(
    request: EpisodeDurationRequest,
) -> DSM5EvaluationResponse:
    """
    Evaluate if episode duration meets DSM-5 criteria.

    DSM-5 requires:
    - Manic: ≥7 days (or any duration with hospitalization)
    - Hypomanic: ≥4 days
    - Major Depressive: ≥14 days
    """
    try:
        # Map string to enum
        episode_map = {
            "manic": EpisodeType.MANIC,
            "hypomanic": EpisodeType.HYPOMANIC,
            "depressive": EpisodeType.DEPRESSIVE,
        }

        episode_type = episode_map.get(request.episode_type.lower())
        if not episode_type:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid episode type. Must be one of: {list(episode_map.keys())}",
            )

        result = interpreter.evaluate_episode_duration(
            episode_type=episode_type,
            symptom_days=request.symptom_days,
            hospitalization=request.hospitalization,
        )

        return DSM5EvaluationResponse(
            meets_dsm5_criteria=result.meets_dsm5_criteria,
            clinical_note=result.clinical_note,
            duration_met=result.duration_met,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/interpret/biomarkers", response_model=BiomarkerInterpretationResponse)
async def interpret_biomarkers(
    request: DigitalBiomarkersRequest,
) -> BiomarkerInterpretationResponse:
    """
    Interpret digital biomarkers for mood episode risk.

    Analyzes sleep, activity, and circadian rhythm metrics to identify
    risk factors for manic or depressive episodes.
    """
    try:
        # Interpret sleep biomarkers
        sleep_result = interpreter.interpret_sleep_biomarkers(
            sleep_duration=request.sleep_duration,
            sleep_efficiency=request.sleep_efficiency,
            sleep_timing_variance=request.sleep_timing_variance,
        )

        # Combine results
        mania_risk = sleep_result.mania_risk_factors
        depression_risk = sleep_result.depression_risk_factors
        clinical_notes = sleep_result.clinical_notes.copy()
        priority = sleep_result.recommendation_priority

        # Add activity biomarkers if provided
        if all(
            v is not None
            for v in [
                request.daily_steps,
                request.sedentary_hours,
                request.activity_variance,
            ]
        ):
            # Type narrowing - we know these are not None after the check
            activity_result = interpreter.interpret_activity_biomarkers(
                daily_steps=request.daily_steps,  # type: ignore[arg-type]
                sedentary_hours=request.sedentary_hours,  # type: ignore[arg-type]
                activity_variance=request.activity_variance,  # type: ignore[arg-type]
            )
            mania_risk += activity_result.mania_risk_factors
            depression_risk += activity_result.depression_risk_factors
            clinical_notes.extend(activity_result.clinical_notes)
            if activity_result.recommendation_priority == "urgent":
                priority = "urgent"

        # Add circadian biomarkers if provided
        if all(
            v is not None
            for v in [
                request.circadian_phase_advance,
                request.interdaily_stability,
                request.intradaily_variability,
            ]
        ):
            circadian_result = interpreter.interpret_circadian_biomarkers(
                circadian_phase_advance=request.circadian_phase_advance,  # type: ignore[arg-type]
                interdaily_stability=request.interdaily_stability,  # type: ignore[arg-type]
                intradaily_variability=request.intradaily_variability,  # type: ignore[arg-type]
            )
            mania_risk += circadian_result.mania_risk_factors
            depression_risk += circadian_result.depression_risk_factors
            clinical_notes.extend(circadian_result.clinical_notes)
            mood_instability = circadian_result.mood_instability_risk
        else:
            mood_instability = "unknown"

        return BiomarkerInterpretationResponse(
            mania_risk_factors=mania_risk,
            depression_risk_factors=depression_risk,
            mood_instability_risk=mood_instability,
            clinical_notes=clinical_notes,
            recommendation_priority=priority,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/thresholds")
async def get_clinical_thresholds() -> dict[str, Any]:
    """
    Get clinical thresholds used for interpretation.

    Returns evidence-based thresholds from the Clinical Dossier including
    PHQ-8/9 cutoffs, ASRM cutoffs, sleep duration thresholds, and more.
    """
    return {
        "depression": {
            "phq8_cutoff": 10,
            "phq9_mild": {"min": 5, "max": 9},
            "phq9_moderate": {"min": 10, "max": 14},
            "phq9_moderately_severe": {"min": 15, "max": 19},
            "phq9_severe": {"min": 20, "max": 27},
        },
        "mania": {
            "asrm_cutoff": 6,
            "asrm_moderate": {"min": 6, "max": 10},
            "asrm_high": {"min": 11, "max": 15},
            "asrm_critical": {"min": 16},
        },
        "sleep": {
            "normal_range": {"min": 7, "max": 9},
            "short_sleep_risk": {"max": 6},
            "long_sleep_risk": {"min": 9},
            "critical_short": {"max": 3},
            "critical_long": {"min": 12},
            "window_merging": 3.75,
        },
        "activity": {
            "mean_daily_steps": 6631,
            "low_activity_depression": {"max": 5000},
            "high_activity_mania": {"min": 15000},
            "extreme_activity": {"min": 20000},
        },
        "dsm5_duration": {
            "manic": 7,
            "hypomanic": 4,
            "depressive": 14,
        },
        "data_quality": {
            "minimum_days": 30,
            "completeness_threshold": 0.75,
        },
    }
