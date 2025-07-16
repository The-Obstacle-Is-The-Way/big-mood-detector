"""
Clinical Interpreter Service (Refactored)

Facade that orchestrates the extracted services for clinical interpretation.
This refactored version delegates to specialized services following SRP.

Design Patterns:
- Facade Pattern: Provides simplified interface to subsystems
- Dependency Injection: Services injected via constructor
- Delegation: Delegates to specialized services
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from big_mood_detector.domain.services.biomarker_interpreter import (
    BiomarkerInterpreter,
)
from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
    load_clinical_thresholds,
)
from big_mood_detector.domain.services.episode_interpreter import (
    EpisodeInterpreter,
)
from big_mood_detector.domain.services.treatment_recommender import (
    TreatmentRecommender,
)


# Re-export enums for backward compatibility
class RiskLevel(Enum):
    """Clinical risk stratification levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EpisodeType(Enum):
    """DSM-5 mood episode types."""
    NONE = "none"
    DEPRESSIVE = "depressive"
    MANIC = "manic"
    HYPOMANIC = "hypomanic"
    DEPRESSIVE_MIXED = "depressive_with_mixed_features"
    MANIC_MIXED = "manic_with_mixed_features"


@dataclass
class ClinicalRecommendation:
    """Clinical treatment recommendation."""
    medication: str
    evidence_level: str
    description: str
    contraindications: list[str] = field(default_factory=list)


@dataclass
class DSM5Criteria:
    """DSM-5 episode criteria evaluation."""
    meets_dsm5_criteria: bool
    clinical_note: str
    duration_met: bool = True
    symptom_count_met: bool = True
    functional_impairment: bool = True


@dataclass
class ClinicalInterpretation:
    """Complete clinical interpretation of mood state."""
    risk_level: RiskLevel
    episode_type: EpisodeType
    confidence: float
    clinical_summary: str
    recommendations: list[ClinicalRecommendation] = field(default_factory=list)
    dsm5_criteria_met: bool = False
    clinical_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class BiomarkerInterpretation:
    """Interpretation of digital biomarkers."""
    mania_risk_factors: int = 0
    depression_risk_factors: int = 0
    clinical_notes: list[str] = field(default_factory=list)
    recommendation_priority: str = "routine"
    mood_instability_risk: str = "low"
    clinical_summary: str = ""


class ClinicalInterpreter:
    """
    Refactored Clinical Interpreter - Facade for clinical interpretation services.
    
    This class now acts as a facade, delegating to specialized services:
    - EpisodeInterpreter: Interprets mood episodes
    - BiomarkerInterpreter: Interprets digital biomarkers
    - TreatmentRecommender: Provides treatment recommendations
    
    This follows the Single Responsibility Principle and makes the code
    more maintainable and testable.
    """
    
    def __init__(self, config: ClinicalThresholdsConfig | None = None):
        """
        Initialize the clinical interpreter with injected services.
        
        Args:
            config: Clinical thresholds configuration. If None, loads from default path.
        """
        if config is None:
            default_path = Path("config/clinical_thresholds.yaml")
            if default_path.exists():
                config = load_clinical_thresholds(default_path)
            else:
                raise ValueError("No configuration provided and default not found")
        
        self.config = config
        
        # Initialize specialized services
        self.episode_interpreter = EpisodeInterpreter(config)
        self.biomarker_interpreter = BiomarkerInterpreter(config)
        self.treatment_recommender = TreatmentRecommender(config)
    
    def interpret_depression_score(
        self,
        phq_score: float,
        sleep_hours: float,
        activity_steps: int,
        suicidal_ideation: bool = False,
    ) -> ClinicalInterpretation:
        """
        Interpret depression scores - delegates to EpisodeInterpreter.
        """
        result = self.episode_interpreter.interpret_depression(
            phq_score=phq_score,
            sleep_hours=sleep_hours,
            activity_steps=activity_steps,
            suicidal_ideation=suicidal_ideation,
        )
        
        # Convert to legacy format for backward compatibility
        risk_level = RiskLevel(result.risk_level)
        episode_type = EpisodeType(result.episode_type)
        
        # Get recommendations if needed
        recommendations = []
        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recs = self.treatment_recommender.get_recommendations(
                episode_type=result.episode_type,
                severity=result.risk_level,
                current_medications=[],
            )
            recommendations = [
                ClinicalRecommendation(
                    medication=r.medication,
                    evidence_level=r.evidence_level,
                    description=r.description,
                    contraindications=r.contraindications,
                )
                for r in recs
            ]
        
        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            recommendations=recommendations,
            dsm5_criteria_met=result.dsm5_criteria_met,
            clinical_features={
                "phq_score": phq_score,
                "sleep_hours": sleep_hours,
                "activity_steps": activity_steps,
            }
        )
    
    def interpret_mania_score(
        self,
        asrm_score: float,
        sleep_hours: float,
        activity_steps: int,
        psychotic_features: bool = False,
    ) -> ClinicalInterpretation:
        """
        Interpret mania scores - delegates to EpisodeInterpreter.
        """
        result = self.episode_interpreter.interpret_mania(
            asrm_score=asrm_score,
            sleep_hours=sleep_hours,
            activity_steps=activity_steps,
            psychotic_features=psychotic_features,
        )
        
        # Convert and get recommendations
        risk_level = RiskLevel(result.risk_level)
        episode_type = EpisodeType(result.episode_type)
        
        recommendations = []
        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recs = self.treatment_recommender.get_recommendations(
                episode_type=result.episode_type,
                severity=result.risk_level,
                current_medications=[],
            )
            recommendations = [
                ClinicalRecommendation(
                    medication=r.medication,
                    evidence_level=r.evidence_level,
                    description=r.description,
                    contraindications=r.contraindications,
                )
                for r in recs
            ]
        
        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            recommendations=recommendations,
            dsm5_criteria_met=result.dsm5_criteria_met,
            clinical_features={
                "asrm_score": asrm_score,
                "sleep_hours": sleep_hours,
                "activity_steps": activity_steps,
            }
        )
    
    def interpret_sleep_biomarkers(
        self,
        sleep_duration: float,
        sleep_efficiency: float,
        sleep_timing_variance: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret sleep biomarkers - delegates to BiomarkerInterpreter.
        """
        result = self.biomarker_interpreter.interpret_sleep(
            sleep_duration=sleep_duration,
            sleep_efficiency=sleep_efficiency,
            sleep_timing_variance=sleep_timing_variance,
        )
        
        # Convert to legacy format
        return BiomarkerInterpretation(
            mania_risk_factors=result.mania_risk_factors,
            depression_risk_factors=result.depression_risk_factors,
            clinical_notes=result.clinical_notes,
            recommendation_priority=result.recommendation_priority,
            mood_instability_risk=result.mood_instability_risk,
            clinical_summary=result.clinical_summary,
        )
    
    def interpret_activity_biomarkers(
        self,
        daily_steps: int,
        sedentary_hours: float,
        activity_variance: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret activity biomarkers - delegates to BiomarkerInterpreter.
        """
        result = self.biomarker_interpreter.interpret_activity(
            daily_steps=daily_steps,
            sedentary_hours=sedentary_hours,
        )
        
        return BiomarkerInterpretation(
            mania_risk_factors=result.mania_risk_factors,
            depression_risk_factors=result.depression_risk_factors,
            clinical_notes=result.clinical_notes,
            recommendation_priority=result.recommendation_priority,
            mood_instability_risk=result.mood_instability_risk,
            clinical_summary=result.clinical_summary,
        )
    
    def interpret_circadian_biomarkers(
        self,
        circadian_phase_advance: float,
        interdaily_stability: float,
        intradaily_variability: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret circadian biomarkers - delegates to BiomarkerInterpreter.
        """
        result = self.biomarker_interpreter.interpret_circadian(
            phase_advance=circadian_phase_advance,
            interdaily_stability=interdaily_stability,
            intradaily_variability=intradaily_variability,
        )
        
        return BiomarkerInterpretation(
            mania_risk_factors=result.mania_risk_factors,
            depression_risk_factors=result.depression_risk_factors,
            clinical_notes=result.clinical_notes,
            recommendation_priority=result.recommendation_priority,
            mood_instability_risk=result.mood_instability_risk,
            clinical_summary=result.clinical_summary,
        )
    
    def get_treatment_recommendations(
        self,
        episode_type: EpisodeType,
        severity: RiskLevel,
        current_medications: list[str],
        contraindications: list[str] | None = None,
        rapid_cycling: bool = False,
    ) -> list[ClinicalRecommendation]:
        """
        Get treatment recommendations - delegates to TreatmentRecommender.
        """
        recs = self.treatment_recommender.get_recommendations(
            episode_type=episode_type.value,
            severity=severity.value,
            current_medications=current_medications,
            contraindications=contraindications,
            rapid_cycling=rapid_cycling,
        )
        
        return [
            ClinicalRecommendation(
                medication=r.medication,
                evidence_level=r.evidence_level,
                description=r.description,
                contraindications=r.contraindications,
            )
            for r in recs
        ]
    
    def apply_clinical_rules(
        self,
        diagnosis: str,
        proposed_treatment: str,
        current_medications: list[str],
        mood_state: str,
    ) -> dict[str, Any]:
        """
        Apply clinical rules - delegates to TreatmentRecommender.
        """
        decision = self.treatment_recommender.apply_clinical_rules(
            diagnosis=diagnosis,
            proposed_treatment=proposed_treatment,
            current_medications=current_medications,
            mood_state=mood_state,
        )
        
        return {
            "approved": decision.approved,
            "rationale": decision.rationale,
        }
    
    def interpret_mixed_state(
        self,
        phq_score: float,
        asrm_score: float,
        sleep_hours: float,
        activity_steps: int,
        racing_thoughts: bool = False,
        increased_energy: bool = False,
        decreased_sleep: bool = False,
        depressed_mood: bool = False,
        anhedonia: bool = False,
        guilt: bool = False,
    ) -> ClinicalInterpretation:
        """
        Interpret mixed mood state - delegates to EpisodeInterpreter.
        
        Mixed episodes have symptoms of both depression and mania/hypomania.
        """
        result = self.episode_interpreter.interpret_mixed_state(
            phq_score=phq_score,
            asrm_score=asrm_score,
            sleep_hours=sleep_hours,
            activity_steps=activity_steps,
            racing_thoughts=racing_thoughts,
            increased_energy=increased_energy,
            decreased_sleep=decreased_sleep,
            depressed_mood=depressed_mood,
            anhedonia=anhedonia,
            guilt=guilt,
        )
        
        # Convert to legacy format for backward compatibility
        risk_level = RiskLevel(result.risk_level)
        episode_type = EpisodeType(result.episode_type)
        
        # Get recommendations
        recommendations = []
        if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recs = self.treatment_recommender.get_recommendations(
                episode_type=result.episode_type,
                severity=result.risk_level,
                current_medications=[],
            )
            recommendations = [
                ClinicalRecommendation(
                    medication=r.medication,
                    evidence_level=r.evidence_level,
                    description=r.description,
                    contraindications=r.contraindications,
                )
                for r in recs
            ]
        
        return ClinicalInterpretation(
            risk_level=risk_level,
            episode_type=episode_type,
            confidence=result.confidence,
            clinical_summary=result.clinical_summary,
            recommendations=recommendations,
            dsm5_criteria_met=result.dsm5_criteria_met,
            clinical_features={
                "phq_score": phq_score,
                "asrm_score": asrm_score,
                "sleep_hours": sleep_hours,
                "activity_steps": activity_steps,
                "racing_thoughts": racing_thoughts,
                "increased_energy": increased_energy,
            }
        )
    
    def evaluate_episode_duration(
        self,
        episode_type: EpisodeType,
        symptom_days: int,
        hospitalization: bool = False,
    ) -> DSM5Criteria:
        """
        Evaluate if episode duration meets DSM-5 criteria.
        
        DSM-5 Duration Requirements:
        - Manic: ≥7 days (or any duration if hospitalization)
        - Hypomanic: ≥4 days
        - Depressive: ≥14 days
        - Mixed: Follows primary episode requirements
        """
        # Get duration requirements from config
        duration_config = self.config.dsm5_duration
        
        # Check requirements based on episode type
        duration_met = False
        required_days = 0
        
        if episode_type == EpisodeType.MANIC:
            required_days = duration_config.manic_days
            duration_met = symptom_days >= required_days or hospitalization
        elif episode_type == EpisodeType.HYPOMANIC:
            required_days = duration_config.hypomanic_days
            duration_met = symptom_days >= required_days and not hospitalization
        elif episode_type == EpisodeType.DEPRESSIVE:
            required_days = duration_config.depressive_days
            duration_met = symptom_days >= required_days
        elif "mixed" in episode_type.value:
            # Mixed episodes follow primary pole duration
            if "depressive" in episode_type.value:
                required_days = duration_config.depressive_days
            else:
                required_days = duration_config.manic_days
            duration_met = symptom_days >= required_days or (
                "manic" in episode_type.value and hospitalization
            )
        
        # Create criteria result
        if duration_met:
            note = f"Episode duration of {symptom_days} days meets DSM-5 criteria"
        else:
            # Format message to match test expectations
            episode_name = episode_type.value.replace("_", " ")
            note = f"Duration insufficient for {episode_name} episode ({symptom_days} days < {required_days} days required)"
        
        if hospitalization and episode_type == EpisodeType.MANIC:
            note += " (hospitalization overrides duration requirement)"
        
        return DSM5Criteria(
            meets_dsm5_criteria=duration_met,
            clinical_note=note,
            duration_met=duration_met,
            symptom_count_met=True,  # Assume this was evaluated separately
            functional_impairment=True,  # Assume this was evaluated separately
        )