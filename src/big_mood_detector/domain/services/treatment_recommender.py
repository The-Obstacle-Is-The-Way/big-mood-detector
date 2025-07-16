"""
Treatment Recommender Service

Provides evidence-based treatment recommendations and applies clinical rules.

Design Patterns:
- Strategy Pattern: Different recommendation strategies per episode type
- Chain of Responsibility: Clinical rules applied in sequence
- Repository Pattern: Could easily swap recommendation source
"""

from dataclasses import dataclass
from typing import List, Optional

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)


@dataclass
class TreatmentRecommendation:
    """A specific treatment recommendation."""
    medication: str
    evidence_level: str  # first-line, second-line
    description: str
    contraindications: List[str] = None

    def __post_init__(self):
        if self.contraindications is None:
            self.contraindications = []


@dataclass
class ClinicalDecision:
    """Result of applying clinical rules."""
    approved: bool
    rationale: str


class TreatmentRecommender:
    """
    Provides treatment recommendations based on episode type and severity.
    
    Extracted from ClinicalInterpreter following Single Responsibility Principle.
    This service focuses solely on treatment recommendations and clinical rules.
    """
    
    def __init__(self, config: ClinicalThresholdsConfig):
        """
        Initialize with clinical configuration.
        
        Args:
            config: Clinical thresholds configuration
        """
        self.config = config
        
    def get_recommendations(
        self,
        episode_type: str,
        severity: str,
        current_medications: List[str],
        contraindications: Optional[List[str]] = None,
        rapid_cycling: bool = False,
    ) -> List[TreatmentRecommendation]:
        """
        Get treatment recommendations based on clinical presentation.
        
        Args:
            episode_type: Type of mood episode
            severity: Severity level (low, moderate, high, critical)
            current_medications: List of current medications
            contraindications: List of contraindications
            rapid_cycling: Whether patient has rapid cycling
            
        Returns:
            List of treatment recommendations
        """
        if contraindications is None:
            contraindications = []
            
        recommendations = []
        
        # Convert current medications to lowercase for comparison
        current_meds_lower = [med.lower() for med in current_medications]
        
        # Route to appropriate recommendation strategy
        if episode_type == "manic":
            recommendations = self._get_mania_recommendations(severity, current_meds_lower)
        elif episode_type == "depressive":
            recommendations = self._get_depression_recommendations(severity, current_meds_lower, rapid_cycling)
        elif "mixed" in episode_type:
            recommendations = self._get_mixed_state_recommendations(episode_type)
        elif episode_type == "hypomanic":
            recommendations = self._get_hypomania_recommendations(severity, current_meds_lower)
            
        # Filter out current medications
        recommendations = [
            rec for rec in recommendations 
            if rec.medication.lower() not in current_meds_lower
        ]
        
        # Apply contraindications
        if contraindications:
            recommendations = [
                rec for rec in recommendations
                if rec.medication.lower() not in [c.lower() for c in contraindications]
            ]
            
        return recommendations
    
    def _get_mania_recommendations(
        self, 
        severity: str, 
        current_medications: List[str]
    ) -> List[TreatmentRecommendation]:
        """Get recommendations for manic episodes."""
        recommendations = []
        
        if severity == "critical":
            # Urgent intervention needed
            recommendations.append(TreatmentRecommendation(
                medication="urgent hospitalization evaluation",
                evidence_level="first-line",
                description="Immediate safety assessment required",
            ))
        
        # First-line treatments for acute mania
        if "lithium" not in current_medications:
            recommendations.append(TreatmentRecommendation(
                medication="lithium",
                evidence_level="first-line",
                description="First-line mood stabilizer for acute mania",
                contraindications=["renal impairment", "pregnancy"],
            ))
            
        recommendations.append(TreatmentRecommendation(
            medication="quetiapine",
            evidence_level="first-line",
            description="Atypical antipsychotic for acute mania",
            contraindications=["diabetes", "metabolic syndrome"],
        ))
        
        recommendations.append(TreatmentRecommendation(
            medication="valproate",
            evidence_level="first-line",
            description="Mood stabilizer effective for acute mania",
            contraindications=["pregnancy", "liver disease"],
        ))
        
        return recommendations
    
    def _get_depression_recommendations(
        self, 
        severity: str, 
        current_medications: List[str],
        rapid_cycling: bool
    ) -> List[TreatmentRecommendation]:
        """Get recommendations for depressive episodes."""
        recommendations = []
        
        if severity == "critical":
            recommendations.append(TreatmentRecommendation(
                medication="urgent psychiatric evaluation",
                evidence_level="first-line",
                description="Urgent assessment required for severe depression",
            ))
        
        # First-line for bipolar depression
        recommendations.append(TreatmentRecommendation(
            medication="quetiapine",
            evidence_level="first-line",
            description="First-line treatment for bipolar depression",
        ))
        
        recommendations.append(TreatmentRecommendation(
            medication="lurasidone",
            evidence_level="first-line",
            description="FDA-approved for bipolar depression",
        ))
        
        # Lamotrigine - but not for rapid cycling
        if not rapid_cycling:
            recommendations.append(TreatmentRecommendation(
                medication="lamotrigine",
                evidence_level="first-line",
                description="Mood stabilizer for bipolar depression (not for rapid cycling)",
                contraindications=["rapid cycling", "Stevens-Johnson syndrome history"],
            ))
            
        return recommendations
    
    def _get_mixed_state_recommendations(
        self, 
        episode_type: str
    ) -> List[TreatmentRecommendation]:
        """Get recommendations for mixed episodes."""
        recommendations = []
        
        if episode_type == "depressive_with_mixed_features":
            # Second-line for depression with mixed features
            recommendations.extend([
                TreatmentRecommendation(
                    medication="cariprazine",
                    evidence_level="second-line",
                    description="Cariprazine is effective for depression with mixed features",
                ),
                TreatmentRecommendation(
                    medication="lurasidone",
                    evidence_level="second-line",
                    description="Lurasidone is an alternative for mixed depression",
                ),
            ])
        elif episode_type == "manic_with_mixed_features":
            # Second-line for mania with mixed features
            recommendations.extend([
                TreatmentRecommendation(
                    medication="asenapine",
                    evidence_level="second-line",
                    description="Effective for mania with mixed features",
                ),
                TreatmentRecommendation(
                    medication="cariprazine",
                    evidence_level="second-line",
                    description="Covers both manic and depressive symptoms",
                ),
            ])
            
        return recommendations
    
    def _get_hypomania_recommendations(
        self, 
        severity: str, 
        current_medications: List[str]
    ) -> List[TreatmentRecommendation]:
        """Get recommendations for hypomanic episodes."""
        # Similar to mania but less aggressive
        return self._get_mania_recommendations(severity, current_medications)
    
    def apply_clinical_rules(
        self,
        diagnosis: str,
        proposed_treatment: str,
        current_medications: List[str],
        mood_state: str,
    ) -> ClinicalDecision:
        """
        Apply evidence-based clinical decision rules.
        
        This demonstrates Chain of Responsibility pattern - each rule
        is checked in sequence.
        
        Args:
            diagnosis: Patient's diagnosis
            proposed_treatment: Proposed medication
            current_medications: Current medication list
            mood_state: Current mood state
            
        Returns:
            ClinicalDecision with approval status and rationale
        """
        # Rule 1: No antidepressant monotherapy in bipolar disorder
        if (diagnosis == "bipolar_disorder" and
            "depressed" in mood_state and
            proposed_treatment.lower() in ["sertraline", "fluoxetine", "escitalopram", "venlafaxine"] and
            not any(med.lower() in ["lithium", "valproate", "lamotrigine", "quetiapine"]
                   for med in current_medications)):
            return ClinicalDecision(
                approved=False,
                rationale="Antidepressant monotherapy is contraindicated in bipolar disorder",
            )
        
        # Rule 2: Antidepressant with mood stabilizer is acceptable
        if (diagnosis == "bipolar_disorder" and
            proposed_treatment.lower() in ["sertraline", "fluoxetine", "escitalopram"] and
            any(med.lower() in ["lithium", "valproate", "lamotrigine"]
                for med in current_medications)):
            return ClinicalDecision(
                approved=True,
                rationale="Antidepressant acceptable with mood stabilizer coverage",
            )
        
        # Rule 3: Lithium requires monitoring
        if proposed_treatment.lower() == "lithium":
            return ClinicalDecision(
                approved=True,
                rationale="Lithium approved - ensure regular monitoring of levels and renal function",
            )
        
        # Default approval
        return ClinicalDecision(
            approved=True,
            rationale="Treatment within clinical guidelines",
        )