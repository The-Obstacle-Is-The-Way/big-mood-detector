"""
Biomarker Interpreter Service

Interprets digital biomarkers (sleep, activity, circadian rhythms) for
clinical risk assessment.

Design Patterns:
- Factory Pattern: Creates specific interpretations based on biomarker type
- Builder Pattern: Builds comprehensive biomarker assessment
- Single Responsibility: Only handles biomarker interpretation
"""

from dataclasses import dataclass, field
from typing import List

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)


@dataclass
class BiomarkerInterpretation:
    """Result of biomarker interpretation."""
    mania_risk_factors: int = 0
    depression_risk_factors: int = 0
    clinical_notes: List[str] = field(default_factory=list)
    recommendation_priority: str = "routine"  # routine, urgent
    mood_instability_risk: str = "low"  # low, moderate, high
    clinical_summary: str = ""


class BiomarkerInterpreter:
    """
    Interprets digital biomarkers for clinical risk assessment.
    
    Extracted from ClinicalInterpreter following Single Responsibility Principle.
    Each interpretation method is focused and testable.
    """
    
    def __init__(self, config: ClinicalThresholdsConfig):
        """
        Initialize with clinical thresholds configuration.
        
        Args:
            config: Clinical thresholds for biomarker interpretation
        """
        self.config = config
    
    def interpret_sleep(
        self,
        sleep_duration: float,
        sleep_efficiency: float,
        sleep_timing_variance: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret sleep-related biomarkers.
        
        Args:
            sleep_duration: Hours of sleep
            sleep_efficiency: Percentage of time in bed actually sleeping (0-1)
            sleep_timing_variance: Variance in sleep timing (hours)
            
        Returns:
            BiomarkerInterpretation with risk assessment
        """
        result = BiomarkerInterpretation()
        
        # Critical short sleep (mania indicator)
        if sleep_duration < self.config.mania.sleep_hours.critical_threshold:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Critical short sleep duration indicates mania risk")
            result.recommendation_priority = "urgent"
        
        # Poor sleep efficiency
        if sleep_efficiency < self.config.biomarkers.sleep.efficiency_threshold:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Poor sleep efficiency")
        
        # Variable sleep timing
        if sleep_timing_variance > self.config.biomarkers.sleep.timing_variance_threshold:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Highly variable sleep schedule")
        
        # Generate summary if risks detected
        if result.mania_risk_factors > 0:
            result.clinical_summary = f"Sleep analysis reveals {result.mania_risk_factors} risk factors"
        
        return result
    
    def interpret_activity(
        self,
        daily_steps: int,
        sedentary_hours: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret activity-related biomarkers.
        
        Args:
            daily_steps: Number of steps per day
            sedentary_hours: Hours spent sedentary
            
        Returns:
            BiomarkerInterpretation with risk assessment
        """
        result = BiomarkerInterpretation()
        
        # Elevated activity (mania indicator)
        if daily_steps > self.config.mania.activity_steps.elevated_threshold:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Significantly elevated activity level")
        
        # Extreme activity
        if daily_steps > self.config.mania.activity_steps.extreme_threshold:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Extreme activity elevation")
        
        # Low sedentary time (mania indicator)
        if sedentary_hours < 4:  # Hard-coded for now, can add to config later
            result.mania_risk_factors += 1
            result.clinical_notes.append("Minimal rest periods")
        
        # Low activity (depression indicator)
        if daily_steps < self.config.depression.activity_steps.severe_reduction:
            result.depression_risk_factors += 1
            result.clinical_notes.append("Severe activity reduction")
        
        return result
    
    def interpret_circadian(
        self,
        phase_advance: float,
        interdaily_stability: float,
        intradaily_variability: float,
    ) -> BiomarkerInterpretation:
        """
        Interpret circadian rhythm biomarkers.
        
        Args:
            phase_advance: Hours of circadian phase advance
            interdaily_stability: Stability of daily rhythms (0-1)
            intradaily_variability: Fragmentation of daily rhythms
            
        Returns:
            BiomarkerInterpretation with risk assessment
        """
        result = BiomarkerInterpretation()
        
        # Phase advance (mania risk)
        if phase_advance > self.config.biomarkers.circadian.phase_advance_threshold:
            result.mania_risk_factors += 1
            result.clinical_notes.append("Significant circadian phase advance")
        
        # Low interdaily stability
        if interdaily_stability < self.config.biomarkers.circadian.interdaily_stability_low:
            result.mood_instability_risk = "high"
            result.clinical_notes.append("Low circadian rhythm stability")
        
        # High intradaily variability
        if intradaily_variability > self.config.biomarkers.circadian.intradaily_variability_high:
            result.mood_instability_risk = "high"
            result.clinical_notes.append("High circadian fragmentation")
        
        # Set summary for significant disruption
        if result.mood_instability_risk == "high" or result.mania_risk_factors > 0:
            result.clinical_summary = "Significant circadian disruption detected"
        
        return result
    
    def interpret_combined(
        self,
        sleep_result: BiomarkerInterpretation,
        activity_result: BiomarkerInterpretation,
        circadian_result: BiomarkerInterpretation,
    ) -> BiomarkerInterpretation:
        """
        Combine multiple biomarker interpretations.
        
        This method demonstrates the Builder pattern - building a comprehensive
        assessment from individual components.
        
        Args:
            sleep_result: Sleep biomarker interpretation
            activity_result: Activity biomarker interpretation
            circadian_result: Circadian biomarker interpretation
            
        Returns:
            Combined BiomarkerInterpretation
        """
        combined = BiomarkerInterpretation()
        
        # Aggregate risk factors
        combined.mania_risk_factors = (
            sleep_result.mania_risk_factors +
            activity_result.mania_risk_factors +
            circadian_result.mania_risk_factors
        )
        
        combined.depression_risk_factors = (
            sleep_result.depression_risk_factors +
            activity_result.depression_risk_factors +
            circadian_result.depression_risk_factors
        )
        
        # Combine clinical notes
        combined.clinical_notes.extend(sleep_result.clinical_notes)
        combined.clinical_notes.extend(activity_result.clinical_notes)
        combined.clinical_notes.extend(circadian_result.clinical_notes)
        
        # Set priority based on highest urgency
        if any(r.recommendation_priority == "urgent" for r in [sleep_result, activity_result, circadian_result]):
            combined.recommendation_priority = "urgent"
        
        # Set instability risk to highest level
        risk_levels = [sleep_result.mood_instability_risk, activity_result.mood_instability_risk, circadian_result.mood_instability_risk]
        if "high" in risk_levels:
            combined.mood_instability_risk = "high"
        elif "moderate" in risk_levels:
            combined.mood_instability_risk = "moderate"
        
        # Generate comprehensive summary
        if combined.mania_risk_factors > 0 or combined.depression_risk_factors > 0:
            combined.clinical_summary = (
                f"Digital biomarkers indicate {combined.mania_risk_factors} mania risk factors "
                f"and {combined.depression_risk_factors} depression risk factors"
            )
        
        return combined