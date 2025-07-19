"""
Domain Services

Re-export key services and dataclasses for backward compatibility.
"""

# Clinical Interpretation Services
# Other Core Services
from .activity_aggregator import ActivityAggregator
from .activity_feature_calculator import ActivityFeatureCalculator
from .activity_sequence_extractor import ActivitySequenceExtractor
from .advanced_feature_engineering import AdvancedFeatureEngineer
from .biomarker_interpreter import BiomarkerInterpreter
from .circadian_feature_calculator import CircadianFeatureCalculator
from .circadian_rhythm_analyzer import CircadianRhythmAnalyzer

# New Services (Phase 2)
from .clinical_assessment_service import (
    ClinicalAssessment,
    ClinicalAssessmentService,
)
from .clinical_feature_extractor import ClinicalFeatureExtractor
from .clinical_interpreter import (
    ClinicalInterpretation,
    ClinicalInterpreter,
    ClinicalRecommendation,
    EpisodeType,
    RiskLevel,
)

# Clinical Configuration
from .clinical_thresholds import (
    ClinicalThresholdsConfig,
    load_clinical_thresholds,
)
from .dlmo_calculator import DLMOCalculator
from .dsm5_criteria_evaluator import DSM5CriteriaEvaluator
from .early_warning_detector import EarlyWarningDetector
from .episode_interpreter import EpisodeInterpreter
from .episode_labeler import EpisodeLabeler
from .feature_engineering_orchestrator import FeatureEngineeringOrchestrator
from .feature_extraction_service import FeatureExtractionService
from .heart_rate_aggregator import HeartRateAggregator
from .interpolation_strategies import InterpolationStrategy
from .intervention_evaluation_service import (
    InterventionDecision,
    InterventionEvaluationService,
)
from .longitudinal_assessment_service import (
    LongitudinalAssessment,
    LongitudinalAssessmentService,
)
from .mood_predictor import MoodPredictor
from .pat_sequence_builder import PATSequenceBuilder
from .risk_level_assessor import RiskLevelAssessor
from .sleep_aggregator import SleepAggregator
from .sleep_feature_calculator import SleepFeatureCalculator
from .sleep_window_analyzer import SleepWindowAnalyzer
from .sparse_data_handler import SparseDataHandler
from .temporal_feature_calculator import TemporalFeatureCalculator
from .treatment_recommender import TreatmentRecommender

__all__ = [
    # Clinical Interpretation
    "ClinicalInterpreter",
    "ClinicalInterpretation",
    "ClinicalRecommendation",
    "EpisodeType",
    "RiskLevel",
    # New Services
    "ClinicalAssessment",
    "ClinicalAssessmentService",
    "InterventionDecision",
    "InterventionEvaluationService",
    "LongitudinalAssessment",
    "LongitudinalAssessmentService",
    # Configuration
    "ClinicalThresholdsConfig",
    "load_clinical_thresholds",
    # Other Services
    "ActivityAggregator",
    "ActivityFeatureCalculator",
    "ActivitySequenceExtractor",
    "AdvancedFeatureEngineer",
    "BiomarkerInterpreter",
    "CircadianFeatureCalculator",
    "CircadianRhythmAnalyzer",
    "ClinicalFeatureExtractor",
    "DLMOCalculator",
    "DSM5CriteriaEvaluator",
    "EarlyWarningDetector",
    "EpisodeInterpreter",
    "EpisodeLabeler",
    "FeatureEngineeringOrchestrator",
    "FeatureExtractionService",
    "HeartRateAggregator",
    "InterpolationStrategy",
    "MoodPredictor",
    "PATSequenceBuilder",
    "RiskLevelAssessor",
    "SleepAggregator",
    "SleepFeatureCalculator",
    "SleepWindowAnalyzer",
    "SparseDataHandler",
    "TemporalFeatureCalculator",
    "TreatmentRecommender",
]
