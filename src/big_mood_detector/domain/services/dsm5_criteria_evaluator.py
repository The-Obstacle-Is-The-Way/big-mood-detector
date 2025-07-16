"""
DSM-5 Criteria Evaluator Service

Evaluates mood episodes against DSM-5 diagnostic criteria.
Extracted from ClinicalInterpreter to follow Single Responsibility Principle.

Design Patterns:
- Single Responsibility: Only evaluates DSM-5 criteria
- Value Objects: Immutable evaluation results
- Dependency Injection: Configuration injected
"""

from dataclasses import dataclass

from big_mood_detector.domain.services.clinical_thresholds import (
    ClinicalThresholdsConfig,
)


@dataclass(frozen=True)
class DurationCriteria:
    """DSM-5 duration criteria evaluation result."""

    duration_met: bool
    meets_criteria: bool
    required_days: int
    actual_days: int
    clinical_note: str


@dataclass(frozen=True)
class SymptomCriteria:
    """DSM-5 symptom count criteria evaluation result."""

    symptom_count_met: bool
    symptom_count: int
    required_count: int
    clinical_note: str


@dataclass(frozen=True)
class FunctionalCriteria:
    """DSM-5 functional impairment criteria evaluation result."""

    functional_impairment_met: bool
    impairment_domains: list[str]
    clinical_note: str


@dataclass(frozen=True)
class CompleteDSM5Evaluation:
    """Complete DSM-5 criteria evaluation result."""

    meets_all_criteria: bool
    duration_criteria: DurationCriteria
    symptom_criteria: SymptomCriteria
    functional_criteria: FunctionalCriteria
    episode_type: str
    summary: str


class DSM5CriteriaEvaluator:
    """
    Evaluates mood episodes against DSM-5 diagnostic criteria.

    This service focuses solely on DSM-5 criteria evaluation,
    extracted from the monolithic ClinicalInterpreter.
    """

    # DSM-5 minimum symptom counts
    DEPRESSION_MIN_SYMPTOMS = 5
    MANIA_MIN_SYMPTOMS = 3  # Plus elevated/irritable mood
    HYPOMANIA_MIN_SYMPTOMS = 3

    def __init__(self, config: ClinicalThresholdsConfig):
        """
        Initialize with clinical configuration.

        Args:
            config: Clinical thresholds configuration
        """
        self.config = config

    def evaluate_episode_duration(
        self,
        episode_type: str,
        symptom_days: int,
        hospitalization: bool = False,
    ) -> DurationCriteria:
        """
        Evaluate if episode duration meets DSM-5 criteria.

        Args:
            episode_type: Type of mood episode
            symptom_days: Number of days with symptoms
            hospitalization: Whether hospitalization occurred

        Returns:
            Duration criteria evaluation result
        """
        duration_config = self.config.dsm5_duration

        # Determine required duration based on episode type
        required_days = 0
        duration_met = False

        if episode_type == "manic":
            required_days = duration_config.manic_days
            # Hospitalization overrides duration requirement for mania
            duration_met = symptom_days >= required_days or hospitalization
        elif episode_type == "hypomanic":
            required_days = duration_config.hypomanic_days
            # Hospitalization invalidates hypomania (becomes mania)
            if hospitalization:
                duration_met = False
            else:
                duration_met = symptom_days >= required_days
        elif episode_type == "depressive":
            required_days = duration_config.depressive_days
            duration_met = symptom_days >= required_days
        elif "mixed" in episode_type:
            # Mixed episodes follow primary pole duration
            if "depressive" in episode_type:
                required_days = duration_config.depressive_days
            else:
                required_days = duration_config.manic_days
            duration_met = symptom_days >= required_days or (
                "manic" in episode_type and hospitalization
            )

        # Generate clinical note
        if duration_met:
            note = f"Episode duration of {symptom_days} days meets DSM-5 criteria"
            if hospitalization and episode_type == "manic":
                note += " (hospitalization overrides duration requirement)"
        else:
            if hospitalization and episode_type == "hypomanic":
                note = "Hospitalization invalidates hypomanic episode (would be manic)"
            else:
                note = f"Duration insufficient for {episode_type} episode ({symptom_days} days < {required_days} days required)"

        return DurationCriteria(
            duration_met=duration_met,
            meets_criteria=duration_met,
            required_days=required_days,
            actual_days=symptom_days,
            clinical_note=note,
        )

    def evaluate_symptom_count(
        self,
        symptoms: list[str],
        episode_type: str,
    ) -> SymptomCriteria:
        """
        Evaluate if symptom count meets DSM-5 criteria.

        Args:
            symptoms: List of present symptoms
            episode_type: Type of mood episode

        Returns:
            Symptom count criteria evaluation result
        """
        symptom_count = len(symptoms)

        # Determine required symptom count
        if "depressive" in episode_type:
            required_count = self.DEPRESSION_MIN_SYMPTOMS
        elif episode_type in ["manic", "hypomanic"]:
            required_count = self.MANIA_MIN_SYMPTOMS
        else:
            # Mixed episodes use primary pole requirements
            if "depressive" in episode_type:
                required_count = self.DEPRESSION_MIN_SYMPTOMS
            else:
                required_count = self.MANIA_MIN_SYMPTOMS

        symptom_count_met = symptom_count >= required_count

        # Generate clinical note
        if symptom_count_met:
            note = f"{symptom_count} symptoms meets minimum requirement of {required_count}"
        else:
            note = f"Insufficient symptoms: {symptom_count} < {required_count} required"

        return SymptomCriteria(
            symptom_count_met=symptom_count_met,
            symptom_count=symptom_count,
            required_count=required_count,
            clinical_note=note,
        )

    def evaluate_functional_impairment(
        self,
        work_impairment: bool = False,
        social_impairment: bool = False,
        self_care_impairment: bool = False,
        hospitalization: bool = False,
    ) -> FunctionalCriteria:
        """
        Evaluate functional impairment criteria.

        Args:
            work_impairment: Impairment in occupational functioning
            social_impairment: Impairment in social functioning
            self_care_impairment: Impairment in self-care
            hospitalization: Whether hospitalization occurred

        Returns:
            Functional impairment evaluation result
        """
        impairment_domains = []

        if work_impairment:
            impairment_domains.append("work/occupational")
        if social_impairment:
            impairment_domains.append("social/interpersonal")
        if self_care_impairment:
            impairment_domains.append("self-care")
        if hospitalization:
            impairment_domains.append("hospitalization required")

        # DSM-5 requires significant impairment in at least one domain
        functional_impairment_met = len(impairment_domains) > 0

        # Generate clinical note
        if functional_impairment_met:
            note = f"Significant impairment in: {', '.join(impairment_domains)}"
        else:
            note = "No significant functional impairment documented"

        return FunctionalCriteria(
            functional_impairment_met=functional_impairment_met,
            impairment_domains=impairment_domains,
            clinical_note=note,
        )

    def evaluate_complete_criteria(
        self,
        episode_type: str,
        symptom_days: int,
        symptoms: list[str],
        hospitalization: bool = False,
        functional_impairment: bool = False,
        work_impairment: bool = False,
        social_impairment: bool = False,
        self_care_impairment: bool = False,
    ) -> CompleteDSM5Evaluation:
        """
        Evaluate all DSM-5 criteria for a mood episode.

        Args:
            episode_type: Type of mood episode
            symptom_days: Number of days with symptoms
            symptoms: List of present symptoms
            hospitalization: Whether hospitalization occurred
            functional_impairment: General functional impairment flag
            work_impairment: Work-specific impairment
            social_impairment: Social functioning impairment
            self_care_impairment: Self-care impairment

        Returns:
            Complete DSM-5 evaluation result
        """
        # Evaluate each criterion
        duration = self.evaluate_episode_duration(
            episode_type, symptom_days, hospitalization
        )

        symptom_count = self.evaluate_symptom_count(symptoms, episode_type)

        functional = self.evaluate_functional_impairment(
            work_impairment=work_impairment or functional_impairment,
            social_impairment=social_impairment,
            self_care_impairment=self_care_impairment,
            hospitalization=hospitalization,
        )

        # All criteria must be met
        meets_all = (
            duration.meets_criteria
            and symptom_count.symptom_count_met
            and functional.functional_impairment_met
        )

        # Generate summary
        if meets_all:
            summary = f"Meets DSM-5 criteria for {episode_type} episode"
        else:
            summary = f"Does not meet DSM-5 criteria for {episode_type} episode"

        return CompleteDSM5Evaluation(
            meets_all_criteria=meets_all,
            duration_criteria=duration,
            symptom_criteria=symptom_count,
            functional_criteria=functional,
            episode_type=episode_type,
            summary=summary,
        )

    def generate_clinical_summary(self, evaluation: CompleteDSM5Evaluation) -> str:
        """
        Generate a clinical summary of DSM-5 evaluation.

        Args:
            evaluation: Complete DSM-5 evaluation result

        Returns:
            Human-readable clinical summary
        """
        parts = []

        # Episode type
        parts.append(f"DSM-5 Evaluation for {evaluation.episode_type.title()} Episode:")

        # Duration
        if evaluation.duration_criteria.duration_met:
            parts.append(
                f"- Duration: Sufficient ({evaluation.duration_criteria.actual_days} days)"
            )
        else:
            parts.append(
                f"- Duration: Insufficient ({evaluation.duration_criteria.actual_days}/{evaluation.duration_criteria.required_days} days)"
            )

        # Symptoms
        parts.append(
            f"- Symptoms: {evaluation.symptom_criteria.symptom_count}/{evaluation.symptom_criteria.required_count}"
        )

        # Functional impairment
        if evaluation.functional_criteria.functional_impairment_met:
            parts.append(
                f"- Functional impairment: Present ({', '.join(evaluation.functional_criteria.impairment_domains)})"
            )
        else:
            parts.append("- Functional impairment: Not documented")

        # Overall conclusion
        if evaluation.meets_all_criteria:
            parts.append(
                f"\nConclusion: Meets DSM-5 criteria for {evaluation.episode_type} episode"
            )
        else:
            parts.append("\nConclusion: Does not meet full DSM-5 criteria")

        return "\n".join(parts)
