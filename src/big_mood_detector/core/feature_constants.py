"""
Feature constants and mappings for the XGBoost models.

Based on the Seoul National University study that achieved AUC 0.80-0.98.
"""

# Complete list of 36 features in the exact order expected by XGBoost models
XGBOOST_FEATURE_NAMES = [
    # Basic Sleep Features (0-4)
    "sleep_duration_hours",
    "sleep_efficiency",
    "sleep_onset_hour",
    "wake_time_hour",
    "sleep_fragmentation",
    # Advanced Sleep Features (5-9)
    "sleep_regularity_index",
    "short_sleep_window_pct",
    "long_sleep_window_pct",
    "sleep_onset_variance",
    "wake_time_variance",
    # Circadian Rhythm Features (10-17)
    "interdaily_stability",
    "intradaily_variability",
    "relative_amplitude",
    "l5_value",
    "m10_value",
    "l5_onset_hour",
    "m10_onset_hour",
    "dlmo_hour",
    # Activity Features (18-23)
    "total_steps",
    "activity_variance",
    "sedentary_hours",
    "activity_fragmentation",
    "sedentary_bout_mean",
    "activity_intensity_ratio",
    # Heart Rate Features (24-27)
    "avg_resting_hr",
    "hrv_sdnn",
    "hr_circadian_range",
    "hr_minimum_hour",
    # Phase Features (28-31)
    "circadian_phase_advance",
    "circadian_phase_delay",
    "dlmo_confidence",
    "pat_hour",
    # Z-Score Features (32-35)
    "sleep_duration_zscore",
    "activity_zscore",
    "circadian_phase_zscore",
    "sleep_efficiency_zscore",
]

# Mapping from API feature names to XGBoost feature indices
API_TO_XGBOOST_MAPPING = {
    "sleep_duration": 0,  # Maps to sleep_duration_hours
    "sleep_efficiency": 1,
    "sleep_timing_variance": 9,  # Maps to wake_time_variance
    "daily_steps": 18,  # Maps to total_steps
    "activity_variance": 19,
    "sedentary_hours": 20,
    "interdaily_stability": 10,
    "intradaily_variability": 11,
    "relative_amplitude": 12,
    "resting_hr": 24,  # Maps to avg_resting_hr
    "hrv_rmssd": 25,  # Maps to hrv_sdnn
}

# Clinical thresholds based on the literature
CLINICAL_THRESHOLDS = {
    "depression_risk": 0.5,
    "hypomanic_risk": 0.5,
    "manic_risk": 0.3,  # Lower threshold for mania due to severity
}

# Feature importance from the paper (top features)
TOP_FEATURES = [
    "circadian_phase_zscore",  # Most important per paper
    "interdaily_stability",
    "wake_time_variance",
    "sleep_duration_hours",
    "relative_amplitude",
]
