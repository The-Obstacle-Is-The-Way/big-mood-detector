# Feature Bug Fix Plan - The Real Issue
Generated: 2025-07-23

## The Actual Problem

After deep investigation, the bug is more subtle than initially thought:

1. **AggregationPipeline** has the code to calculate Seoul statistical features (lines 810-984)
2. But it's returning `ClinicalFeatureSet` with `SeoulXGBoostFeatures`
3. `SeoulXGBoostFeatures` has clinical names (sleep_duration_hours, etc.)
4. But XGBoost expects `DailyFeatures` with Seoul names (sleep_percentage_MN, etc.)

## The Issue

The `_calculate_features_with_stats` method:
- DOES calculate the statistics (mean, std, zscore)
- Stores them in variables like `sleep_features["sleep_percentage_mean"]`
- But then creates a `SeoulXGBoostFeatures` object that ignores these stats!
- Returns a `ClinicalFeatureSet` instead of `DailyFeatures`

## The Fix

We need to modify `_calculate_features_with_stats` to:
1. Create a `DailyFeatures` object instead of `ClinicalFeatureSet`
2. Map the calculated statistics to the proper fields

OR (simpler):

Add a new method that specifically generates `DailyFeatures` for XGBoost.

## Code Location

File: `/src/big_mood_detector/application/services/aggregation_pipeline.py`
- Line 810: `_calculate_features_with_stats` - calculates stats but returns wrong type
- Lines 826-844: Correctly calculates Seoul statistics
- Lines 872-984: Creates wrong object type

## The Two-Line Patch

Actually, we need more than 2 lines, but the concept is simple:
1. Create a method that returns `DailyFeatures` 
2. Use that for XGBoost predictions instead of the clinical features