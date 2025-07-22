#!/usr/bin/env python3
"""
Assert that DailyFeatures.to_dict() matches the Seoul paper schema exactly.

This ensures we maintain the 36-feature contract required by the XGBoost models.
"""

import re
import sys
from pathlib import Path

import yaml


def main():
    """Validate feature schema against Seoul paper specification."""
    # Load expected features from YAML
    schema_path = Path("config/seoul_feature_schema.yml")
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)
    
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    expected_features = set(schema["features"])
    
    # Parse actual features from aggregation pipeline
    pipeline_path = Path("src/big_mood_detector/application/services/aggregation_pipeline.py")
    if not pipeline_path.exists():
        print(f"❌ Pipeline file not found: {pipeline_path}")
        sys.exit(1)
    
    pipeline_content = pipeline_path.read_text()
    
    # Find all feature keys in to_dict() method
    # Pattern matches: "feature_name_MN": self.feature_name_mean, etc.
    # Note: Also matches uppercase letters in feature names like "long_ST_MN"
    feature_pattern = re.compile(r'"([a-zA-Z_]+_(?:MN|SD|Z))"')
    found_features = set(feature_pattern.findall(pipeline_content))
    
    # Filter out non-Seoul features (activity, daily, etc.)
    seoul_features = {
        f for f in found_features 
        if any(f.startswith(prefix) for prefix in ["sleep_", "long_", "short_", "circadian_"])
    }
    
    # Check for discrepancies
    missing = sorted(expected_features - seoul_features)
    extra = sorted(seoul_features - expected_features)
    
    if missing or extra:
        print("❌ Feature schema drift detected!")
        if missing:
            print(f"   Missing features: {', '.join(missing)}")
        if extra:
            print(f"   Extra features: {', '.join(extra)}")
        print(f"\nExpected: {len(expected_features)} features")
        print(f"Found: {len(seoul_features)} features")
        sys.exit(1)
    
    print(f"✅ Found {len(seoul_features)}/{len(expected_features)} expected Seoul features")
    print("✅ Schema validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())