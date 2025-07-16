#!/usr/bin/env python3
"""
Convert XGBoost models from old format to current format.

This script loads the mood_ml XGBoost models and re-saves them in the current
XGBoost format to eliminate version warnings.
"""

import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import xgboost as xgb

    print(f"Using XGBoost version: {xgb.__version__}")
except ImportError:
    print("ERROR: XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)


def convert_model(input_path: Path, output_path: Path) -> bool:
    """
    Convert an XGBoost model to the current format.

    Args:
        input_path: Path to the old .pkl file
        output_path: Path to save the converted model

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\nConverting {input_path.name}...")

        # Load the old model
        with open(input_path, "rb") as f:
            old_model = pickle.load(f)

        # Check if it's actually an XGBoost model
        if not isinstance(
            old_model, xgb.Booster | xgb.XGBClassifier | xgb.XGBRegressor
        ):
            print("  ⚠️  Not an XGBoost model, skipping")
            return False

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Option 1: Save in native XGBoost format (recommended)
        native_path = output_path.with_suffix(".json")
        if hasattr(old_model, "save_model"):
            old_model.save_model(native_path)
            print(f"  ✅ Saved native format: {native_path.name}")

        # Option 2: Re-pickle with current version
        with open(output_path, "wb") as f:
            pickle.dump(old_model, f)
        print(f"  ✅ Saved pickle format: {output_path.name}")

        return True

    except Exception as e:
        print(f"  ❌ Error converting {input_path.name}: {e}")
        return False


def main():
    """Convert all mood_ml XGBoost models."""

    # Define paths
    mood_ml_dir = Path("reference_repos/mood_ml")
    output_dir = Path("model_weights/xgboost/converted")

    # Models to convert
    models = ["XGBoost_DE.pkl", "XGBoost_HME.pkl", "XGBoost_ME.pkl"]

    print("XGBoost Model Conversion")
    print("=" * 60)

    success_count = 0

    for model_name in models:
        input_path = mood_ml_dir / model_name
        output_path = output_dir / model_name

        if not input_path.exists():
            print(f"\n❌ Model not found: {input_path}")
            continue

        if convert_model(input_path, output_path):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Conversion complete: {success_count}/{len(models)} models converted")

    if success_count == len(models):
        print("\n✅ All models converted successfully!")
        print(f"\nConverted models saved to: {output_dir}")
        print("\nTo use the converted models, update the model path in:")
        print("  src/big_mood_detector/domain/services/mood_predictor.py")
        print("\nChange from:")
        print('  model_dir = Path("reference_repos/mood_ml")')
        print("To:")
        print(f'  model_dir = Path("{output_dir}")')
    else:
        print("\n⚠️  Some models failed to convert")

    # Test loading the converted models
    print("\n" + "-" * 60)
    print("Testing converted models...")

    for model_name in models:
        converted_path = output_dir / model_name
        if converted_path.exists():
            try:
                with open(converted_path, "rb") as f:
                    pickle.load(f)
                print(f"  ✅ {model_name} loads without warnings")
            except Exception as e:
                print(f"  ❌ {model_name} failed to load: {e}")


if __name__ == "__main__":
    main()
