#!/usr/bin/env python3
"""
Verify Big Mood Detector setup - checks all required files and directories.
Run this after cloning to ensure everything is properly configured.
"""
import sys
from pathlib import Path


def check_file(path: str, required: bool = True) -> bool:
    """Check if a file/directory exists and print status."""
    path_obj = Path(path)
    exists = path_obj.exists()

    # For directories, check if they have content
    if exists and path_obj.is_dir():
        if path.endswith("health_auto_export/"):
            exists = any(path_obj.glob("*.json"))
        elif path.endswith("apple_export/"):
            exists = (path_obj / "export.xml").exists()

    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (REQUIRED)" if required else " (optional)"
    print(f"{status} {path}{req_text}")

    return exists


def main():
    """Main verification routine."""
    print("🔍 Checking Big Mood Detector Setup...\n")
    print("=" * 60)

    all_good = True

    # Required model files
    print("\n📊 Required Model Files:")
    print("-" * 40)
    models_ok = True
    models_ok &= check_file("model_weights/xgboost/converted/XGBoost_DE.json")
    models_ok &= check_file("model_weights/xgboost/converted/XGBoost_HME.json")
    models_ok &= check_file("model_weights/xgboost/converted/XGBoost_ME.json")
    all_good &= models_ok

    # Optional PAT models
    print("\n🧠 Optional PAT Models:")
    print("-" * 40)
    check_file("model_weights/pat/pretrained/PAT-S_29k_weights.h5", False)
    check_file("model_weights/pat/pretrained/PAT-M_29k_weights.h5", False)
    check_file("model_weights/pat/pretrained/PAT-L_29k_weights.h5", False)

    # Health data (need at least one source)
    print("\n📱 Health Data Sources (need at least one):")
    print("-" * 40)
    has_json = check_file("data/input/health_auto_export/", False)
    has_xml = check_file("data/input/apple_export/export.xml", False)
    has_data = has_json or has_xml

    if not has_data:
        print("\n⚠️  No health data found!")
        all_good = False

    # Optional training data
    print("\n📚 Training Data (for developers only):")
    print("-" * 40)
    check_file("data/nhanes/2013-2014/PAXMIN_H.xpt", False)
    check_file("data/nhanes/2013-2014/DPQ_H.xpt", False)
    check_file("data/nhanes/2013-2014/DEMO_H.xpt", False)
    check_file("data/cache/nhanes_pat_data_subsetNone.npz", False)

    # Check directories
    print("\n📁 Required Directories:")
    print("-" * 40)
    dirs = [
        "data/baselines",
        "data/cache",
        "logs",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        check_file(dir_path, True)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print("=" * 60)

    if all_good:
        print("✅ Basic setup complete! You can run predictions.")
        print("\n🚀 Quick start commands:")
        print("   python src/big_mood_detector/main.py process data/input/")
        print("   python src/big_mood_detector/main.py predict data/input/ --report")
    else:
        print("❌ Setup incomplete!")
        print("\n📋 Next steps:")
        if not models_ok:
            print("1. Download XGBoost models:")
            print("   python scripts/maintenance/download_model_weights.py")
        if not has_data:
            print("2. Add your health data:")
            print("   - Apple Health: Export from iPhone → data/input/apple_export/")
            print("   - Health Auto Export: JSON files → data/input/health_auto_export/")
        print("\n📖 See DATA_SETUP_GUIDE.md for detailed instructions")

    print("\n" + "=" * 60)
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
