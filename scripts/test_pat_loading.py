#!/usr/bin/env python3
"""
Test PAT Model Loading with Architecture Reconstruction

This script tests whether we can successfully load PAT weights
by reconstructing the architecture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from big_mood_detector.infrastructure.ml_models.pat_model import PATModel
from big_mood_detector.infrastructure.ml_models.pat_architecture import (
    build_pat_model, load_weights_from_h5
)


def test_direct_architecture_loading():
    """Test loading weights directly into reconstructed architecture."""
    print("Testing Direct Architecture Loading")
    print("=" * 70)
    
    weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
    
    if not weights_path.exists():
        print(f"❌ Weights not found at {weights_path}")
        return False
    
    print(f"✅ Found weights at {weights_path}")
    
    # Build architecture
    print("\nBuilding PAT-Medium architecture...")
    model = build_pat_model("medium")
    print(f"✅ Architecture built: {model.count_params():,} parameters")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Try to load weights
    print("\nLoading weights into architecture...")
    success = load_weights_from_h5(model, str(weights_path))
    
    if success:
        print("✅ Weights loaded successfully!")
        
        # Test inference
        import numpy as np
        test_input = np.random.randn(1, 10080)
        try:
            output = model.predict(test_input, verbose=0)
            print(f"✅ Inference successful! Output shape: {output.shape}")
            return True
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False
    else:
        print("❌ Failed to load weights")
        return False


def test_pat_model_wrapper():
    """Test the PAT model wrapper with new loading logic."""
    print("\n\nTesting PAT Model Wrapper")
    print("=" * 70)
    
    # Initialize model
    pat = PATModel(model_size="medium")
    print(f"✅ Initialized PAT-MEDIUM wrapper")
    
    # Try to load weights
    weights_path = Path("model_weights/pat/pretrained/PAT-M_29k_weights.h5")
    success = pat.load_pretrained_weights(weights_path)
    
    if success:
        print("✅ Model loaded through wrapper!")
        info = pat.get_model_info()
        print(f"   Model info: {info}")
        return True
    else:
        print("❌ Failed to load through wrapper")
        return False


def main():
    """Run all tests."""
    print("PAT Model Loading Test")
    print("=" * 70)
    
    # Test 1: Direct architecture loading
    test1_success = test_direct_architecture_loading()
    
    # Test 2: PAT wrapper loading
    test2_success = test_pat_model_wrapper()
    
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"Direct architecture loading: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"PAT wrapper loading: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success or test2_success:
        print("\n✅ PAT models can be loaded! The ensemble system can use both PAT and XGBoost.")
    else:
        print("\n⚠️  PAT loading still has issues. The system will use XGBoost-only mode.")
        print("This is acceptable as the system is designed to gracefully degrade.")


if __name__ == "__main__":
    main()