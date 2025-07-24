#!/usr/bin/env python3
"""
Debug PAT training issues - check data and weights.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.ml_models.pat_pytorch import PATDepressionNet


def check_data_quality():
    """Check NHANES data loading and labels."""
    print("\n🔍 Checking Data Quality...")

    # Load cached data
    cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
    if not cache_path.exists():
        print("❌ No cached data found. Run training script first.")
        return None, None, None, None

    data = np.load(cache_path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    print("\n📊 Dataset Statistics:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Train positive: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
    print(f"  Val positive: {sum(y_val)} ({100*sum(y_val)/len(y_val):.1f}%)")

    # Check for data issues
    print("\n🔍 Data Quality Checks:")

    # Check for NaNs
    train_nans = np.isnan(X_train).sum()
    val_nans = np.isnan(X_val).sum()
    print(f"  NaNs in train: {train_nans}")
    print(f"  NaNs in val: {val_nans}")

    # Check value ranges
    print(f"  Train value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  Val value range: [{X_val.min():.2f}, {X_val.max():.2f}]")

    # Check for all-zero sequences
    zero_sequences_train = (X_train.sum(axis=1) == 0).sum()
    zero_sequences_val = (X_val.sum(axis=1) == 0).sum()
    print(f"  All-zero sequences in train: {zero_sequences_train}")
    print(f"  All-zero sequences in val: {zero_sequences_val}")

    # Check activity patterns
    mean_activity_train = X_train.mean(axis=1)
    mean_activity_val = X_val.mean(axis=1)

    print("\n📈 Activity Level Statistics:")
    print(f"  Train mean activity: {mean_activity_train.mean():.2f} (±{mean_activity_train.std():.2f})")
    print(f"  Val mean activity: {mean_activity_val.mean():.2f} (±{mean_activity_val.std():.2f})")

    # Check label distribution by activity level
    high_activity_mask_train = mean_activity_train > np.percentile(mean_activity_train, 75)
    low_activity_mask_train = mean_activity_train < np.percentile(mean_activity_train, 25)

    high_activity_positive_rate = y_train[high_activity_mask_train].mean()
    low_activity_positive_rate = y_train[low_activity_mask_train].mean()

    print("\n🏷️ Label Distribution by Activity:")
    print(f"  High activity positive rate: {100*high_activity_positive_rate:.1f}%")
    print(f"  Low activity positive rate: {100*low_activity_positive_rate:.1f}%")

    # Plot sample sequences
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot positive examples
    pos_indices = np.where(y_train == 1)[0][:2]
    for i, idx in enumerate(pos_indices):
        axes[0, i].plot(X_train[idx, :1440])  # First day
        axes[0, i].set_title(f'Positive Example {i+1} (Day 1)')
        axes[0, i].set_xlabel('Minutes')
        axes[0, i].set_ylabel('Activity')

    # Plot negative examples
    neg_indices = np.where(y_train == 0)[0][:2]
    for i, idx in enumerate(neg_indices):
        axes[1, i].plot(X_train[idx, :1440])  # First day
        axes[1, i].set_title(f'Negative Example {i+1} (Day 1)')
        axes[1, i].set_xlabel('Minutes')
        axes[1, i].set_ylabel('Activity')

    plt.tight_layout()
    plt.savefig('analysis/pat_training/sample_sequences.png', dpi=150)
    plt.close()
    print("\n📊 Sample sequences plotted to analysis/pat_training/sample_sequences.png")

    return X_train, y_train, X_val, y_val


def check_model_initialization():
    """Check model weights and initialization."""
    print("\n🔍 Checking Model Initialization...")

    # Create model
    model = PATDepressionNet(model_size="large", unfreeze_last_n=0)

    # Check if pretrained weights exist
    weights_paths = [
        Path("model_weights/pat/pytorch/pat_large_weights.pt"),
        Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5"),
        Path("reference_repos/Pretrained-Actigraphy-Transformer/model_weights/PAT-L_29k_weights.h5")
    ]

    weights_found = False
    for path in weights_paths:
        if path.exists():
            print(f"✅ Found pretrained weights at: {path}")
            weights_found = True

            # Try to load weights
            try:
                if path.suffix == '.h5':
                    success = model.encoder.load_tf_weights(path)
                    if success:
                        print("✅ Successfully loaded TensorFlow weights")
                else:
                    state_dict = torch.load(path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=False)
                    print("✅ Successfully loaded PyTorch weights")
            except Exception as e:
                print(f"❌ Failed to load weights: {e}")

            break

    if not weights_found:
        print("❌ No pretrained weights found!")
        print("   This is likely why training performance is poor.")
        print("\n   To fix:")
        print("   1. Download PAT-L weights from the reference repo")
        print("   2. Place in model_weights/pat/pretrained/PAT-L_29k_weights.h5")

    # Check head initialization
    print("\n🔍 Checking Head Initialization:")
    for name, param in model.head.named_parameters():
        values = param.data.numpy().flatten()
        print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}")

    # Test forward pass
    print("\n🔍 Testing Forward Pass:")
    dummy_input = torch.randn(2, 10080)

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful: output shape = {output.shape}")
        print(f"   Output values: {output.numpy().flatten()}")

        # Check if outputs are reasonable
        probs = torch.sigmoid(output).numpy().flatten()
        print(f"   Sigmoid outputs: {probs}")

        if all(abs(p - 0.5) < 0.01 for p in probs):
            print("⚠️  All outputs near 0.5 - model may not be learning")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

    return model


def test_simple_baseline():
    """Test with a simple linear model to establish baseline."""
    print("\n🔍 Testing Simple Linear Baseline...")

    # Load data
    X_train, y_train, X_val, y_val = check_data_quality()
    if X_train is None:
        return

    # Create simple model
    class SimpleLinearModel(torch.nn.Module):
        def __init__(self, input_dim=10080):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.fc(x)

    # Use mean activity as a simple feature
    X_train_mean = X_train.mean(axis=1, keepdims=True)
    X_val_mean = X_val.mean(axis=1, keepdims=True)

    # Quick training
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_mean, y_train)

    # Evaluate
    from sklearn.metrics import roc_auc_score
    train_pred = lr.predict_proba(X_train_mean)[:, 1]
    val_pred = lr.predict_proba(X_val_mean)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred)
    val_auc = roc_auc_score(y_val, val_pred)

    print("\n📊 Simple Mean Activity Baseline:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Val AUC: {val_auc:.4f}")

    if val_auc > 0.5:
        print("\n✅ Baseline works! This confirms:")
        print("   - Data and labels are correctly aligned")
        print("   - There is signal in the data")
        print("   - The issue is likely with PAT model training")
    else:
        print("\n❌ Even simple baseline fails!")
        print("   - Check data loading pipeline")
        print("   - Verify depression labels are correct")


def main():
    # Create output directory
    Path("analysis/pat_training").mkdir(parents=True, exist_ok=True)

    print("🔍 PAT Training Debugger")
    print("=" * 50)

    # Check data
    check_data_quality()

    # Check model
    check_model_initialization()

    # Test baseline
    test_simple_baseline()

    print("\n" + "=" * 50)
    print("🎯 Summary and Recommendations:")
    print("\n1. If pretrained weights are missing:")
    print("   - This is the most likely cause of poor performance")
    print("   - Download from reference repository")
    print("\n2. If data checks show issues:")
    print("   - Fix data preprocessing pipeline")
    print("   - Check label alignment")
    print("\n3. If simple baseline works but PAT doesn't:")
    print("   - Start with higher learning rate (1e-2)")
    print("   - Use simpler head architecture first")
    print("   - Check gradient flow")


if __name__ == "__main__":
    main()
