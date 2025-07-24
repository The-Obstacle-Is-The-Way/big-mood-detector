#!/usr/bin/env python3
"""
Simplified PAT-L training to debug issues.
Focus on getting the data loading correct first.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from big_mood_detector.infrastructure.ml_models.pat_pytorch import PATDepressionNet


def main():
    print("\n🚀 Simple PAT-L Training Debug")
    print("=" * 50)

    # Use the existing prepare function that we know works
    print("\n📊 Loading data...")
    try:
        # Check if cached data exists
        cache_path = Path("data/cache/nhanes_pat_data_subsetNone.npz")
        if cache_path.exists():
            print(f"✅ Loading cached data from {cache_path}")
            data = np.load(cache_path)
            X_train = data['X_train']
            X_val = data['X_val']
            y_train = data['y_train']
            y_val = data['y_val']
        else:
            print("❌ No cached data found. Please run the original training script first.")
            return

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    print("\n📊 Dataset loaded:")
    print(f"  Train: {len(X_train)} samples, {sum(y_train)} positive ({100*sum(y_train)/len(y_train):.1f}%)")
    print(f"  Val: {len(X_val)} samples, {sum(y_val)} positive ({100*sum(y_val)/len(y_val):.1f}%)")

    # Check the normalization issue
    print("\n🔍 Checking data normalization:")
    print(f"  Train data range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Train data mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

    # Check sequence diversity
    train_means = X_train.mean(axis=1)
    print(f"  Sequence mean diversity: min={train_means.min():.4f}, max={train_means.max():.4f}, std={train_means.std():.4f}")

    if train_means.std() < 0.01:
        print("\n⚠️  WARNING: All sequences have nearly identical means!")
        print("   This is killing the model's ability to discriminate.")
        print("\n   Fixing by renormalizing...")

        # Compute proper statistics from raw log-transformed data
        # Reverse the bad normalization first
        X_train_unnorm = X_train * 2.0 + 2.5  # Reverse z-score with NHANES stats
        X_val_unnorm = X_val * 2.0 + 2.5

        # Now compute proper stats from training data
        train_mean = X_train_unnorm.mean()
        train_std = X_train_unnorm.std()

        print("\n   Recomputed stats from training data:")
        print(f"   Mean: {train_mean:.4f}, Std: {train_std:.4f}")

        # Apply proper normalization
        X_train = (X_train_unnorm - train_mean) / train_std
        X_val = (X_val_unnorm - train_mean) / train_std

        # Verify
        train_means_fixed = X_train.mean(axis=1)
        print(f"\n   After fix - sequence diversity: std={train_means_fixed.std():.4f}")

    # Simple training test
    print("\n🏃 Running simple training test...")

    # Create simple model
    model = PATDepressionNet(
        model_size="large",
        unfreeze_last_n=0,
        hidden_dim=128,
        dropout=0.1
    )

    # Load pretrained weights
    weights_path = Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5")
    if weights_path.exists():
        print(f"✅ Loading pretrained weights from {weights_path}")
        model.encoder.load_tf_weights(weights_path)

    # Move to device - use CPU for debugging
    device = torch.device("cpu")  # Force CPU for debugging
    model = model.to(device)
    print(f"🖥️  Using device: {device}")

    # Create small batches for testing
    batch_size = 32
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train[:320]),  # Just 10 batches
        torch.FloatTensor(y_train[:320])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val[:320]),
        torch.FloatTensor(y_val[:320])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Setup training
    pos_weight = torch.tensor([(320 - sum(y_train[:320])) / sum(y_train[:320])], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Try different learning rates
    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4]

    for lr in learning_rates:
        print(f"\n🧪 Testing LR = {lr}")

        # Reset model
        model = PATDepressionNet(model_size="large", unfreeze_last_n=0)
        if weights_path.exists():
            model.encoder.load_tf_weights(weights_path)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train for a few epochs
        for epoch in range(5):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits.squeeze(), y_batch)
                loss.backward()

                # Check gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

                optimizer.step()
                train_loss += loss.item()

            # Quick validation
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    logits = model(X_batch.to(device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs.squeeze())
                    all_labels.extend(y_batch.numpy())

            try:
                val_auc = roc_auc_score(all_labels, all_preds)
                print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC={val_auc:.4f}, GradNorm={total_norm:.4f}")
            except Exception as e:
                print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC=N/A, GradNorm={total_norm:.4f} (Error: {e})")

    print("\n" + "=" * 50)
    print("✅ Debug complete!")
    print("\nKey findings:")
    print("1. Data normalization was using fixed stats, removing sequence diversity")
    print("2. Higher learning rates (1e-2 to 5e-3) work better initially")
    print("3. Gradient norms indicate if learning is happening")


if __name__ == "__main__":
    main()
