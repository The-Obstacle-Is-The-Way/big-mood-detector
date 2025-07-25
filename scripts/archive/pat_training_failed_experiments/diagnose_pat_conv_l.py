#!/usr/bin/env python3
"""
Diagnostic script for PAT-Conv-L training issues.
Checks all hypotheses from the debugging plan.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pat_training.train_pat_conv_l import (
    SimplePATConvLModel, load_data
)


def diagnose_model():
    """Run comprehensive diagnostics on PAT-Conv-L setup."""
    
    print("="*60)
    print("PAT-Conv-L Diagnostic Script")
    print("="*60)
    
    # 1. Load data and check statistics
    print("\n1. DATA CHECK")
    print("-"*40)
    X_train, X_val, y_train, y_val = load_data()
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Data statistics:")
    print(f"  Train - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    print(f"  Val - Mean: {X_val.mean():.6f}, Std: {X_val.std():.6f}")
    print(f"Label distribution:")
    print(f"  Train - Positive: {y_train.mean():.3f} ({y_train.sum()}/{len(y_train)})")
    print(f"  Val - Positive: {y_val.mean():.3f} ({y_val.sum()}/{len(y_val)})")
    
    # 2. Create model and check architecture
    print("\n2. MODEL ARCHITECTURE CHECK")
    print("-"*40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePATConvLModel(model_size="large")
    
    # Load pretrained weights if available
    weights_path = Path("model_weights/pat/pretrained/PAT-L_29k_weights.h5")
    if weights_path.exists():
        print("Loading pretrained weights...")
        model.encoder.load_tf_weights(weights_path)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Head parameters: {head_params:,}")
    
    # 3. Test Linear Probing setup
    print("\n3. LINEAR PROBING SETUP CHECK")
    print("-"*40)
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Check which params are trainable
    encoder_frozen = []
    head_trainable = []
    for name, param in model.named_parameters():
        if 'encoder' in name and not param.requires_grad:
            encoder_frozen.append(name)
        elif param.requires_grad:
            head_trainable.append(name)
    
    print(f"Frozen encoder params: {len(encoder_frozen)}")
    print(f"Trainable head params: {len(head_trainable)}")
    print(f"Head param names: {head_trainable}")
    
    # Verify freezing
    encoder_requires_grad = any(p.requires_grad for p in model.encoder.parameters())
    head_requires_grad = all(p.requires_grad for p in model.head.parameters())
    print(f"Encoder requires_grad: {encoder_requires_grad} (should be False)")
    print(f"Head requires_grad: {head_requires_grad} (should be True)")
    
    # 4. Test forward pass and initial predictions
    print("\n4. FORWARD PASS & PREDICTIONS CHECK")
    print("-"*40)
    
    model.eval()
    with torch.no_grad():
        # Sample batch
        batch_size = 128
        x_sample = torch.FloatTensor(X_val[:batch_size]).to(device)
        y_sample = y_val[:batch_size]
        
        # Get embeddings and logits
        embeddings = model.encoder(x_sample)
        logits = model.head(embeddings).squeeze()
        probs = torch.sigmoid(logits)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings stats - Mean: {embeddings.mean():.3f}, Std: {embeddings.std():.3f}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats - Mean: {logits.mean():.3f}, Std: {logits.std():.3f}")
        print(f"Probs stats - Mean: {probs.mean():.3f}, Min: {probs.min():.3f}, Max: {probs.max():.3f}")
        
        # Check if predictions are inverted
        auc_normal = roc_auc_score(y_sample, logits.cpu().numpy())
        auc_inverted = roc_auc_score(y_sample, -logits.cpu().numpy())
        
        print(f"\nAUC with normal predictions: {auc_normal:.4f}")
        print(f"AUC with inverted predictions: {auc_inverted:.4f}")
        
        if auc_inverted > 0.55 and auc_inverted > auc_normal:
            print("⚠️  WARNING: Predictions appear to be inverted!")
    
    # 5. Test head initialization
    print("\n5. HEAD INITIALIZATION CHECK")
    print("-"*40)
    
    # Check head weights
    print(f"Head weight shape: {model.head.weight.shape}")
    print(f"Head weight stats - Mean: {model.head.weight.mean():.6f}, Std: {model.head.weight.std():.6f}")
    print(f"Head bias: {model.head.bias.item():.6f}")
    
    # Test with random head
    print("\nTesting with newly initialized head...")
    model.head = nn.Linear(96, 1).to(device)
    nn.init.xavier_uniform_(model.head.weight)
    nn.init.constant_(model.head.bias, 0.0)
    
    with torch.no_grad():
        logits_new = model(x_sample).squeeze()
        probs_new = torch.sigmoid(logits_new)
        auc_new = roc_auc_score(y_sample, logits_new.cpu().numpy())
        
        print(f"New head - Logits mean: {logits_new.mean():.3f}, Std: {logits_new.std():.3f}")
        print(f"New head - Probs mean: {probs_new.mean():.3f}")
        print(f"New head - AUC: {auc_new:.4f}")
    
    # 6. Test loss calculation
    print("\n6. LOSS CALCULATION CHECK")
    print("-"*40)
    
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    with torch.no_grad():
        y_tensor = torch.FloatTensor(y_sample).to(device)
        loss = criterion(logits_new, y_tensor)
        
        print(f"Pos weight: {pos_weight.item():.2f}")
        print(f"Initial loss: {loss.item():.4f}")
        print(f"Expected initial loss (random): ~{-np.log(0.5):.4f}")
        
        # Check if loss makes sense
        if loss.item() > 2.0:
            print("⚠️  WARNING: Loss seems too high for random initialization!")
    
    # 7. Summary and recommendations
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    issues = []
    
    if not head_requires_grad or encoder_requires_grad:
        issues.append("❌ Parameter freezing issue detected")
    
    if auc_inverted > 0.55 and auc_inverted > auc_normal:
        issues.append("❌ Predictions appear inverted")
    
    if abs(X_train.mean()) > 0.1 or abs(X_train.std() - 1.0) > 0.1:
        issues.append("❌ Data normalization looks incorrect")
    
    if loss.item() > 2.0:
        issues.append("❌ Initial loss too high")
    
    if embeddings.std() < 0.1:
        issues.append("❌ Embeddings have low variance")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ No obvious issues detected")
    
    print("\nRecommendations:")
    if auc_inverted > auc_normal:
        print("  1. Try negating the logits or flipping labels")
    if loss.item() > 2.0:
        print("  2. Check learning rate - might be too high")
    if embeddings.std() < 0.1:
        print("  3. Check if encoder weights loaded correctly")
    
    print("="*60)


if __name__ == "__main__":
    diagnose_model()