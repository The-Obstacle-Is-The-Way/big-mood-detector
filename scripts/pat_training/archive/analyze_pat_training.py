#!/usr/bin/env python3
"""
Analyze PAT training checkpoints and suggest improvements.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def analyze_checkpoint(checkpoint_path: Path):
    """Analyze a training checkpoint."""
    print(f"\n📊 Analyzing checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract info
    epoch = checkpoint.get('epoch', 'unknown')
    val_auc = checkpoint.get('val_auc', 0)
    val_pr_auc = checkpoint.get('val_pr_auc', 0)
    val_f1 = checkpoint.get('val_f1', 0)

    print("\n📈 Performance Metrics:")
    print(f"  Epoch: {epoch}")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Validation PR-AUC: {val_pr_auc:.4f}")
    print(f"  Validation F1: {val_f1:.4f}")

    # Analyze model state
    model_state = checkpoint.get('model_state_dict', {})
    if model_state:
        print("\n🔍 Model Analysis:")

        # Count parameters
        total_params = 0

        for name, param in model_state.items():
            params = param.numel()
            total_params += params
            print(f"  {name}: {param.shape} ({params:,} params)")

        print(f"\n  Total parameters: {total_params:,}")

    # Analyze optimizer state
    optimizer_state = checkpoint.get('optimizer_state_dict', {})
    if optimizer_state and 'param_groups' in optimizer_state:
        print("\n⚙️ Optimizer State:")
        for i, group in enumerate(optimizer_state['param_groups']):
            lr = group.get('lr', 0)
            weight_decay = group.get('weight_decay', 0)
            print(f"  Group {i}: LR={lr:.2e}, Weight Decay={weight_decay}")

    return checkpoint


def suggest_improvements(val_auc: float, epoch: int):
    """Suggest improvements based on current performance."""
    print(f"\n💡 Suggestions for improving AUC from {val_auc:.4f}:")

    if val_auc < 0.50:
        print("\n🔴 Performance is below random (0.50). Major changes needed:")
        print("  1. Check data loading and labels - ensure correct alignment")
        print("  2. Try a much higher learning rate (1e-2 for head)")
        print("  3. Reduce model complexity - try single linear layer first")
        print("  4. Verify pretrained weights are loaded correctly")
        print("  5. Check for class imbalance - adjust pos_weight")

    elif val_auc < 0.55:
        print("\n🟡 Performance is marginal. Try these approaches:")
        print("  1. Unfreeze more encoder layers (3-4 blocks)")
        print("  2. Use differential learning rates (encoder: 1e-5, head: 5e-4)")
        print("  3. Add data augmentation (time shifting, noise)")
        print("  4. Try different architectures:")
        print("     - Deeper head: 96 → 256 → 128 → 64 → 1")
        print("     - Add batch normalization")
        print("     - Use GELU instead of ReLU")
        print("  5. Implement label smoothing (smooth factor: 0.1)")

    elif val_auc < 0.60:
        print("\n🟢 Decent performance. Fine-tuning suggestions:")
        print("  1. Fine-tune the entire encoder with very low LR (1e-6)")
        print("  2. Implement mixup augmentation")
        print("  3. Try ensemble of different model sizes")
        print("  4. Use focal loss for hard examples")
        print("  5. Implement curriculum learning")

    else:
        print("\n✅ Good performance! Advanced techniques:")
        print("  1. Implement stochastic weight averaging (SWA)")
        print("  2. Use progressive resizing of sequences")
        print("  3. Try adversarial training")
        print("  4. Implement pseudo-labeling on unlabeled data")

    print("\n📝 General recommendations:")
    print("  - Monitor gradient norms - if too small, increase LR")
    print("  - Use gradient accumulation for larger effective batch size")
    print("  - Try different optimizers: Ranger, AdaBound, Lion")
    print("  - Implement early stopping with patience=20-30")


def plot_weight_distribution(checkpoint_path: Path, output_dir: Path):
    """Plot weight distributions to check for issues."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model_state_dict', {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Select key layers to visualize
    layers_to_plot = [
        ('head.0.weight', 'Head Linear 1'),
        ('head.3.weight', 'Head Linear 2'),
        ('encoder.patch_embed.weight', 'Patch Embedding'),
        ('encoder.blocks.0.attention.q_proj.weight', 'First Attention Q')
    ]

    for idx, (layer_name, title) in enumerate(layers_to_plot):
        if idx >= len(axes):
            break

        if layer_name in model_state:
            weights = model_state[layer_name].numpy().flatten()
            axes[idx].hist(weights, bins=50, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{title}\nMean: {weights.mean():.4f}, Std: {weights.std():.4f}')
            axes[idx].set_xlabel('Weight Value')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'weight_distributions.png', dpi=150)
    plt.close()

    print(f"\n📊 Weight distribution plot saved to {output_dir / 'weight_distributions.png'}")


def create_training_recipe(output_path: Path, current_auc: float):
    """Create a detailed training recipe based on current performance."""

    recipe = {
        "current_auc": current_auc,
        "target_auc": 0.61,  # Paper's reported performance
        "stages": []
    }

    if current_auc < 0.50:
        # Stage 1: Fix fundamental issues
        recipe["stages"].append({
            "name": "Debug and Baseline",
            "epochs": 20,
            "config": {
                "unfreeze_last_n": 0,
                "head_lr": 1e-2,
                "batch_size": 64,
                "architecture": "single_linear",
                "dropout": 0.1
            },
            "expected_auc": 0.52
        })

    # Stage 2: Progressive unfreezing
    recipe["stages"].append({
        "name": "Progressive Unfreezing",
        "epochs": 50,
        "config": {
            "unfreeze_last_n": 2,
            "head_lr": 5e-4,
            "encoder_lr": 1e-5,
            "batch_size": 32,
            "architecture": "2_layer_gelu",
            "dropout": 0.3,
            "augmentation": True
        },
        "expected_auc": 0.56
    })

    # Stage 3: Full fine-tuning
    recipe["stages"].append({
        "name": "Full Fine-tuning",
        "epochs": 30,
        "config": {
            "unfreeze_last_n": 4,  # All blocks for PAT-L
            "head_lr": 1e-4,
            "encoder_lr": 5e-6,
            "batch_size": 16,
            "gradient_accumulation": 2,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2
        },
        "expected_auc": 0.60
    })

    # Stage 4: Advanced techniques
    recipe["stages"].append({
        "name": "Advanced Optimization",
        "epochs": 20,
        "config": {
            "optimizer": "ranger",
            "swa": True,
            "focal_loss": True,
            "curriculum_learning": True
        },
        "expected_auc": 0.61
    })

    with open(output_path, 'w') as f:
        json.dump(recipe, f, indent=2)

    print(f"\n📋 Training recipe saved to {output_path}")

    # Print summary
    print("\n🎯 Training Recipe Summary:")
    for i, stage in enumerate(recipe["stages"], 1):
        print(f"\nStage {i}: {stage['name']}")
        print(f"  Epochs: {stage['epochs']}")
        print(f"  Expected AUC: {stage['expected_auc']}")
        print(f"  Key changes: {', '.join(list(stage['config'].keys())[:3])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='model_weights/pat/pytorch/pat_l_stage1_*/best_auc_*.pt')
    parser.add_argument('--output-dir', type=str, default='analysis/pat_training')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoint
    checkpoint_paths = list(Path('.').glob(args.checkpoint))
    if not checkpoint_paths:
        print(f"❌ No checkpoint found matching: {args.checkpoint}")

        # Analyze from the training output instead
        print("\n📊 Based on your training logs:")
        print("  Best AUC: 0.4756 at epoch 51")
        print("  Training plateaued - no improvement for 20 epochs")

        val_auc = 0.4756
        suggest_improvements(val_auc, 70)
        create_training_recipe(output_dir / 'training_recipe.json', val_auc)

    else:
        # Analyze checkpoint
        checkpoint_path = sorted(checkpoint_paths)[-1]  # Latest
        checkpoint = analyze_checkpoint(checkpoint_path)

        val_auc = checkpoint.get('val_auc', 0)
        epoch = checkpoint.get('epoch', 0)

        suggest_improvements(val_auc, epoch)
        plot_weight_distribution(checkpoint_path, output_dir)
        create_training_recipe(output_dir / 'training_recipe.json', val_auc)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
