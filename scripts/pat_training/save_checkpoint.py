#!/usr/bin/env python3
"""
Emergency checkpoint save script for PAT-L training.
Run this in the same environment as your training to save state.
"""
from datetime import datetime
from pathlib import Path


def save_training_checkpoint():
    """Save current training state before migration."""

    # You'll need to update these values based on your current training
    checkpoint = {
        'timestamp': datetime.now().isoformat(),
        'stage': 2,
        'epoch_in_stage': 5,
        'total_epochs': 35,  # 30 stage1 + 5 stage2
        'best_auc': 0.5788,
        'current_auc': 0.5514,  # Last logged value
        'training_config': {
            'model_size': 'large',
            'batch_size': 32,
            'stage1_lr': 5e-3,
            'stage2_lr': 1e-4,
            'unfrozen_blocks': 2,
            'dropout': 0.1,
            'pos_weight': 2.44,  # Approximate from logs
            'device': 'mps',
        },
        'notes': {
            'issue': 'AUC plateaued around 0.54-0.57',
            'stage1_best': 'Epoch 1 with 0.5788',
            'stage2_status': 'No improvement over stage1',
            'migration_reason': 'Mac resource constraints, slow training'
        }
    }

    # Save checkpoint info
    save_path = Path("model_weights/pat/pytorch/pat_l_training/migration_checkpoint_info.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(save_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"✅ Saved checkpoint info to: {save_path}")
    print("\n⚠️  IMPORTANT: In your tmux session, add this to save model weights:")
    print("""
# Add this to your training script:
import torch
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_auc': val_auc,
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
}, 'model_weights/pat/pytorch/pat_l_training/checkpoint_migration.pt')
print("Checkpoint saved!")
""")


if __name__ == "__main__":
    save_training_checkpoint()
