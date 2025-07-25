# PAT Training Scripts

## THE ONLY SCRIPT YOU NEED

### PAT-Conv-L Simple Training ✅
- **Script**: `scripts/pat_training/train_pat_conv_l_simple.py`
- **Best AUC**: 0.5929 (July 25, 2025)
- **Production Model**: `model_weights/production/pat_conv_l_v0.5929.pth`
- **Configuration**:
  - Optimizer: AdamW with uniform LR=1e-4
  - Scheduler: Cosine annealing over 15 epochs
  - Batch size: 32
  - Weight decay: 0.01
  - Gradient clipping: 1.0
  - No complex LP→FT phases
  - Conv patch embedding initialized randomly

### Key Success Factors
1. **Data Normalization**: Fixed to mean=0, std=1 (was std=0.045)
2. **Simple LR Schedule**: Uniform LR across all parameters
3. **Direct Fine-tuning**: Skip linear probing phase
4. **Stable Training**: No scheduler gymnastics

## Scripts to Use

1. **For Quick Results**: 
   ```bash
   python scripts/pat_training/train_pat_conv_l_simple.py
   ```

2. **For Multiple Runs**:
   ```bash
   python scripts/pat_training/train_pat_stable.py --model pat-l --conv --runs 3
   ```

## What Didn't Work
- Complex LP→FT phases with frozen encoder
- LambdaLR scheduler (cached initial_lr=0 bug)
- Large LR gaps between encoder (3e-5) and conv (1e-3)
- Multiple parameter groups with different schedules

## Next Steps to Reach 0.625 AUC
1. Grid search around working configuration:
   - encoder_lr ∈ {7e-5, 1e-4, 1.5e-4}
   - conv_multiplier ∈ {2x, 3x}
2. Run longer (20-25 epochs) with early stopping
3. Try different seeds for variance

## Clean Training Command
```bash
# Kill any existing sessions
tmux kill-server

# Start fresh training
tmux new -s pat_stable
python scripts/pat_training/train_pat_conv_l_simple.py
```