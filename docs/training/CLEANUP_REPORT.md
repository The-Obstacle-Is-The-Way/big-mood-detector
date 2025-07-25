# Training Cleanup Report
Generated: 2025-07-25 13:19:40

## Production Model
- PAT-Conv-L v0.5929: `model_weights/production/pat_conv_l_v0.5929.pth`
- Metadata: `model_weights/production/pat_conv_l_v0.5929.json`

## Key Training Logs
- PAT-Conv-L (best): `training/logs/pat_conv_l_v0.5929_20250725.log`
- PAT-L (previous best): `training/logs/pat_l_v0.5888_20250724.log`

## Archived Files
- Old models: `training/experiments/archived/old_models_*.tar.gz`
- Old logs: Original locations preserved for reference

## Next Steps
1. Update training scripts to use new paths
2. Update model loading code to use production path
3. Delete old files after verifying archives

## Directory Structure
```
training/
├── experiments/
│   ├── active/      # Current experiments
│   └── archived/    # Old experiments
├── logs/           # Canonical training logs  
└── results/        # Training summaries

model_weights/
├── production/     # Production-ready models
└── pretrained/     # Original pretrained weights
```
