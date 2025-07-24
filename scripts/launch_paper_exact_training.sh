#!/bin/bash
# Launch PAT-L Paper-Exact Training for Depression (Target: 0.610 AUC)

echo "🎯 PAT-L Paper-Exact Training Launcher"
echo "Target: 0.610 AUC for Depression (PHQ-9)"
echo "========================================"

# Kill any existing training
tmux kill-session -t pat-paper 2>/dev/null || true

# Ensure we're in the right directory
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector

# Create directories
mkdir -p model_weights/pat/pytorch
mkdir -p logs/pat_training

# Launch training
tmux new-session -d -s pat-paper "
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:\$PYTHONPATH'
    
    echo '🚀 Starting PAT-L Paper-Exact Training...'
    echo 'Based on: AI Foundation Models for Wearable Movement Data in Mental Health Research'
    echo 'Target: 0.610 AUC (PAT Conv-L on Depression)'
    echo ''
    
    python scripts/pat_training/train_pat_l_paper_exact.py 2>&1 | tee logs/pat_training/paper_exact_$(date +%Y%m%d_%H%M%S).log
    
    echo ''
    echo '✅ Training completed!'
    echo 'Check model_weights/pat/pytorch/ for results'
    read -n 1 -s -r -p 'Press any key to exit...'
"

echo "✅ Training launched in tmux session 'pat-paper'"
echo ""
echo "📊 Commands:"
echo "  tmux attach -t pat-paper     # Watch training"
echo "  tmux kill-session -t pat-paper  # Stop training"
echo ""
echo "🎯 Key differences in this approach:"
echo "  - 2-stage training (frozen encoder → full fine-tuning)"
echo "  - Simple 2-layer head with ReLU (not complex architecture)"
echo "  - Different learning rates for encoder (5e-5) and head (5e-4)"
echo "  - Cosine annealing scheduler"
echo "  - Based on exact paper methodology"