#!/bin/bash
# Launch PAT-L training with higher LR and cosine schedule

SESSION="pat-higher-lr"
LOG_FILE="logs/pat_training/pat_l_higher_lr_$(date +%Y%m%d_%H%M%S).log"

# Kill any existing session
tmux kill-session -t $SESSION 2>/dev/null || true

# Ensure directories exist
mkdir -p model_weights/pat/pytorch
mkdir -p logs/pat_training

echo "🚀 Launching PAT-L Training with HIGHER LR"
echo "Key improvements:"
echo "  - Encoder LR: 5e-5 (was 2e-5)"
echo "  - Cosine annealing schedule"
echo "  - Starting from best checkpoint (0.5686)"
echo ""

tmux new-session -d -s $SESSION "
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:\$PYTHONPATH'
    
    python scripts/pat_training/train_pat_l_higher_lr.py 2>&1 | tee $LOG_FILE
    
    echo ''
    echo '✅ Training completed!'
    read -n 1 -s -r -p 'Press any key to exit...'
"

echo "✅ Training launched in tmux session '$SESSION'"
echo ""
echo "📊 Commands:"
echo "  tmux attach -t $SESSION      # Watch training"
echo "  tail -f $LOG_FILE            # View logs"
echo ""
echo "Expected: Should quickly jump to ~0.59 and push toward 0.62!"