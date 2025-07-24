#!/bin/bash
# Launch optimized PAT-L training with better hyperparameters

SESSION="pat-l-opt"
LOG_FILE="logs/pat_training/pat_l_optimized_$(date +%Y%m%d_%H%M%S).log"

# Kill any existing session
tmux kill-session -t $SESSION 2>/dev/null || true

# Ensure directories exist
mkdir -p model_weights/pat/pytorch
mkdir -p logs/pat_training

echo "🚀 Launching Optimized PAT-L Training"
echo "Target: 0.620 AUC (n=2800)"
echo "Key improvements:"
echo "  - Encoder LR: 5e-6 (preserve pretrained features)"
echo "  - Head LR: 5e-4 (faster adaptation)"
echo "  - Cosine annealing scheduler"
echo "  - Better gradient clipping (0.5)"
echo ""

tmux new-session -d -s $SESSION "
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:\$PYTHONPATH'
    
    python scripts/pat_training/train_pat_l_optimized.py 2>&1 | tee $LOG_FILE
    
    echo ''
    echo '✅ Training completed!'
    read -n 1 -s -r -p 'Press any key to exit...'
"

echo "✅ Training launched in tmux session '$SESSION'"
echo ""
echo "📊 Commands:"
echo "  tmux attach -t $SESSION      # Watch training live"
echo "  tail -f $LOG_FILE            # View logs"
echo ""