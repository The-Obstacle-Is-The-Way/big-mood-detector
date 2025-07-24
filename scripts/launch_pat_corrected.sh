#!/bin/bash
# Launch PAT-L training with corrected data

SESSION="pat-corrected"
LOG_FILE="logs/pat_training/pat_l_corrected_$(date +%Y%m%d_%H%M%S).log"

# Kill any existing session
tmux kill-session -t $SESSION 2>/dev/null || true

# Ensure directories exist
mkdir -p model_weights/pat/pytorch
mkdir -p logs/pat_training

echo "🚀 Launching PAT-L Training with CORRECTED Data"
echo "Data has proper StandardScaler normalization"
echo "Target: 0.620 AUC"
echo ""

tmux new-session -d -s $SESSION "
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:\$PYTHONPATH'
    
    python scripts/pat_training/train_pat_l_corrected.py 2>&1 | tee $LOG_FILE
    
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
echo "Expected: AUC should improve from ~0.57 to ~0.62!"