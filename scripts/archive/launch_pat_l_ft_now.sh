#!/bin/bash
# Launch PAT-L Full Fine-Tuning for n=2800 (targeting 0.620 AUC)

SESSION="pat-l-ft"
LOG_FILE="logs/pat_training/pat_l_ft_$(date +%Y%m%d_%H%M%S).log"

# Kill any existing session
tmux kill-session -t $SESSION 2>/dev/null || true

# Ensure directories exist
mkdir -p model_weights/pat/pytorch
mkdir -p logs/pat_training

echo "🚀 Launching PAT-L Full Fine-Tuning (FT)"
echo "Target: 0.620 AUC (n=2800 from paper)"
echo ""

tmux new-session -d -s $SESSION "
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:\$PYTHONPATH'
    
    echo '🔬 PAT-L Full Fine-Tuning (FT)'
    echo 'Paper: AI Foundation Models for Wearable Movement Data'
    echo 'Target: 0.620 AUC (Table A.22, n=2800)'
    echo ''
    echo 'Training Configuration:'
    echo '- Model: PAT-L (standard, not Conv)'
    echo '- Method: Full Fine-Tuning (train everything)'
    echo '- Learning rate: 1e-4'
    echo '- Batch size: 32'
    echo '- Simple Linear(96,1) head'
    echo ''
    
    python scripts/pat_training/train_pat_l_simple_ft.py 2>&1 | tee $LOG_FILE
    
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
echo "🎯 Expected: ~0.620 AUC based on n=2800 dataset"