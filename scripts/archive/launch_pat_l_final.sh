#!/bin/bash
# Launch final PAT-L training with proper hyperparameters

SESSION="pat-l-final"
LOG_FILE="logs/pat_training/pat_l_final_$(date +%Y%m%d_%H%M%S).log"

# Kill any existing session
tmux kill-session -t $SESSION 2>/dev/null || true

# Ensure directories exist
mkdir -p model_weights/pat/pytorch
mkdir -p logs/pat_training

echo "🚀 Launching Final PAT-L Training"
echo "Target: 0.620 AUC (Paper Table A.22, n=2800)"
echo ""
echo "Key changes from previous run:"
echo "  ✅ Encoder LR: 2e-5 (was 5e-6 - 4x increase!)"
echo "  ✅ Head LR: 5e-4"
echo "  ✅ Cosine scheduler with T_max=30"
echo "  ✅ Batch size: 64 (if GPU allows)"
echo "  ✅ More patience: 15 epochs"
echo ""

tmux new-session -d -s $SESSION "
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:\$PYTHONPATH'
    
    echo '🔬 PAT-L Final Training Run'
    echo '=========================='
    echo ''
    
    python scripts/pat_training/train_pat_l_final.py 2>&1 | tee $LOG_FILE
    
    echo ''
    echo '✅ Training completed! Check results above.'
    read -n 1 -s -r -p 'Press any key to exit...'
"

echo "✅ Training launched in tmux session '$SESSION'"
echo ""
echo "📊 Commands:"
echo "  tmux attach -t $SESSION      # Watch training live"
echo "  tail -f $LOG_FILE            # View logs only"
echo ""
echo "🎯 Expected: Should reach ~0.60 AUC within 10 epochs"
echo "            and climb to 0.620 by epoch 20-30"