#!/bin/bash
# Launch clean PAT-L training (FT or LP) based on paper methodology

echo "🎯 PAT-L Clean Training Launcher"
echo "Based on exact paper methodology"
echo "================================"
echo ""
echo "Choose training method:"
echo "1) Full Fine-Tuning (FT) - Target: 0.589 AUC"
echo "2) Linear Probing (LP) - Target: 0.582 AUC"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        METHOD="FT"
        SCRIPT="train_pat_l_simple_ft.py"
        TARGET="0.589"
        SESSION="pat-l-ft"
        ;;
    2)
        METHOD="LP"
        SCRIPT="train_pat_l_simple_lp.py"
        TARGET="0.582"
        SESSION="pat-l-lp"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Kill any existing session
tmux kill-session -t $SESSION 2>/dev/null || true

# Ensure directories exist
mkdir -p model_weights/pat/pytorch
mkdir -p logs/pat_training

# Launch training
echo ""
echo "🚀 Launching PAT-L ($METHOD) training..."
echo "Target AUC: $TARGET"
echo ""

LOG_FILE="logs/pat_training/pat_l_${METHOD}_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s $SESSION "
    cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:\$PYTHONPATH'
    
    echo '🔬 PAT-L $METHOD Training'
    echo 'Paper: AI Foundation Models for Wearable Movement Data'
    echo 'Target: $TARGET AUC'
    echo ''
    echo 'Key differences from paper:'
    echo '- FT: Train everything from start (no staging)'
    echo '- LP: Freeze encoder completely, train head only'
    echo '- Simple head: Linear(96, 1) + BCEWithLogitsLoss'
    echo ''
    
    python scripts/pat_training/$SCRIPT 2>&1 | tee $LOG_FILE
    
    echo ''
    echo '✅ Training completed! Check results in model_weights/pat/pytorch/'
    read -n 1 -s -r -p 'Press any key to exit...'
"

echo "✅ Training launched in tmux session '$SESSION'"
echo ""
echo "📊 Commands:"
echo "  tmux attach -t $SESSION      # Watch training"
echo "  tail -f $LOG_FILE            # View logs"
echo "  tmux kill-session -t $SESSION # Stop training"
echo ""
echo "🎯 Expected results based on paper:"
echo "  - FT should converge to ~0.589 AUC"
echo "  - LP should converge to ~0.582 AUC"
echo "  - Training should take 10-30 minutes on RTX 4090"