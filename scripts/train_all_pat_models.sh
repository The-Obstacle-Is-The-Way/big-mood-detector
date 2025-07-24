#!/bin/bash
# Automated PAT Model Training Pipeline
# Trains Small → Medium → Large sequentially

set -e  # Exit on error

echo "🚀 Starting PAT Model Training Pipeline"
echo "$(date): Beginning automated training sequence"

# Function to check if training is complete
check_training_complete() {
    if ! pgrep -f "train_pat_depression_head_full.py" > /dev/null; then
        return 0  # Training complete
    else
        return 1  # Still training
    fi
}

# Function to wait for training completion
wait_for_completion() {
    local model_name=$1
    echo "⏳ Waiting for $model_name training to complete..."
    
    while ! check_training_complete; do
        echo "$(date): $model_name still training..."
        sleep 300  # Check every 5 minutes
    done
    
    echo "✅ $model_name training completed at $(date)"
}

# PAT-S is already running, so just wait for it
echo "📊 PAT-S training is already running..."
wait_for_completion "PAT-S"

# Train PAT-M
echo "🚀 Starting PAT-M training..."
python3 scripts/train_pat_depression_head_full.py \
    --model-size medium \
    --device mps \
    --epochs 50 &

wait_for_completion "PAT-M"

# Train PAT-L (optional)
echo "🚀 Starting PAT-L training..."
python3 scripts/train_pat_depression_head_full.py \
    --model-size large \
    --device mps \
    --epochs 50 &

wait_for_completion "PAT-L"

echo "🎉 All PAT models training completed!"
echo "📁 Check model_weights/pat/heads/ for trained models"
echo "📊 Check training_summary.json for results" 