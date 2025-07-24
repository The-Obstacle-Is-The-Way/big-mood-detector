#!/bin/bash
# Start PAT-L Linear Probe Training

# Kill any existing session
tmux kill-session -t pat_l_lp 2>/dev/null || true

# Create logs directory
mkdir -p logs

# Start new training session
echo "Starting PAT-L Linear Probe training..."
echo "Learning rate: 1e-4 (50x lower than before)"
echo "Encoder: Completely frozen"
echo "Epochs: 150 with patience 50"
echo "This will take 4-6 hours"

tmux new-session -d -s pat_l_lp bash -c "
cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/big-mood-detector && \
source .venv/bin/activate && \
python scripts/train_pat_l_linear_probe.py \
  --device mps \
  --model-size large \
  --epochs 150 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  2>&1 | tee logs/pat_l_linear_probe_$(date +%Y%m%d_%H%M%S).log
"

echo "Training started in tmux session 'pat_l_lp'"
echo "To monitor: tmux attach -t pat_l_lp"
echo "To detach: Ctrl-B then D"