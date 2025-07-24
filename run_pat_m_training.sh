#!/bin/bash
#
# PAT-M (Medium) Depression Training Script
# Based on successful PAT-S results
#

set -euo pipefail

# Configuration
BATCH_SIZE=64
DEVICE=mps
CACHE_DIR=data/cache
OUTPUT_DIR=model_weights/pat/pytorch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "PAT-M Depression Training"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo ""

# Ensure we're in the venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activating virtual environment..."
    source "$(dirname "$0")/.venv/bin/activate"
fi

# Check PyTorch MPS
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
assert torch.backends.mps.is_available(), 'MPS not available!'
"

echo ""
echo "=========================================="
echo "Stage 1: Frozen Encoder (Head Warmup)"
echo "=========================================="
echo ""

python3 scripts/train_pat_depression_pytorch.py \
    --epochs 10 \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --model-size medium \
    --unfreeze-layers 0 \
    --head-lr 3e-4 \
    --encoder-lr 0 \
    --weight-decay 1e-2 \
    --grad-clip 1.0 \
    --scheduler cosine \
    --warmup-epochs 2 \
    --early-stopping-patience 5 \
    --no-sampler \
    --output-dir "$OUTPUT_DIR/pat_m_stage1_$TIMESTAMP"

# Get the best model from stage 1
STAGE1_BEST="$OUTPUT_DIR/pat_m_stage1_$TIMESTAMP/best_model.pt"

if [ ! -f "$STAGE1_BEST" ]; then
    echo "ERROR: Stage 1 best model not found at $STAGE1_BEST"
    exit 1
fi

echo ""
echo "=========================================="
echo "Stage 2: Fine-tune Last 2 Blocks"
echo "=========================================="
echo ""

python3 scripts/train_pat_depression_pytorch.py \
    --epochs 20 \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --model-size medium \
    --unfreeze-layers 2 \
    --head-lr 1e-4 \
    --encoder-lr 3e-5 \
    --weight-decay 1e-2 \
    --grad-clip 1.0 \
    --scheduler cosine \
    --warmup-epochs 2 \
    --early-stopping-patience 5 \
    --checkpoint "$STAGE1_BEST" \
    --no-sampler \
    --output-dir "$OUTPUT_DIR/pat_m_stage2_$TIMESTAMP"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Stage 1 results: $OUTPUT_DIR/pat_m_stage1_$TIMESTAMP/"
echo "Stage 2 results: $OUTPUT_DIR/pat_m_stage2_$TIMESTAMP/"
echo ""

# Print final results
RESULT_FILE=$(ls "$OUTPUT_DIR/pat_m_stage2_$TIMESTAMP"/results_medium_*.json 2>/dev/null | head -1)
if [ -f "$RESULT_FILE" ]; then
    echo "Final results:"
    python3 <<PY
import json
with open("$RESULT_FILE") as f:
    results = json.load(f)
    print(f"Best Val AUC: {results['best_val_auc']:.4f}")
    print(f"Test AUC: {results['test_metrics']['test_auc']:.4f}")
    print(f"Test PR-AUC: {results['test_metrics']['test_pr_auc']:.4f}")
PY
fi