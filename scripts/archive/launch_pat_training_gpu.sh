#!/bin/bash
# Launch PAT-L training with GPU support and monitoring

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 PAT-L GPU Training Launcher${NC}"
echo "================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: Must run from project root directory${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv-wsl" ]; then
    echo -e "${RED}❌ Error: Virtual environment .venv-wsl not found${NC}"
    echo "Run: python3.12 -m venv .venv-wsl"
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source .venv-wsl/bin/activate

# Set environment variables
echo -e "${YELLOW}🔧 Setting environment variables...${NC}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}/src:$PYTHONPATH"
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging

# Check CUDA availability
echo -e "${YELLOW}🔍 Checking CUDA availability...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ CUDA check failed. Ensure NVIDIA drivers are installed.${NC}"
    exit 1
fi

# Check if weights exist
echo -e "${YELLOW}📂 Checking pretrained weights...${NC}"
if [ ! -f "model_weights/pat/pretrained/PAT-L_29k_weights.h5" ]; then
    echo -e "${RED}❌ Error: Pretrained weights not found${NC}"
    echo "Expected location: model_weights/pat/pretrained/PAT-L_29k_weights.h5"
    echo ""
    echo "If you have weights elsewhere, copy them:"
    echo "  mkdir -p model_weights/pat/pretrained/"
    echo "  cp /path/to/PAT-*.h5 model_weights/pat/pretrained/"
    exit 1
fi

# Kill any existing training sessions
echo -e "${YELLOW}🔄 Checking for existing training sessions...${NC}"
tmux list-sessions 2>/dev/null | grep pat-training && {
    echo "Found existing pat-training session. Killing it..."
    tmux kill-session -t pat-training
}

# Create log directory
mkdir -p logs/pat_training
LOG_FILE="logs/pat_training/pat_l_$(date +%Y%m%d_%H%M%S).log"

# Launch training in tmux
echo -e "${GREEN}🎯 Launching PAT-L training in tmux session 'pat-training'${NC}"
echo -e "Log file: $LOG_FILE"
echo ""

tmux new-session -d -s pat-training -c "$PWD" "
    source .venv-wsl/bin/activate
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH='${PWD}/src:\$PYTHONPATH'
    export TF_CPP_MIN_LOG_LEVEL=2
    
    echo '🔥 Starting PAT-L training with GPU support...'
    python scripts/pat_training/train_pat_l_run_now.py 2>&1 | tee '$LOG_FILE'
    
    echo ''
    echo '✅ Training completed!'
    echo 'Press any key to exit...'
    read -n 1
"

# Show instructions
echo -e "${GREEN}✅ Training launched successfully!${NC}"
echo ""
echo "📊 Monitor training:"
echo "  tmux attach -t pat-training    # Watch training progress"
echo "  tail -f $LOG_FILE              # View logs in another terminal"
echo "  watch -n 1 nvidia-smi          # Monitor GPU usage"
echo ""
echo "🔧 Tmux commands:"
echo "  Ctrl+B, D                      # Detach from session"
echo "  tmux ls                        # List sessions"
echo "  tmux kill-session -t pat-training  # Stop training"
echo ""
echo -e "${YELLOW}⏳ Training will take several hours. Check GPU usage to confirm it's running.${NC}"