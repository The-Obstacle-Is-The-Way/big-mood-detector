# Big Mood Detector PAT-L Training Setup Guide

## Quick Start (Windows 11 + WSL2 + RTX 4090)

### Prerequisites
- Windows 11 with WSL2 installed
- NVIDIA RTX 4090 with latest drivers (550+)
- WSL2 GPU support enabled
- Ubuntu 22.04 LTS in WSL2

### Step 1: Clean Environment Setup

```bash
# In WSL2 Ubuntu terminal
cd /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector

# Remove old virtual environments
rm -rf .venv .venv-wsl .venv-*

# Install Python 3.12
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Create new virtual environment
python3.12 -m venv .venv-wsl
source .venv-wsl/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Dependencies with Correct Versions

```bash
# CRITICAL: Install numpy<2.0 first to avoid conflicts
pip install 'numpy<2.0'

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ML dependencies
pip install transformers h5py scikit-learn pandas matplotlib tqdm

# Install project in editable mode
pip install -e ".[dev,ml,monitoring]"
```

### Step 3: Verify CUDA Setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

Expected output:
```
PyTorch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

### Step 4: Download Pretrained Weights (REQUIRED!)

The PAT models require pretrained weights to function. You must place them in the correct directory:

```bash
# Create the weights directory
mkdir -p model_weights/pat/pretrained/

# Download or copy the pretrained weights here:
# - PAT-L_29k_weights.h5 (7.7MB)
# - PAT-M_29k_weights.h5 (3.9MB)  
# - PAT-S_29k_weights.h5 (1.1MB)

# If you have them elsewhere, copy them:
# cp /path/to/weights/PAT-*.h5 model_weights/pat/pretrained/
```

⚠️ **IMPORTANT**: Training will fail without these weights!

### Step 5: Run PAT-L Training

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:$PYTHONPATH"

# Run training
python scripts/pat_training/train_pat_l_run_now.py
```

### Step 6: Run Training in Background with tmux

For long training sessions, use tmux to keep training running even if your connection drops:

```bash
# Install tmux if needed
sudo apt install tmux

# Create a new tmux session named 'pat-training'
tmux new -s pat-training

# Inside tmux, activate environment and start training
source .venv-wsl/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:$PYTHONPATH"
python scripts/pat_training/train_pat_l_run_now.py

# Detach from tmux: Press Ctrl+B, then D

# To reattach later:
tmux attach -t pat-training

# To list sessions:
tmux ls

# To kill session when done:
tmux kill-session -t pat-training
```

## Common Issues and Solutions

### Issue 1: "No module named 'big_mood_detector'"
**Solution**: Make sure PYTHONPATH is set and you installed with `pip install -e .`

### Issue 2: Network timeouts in WSL
**Solution**: Add to `/etc/wsl.conf`:
```ini
[network]
generateResolvConf = false
```
Then in `/etc/resolv.conf`:
```
nameserver 8.8.8.8
nameserver 8.8.4.4
```

### Issue 3: CUDA not detected
**Solution**: 
1. Update Windows NVIDIA drivers to 550+
2. In Windows PowerShell (admin): `wsl --update`
3. Verify with `nvidia-smi` in WSL

### Issue 4: NumPy version conflicts
**Solution**: Always install `numpy<2.0` FIRST before any other packages

### Issue 5: Import errors with transformers
**Solution**: Uninstall and reinstall in correct order:
```bash
pip uninstall -y transformers torch numpy
pip install 'numpy<2.0'
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers
```

## Background Training

To run training in background with monitoring:

```bash
# Start training with output logged
nohup python scripts/pat_training/train_pat_l_run_now.py > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Project Structure

```
big-mood-detector/
├── .venv-wsl/           # WSL virtual environment
├── scripts/
│   └── pat_training/
│       └── train_pat_l_run_now.py  # Training script
├── data/
│   └── cache/
│       └── nhanes_pat_data_subsetNone.npz  # Cached training data
└── src/
    └── big_mood_detector/  # Main package
```

## Activation Script

Create `activate_training.sh`:
```bash
#!/bin/bash
source /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/.venv-wsl/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/big-mood-detector/src:$PYTHONPATH"
echo "✅ PAT-L training environment activated!"
python -c "import torch; print(f'CUDA ready: {torch.cuda.is_available()}')"
```

Then: `chmod +x activate_training.sh && source activate_training.sh`