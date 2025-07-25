#!/bin/bash
# Wrapper script for Makefile commands in WSL
# Ensures virtual environment is activated

# Activate virtual environment
source .venv-wsl/bin/activate

# Run make with all arguments
make "$@"