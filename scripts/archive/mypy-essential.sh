#!/bin/bash
# Quick mypy check for essential modules only

source .venv-wsl/bin/activate || source .venv/bin/activate

# Check core modules only (skip slow ones)
mypy --config-file mypy.ini \
    src/big_mood_detector/domain \
    src/big_mood_detector/application \
    src/big_mood_detector/interfaces/cli \
    --no-error-summary \
    --pretty