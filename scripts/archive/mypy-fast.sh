#!/bin/bash
# Fast mypy checking using daemon mode

# Activate virtual environment if needed
if [ -f ".venv-wsl/bin/activate" ]; then
    source .venv-wsl/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check if dmypy daemon is running
if ! dmypy status > /dev/null 2>&1; then
    echo "Starting mypy daemon..."
    dmypy start -- --config-file mypy.ini
fi

# Run type checking using daemon (much faster)
echo "Running fast type check..."
dmypy run -- src/big_mood_detector