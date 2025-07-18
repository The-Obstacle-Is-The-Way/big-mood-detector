#!/usr/bin/env bash
set -euo pipefail

# If no arguments provided, run the API server
if [[ $# -eq 0 ]]; then
    echo "Starting Big Mood Detector API server..."
    exec gunicorn big_mood_detector.interfaces.api.main:app \
        --bind 0.0.0.0:${PORT:-8000} \
        --workers ${WORKERS:-4} \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout ${TIMEOUT:-120} \
        --access-logfile - \
        --error-logfile -
fi

# Check if first argument is a known command
case "$1" in
    mood-detector|python|bash|sh)
        # Execute the command directly
        exec "$@"
        ;;
    process|predict|label|serve|train|watch)
        # These are mood-detector subcommands
        exec mood-detector "$@"
        ;;
    worker)
        # Start the background worker
        echo "Starting background task worker..."
        shift  # Remove 'worker' from args
        exec python -m big_mood_detector.infrastructure.background.worker "$@"
        ;;
    *)
        # Default: try to execute as-is
        exec "$@"
        ;;
esac