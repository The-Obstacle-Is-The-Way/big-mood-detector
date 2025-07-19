#!/usr/bin/env bash
set -euo pipefail

# Ensure data directory structure exists with proper permissions
# This handles the case where /data is mounted as a volume
if [ -w "${DATA_DIR:-/data}" ]; then
    mkdir -p "${DATA_DIR:-/data}/output" "${DATA_DIR:-/data}/uploads" "${DATA_DIR:-/data}/temp" || true
    # Try to set permissions if we can (might fail in some environments)
    chmod -R 755 "${DATA_DIR:-/data}" 2>/dev/null || true
fi

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
    python|bash|sh)
        # Execute the command directly
        exec "$@"
        ;;
    process|predict|label|serve|train|watch)
        # These are big-mood-detector subcommands - use Python module
        exec python -m big_mood_detector.main "$@"
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