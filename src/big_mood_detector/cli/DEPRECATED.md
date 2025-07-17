# DEPRECATED CLI MODULE

This directory contains deprecated CLI implementations that have been moved to the proper interfaces layer.

## Migration Notice

All CLI functionality has been consolidated into:
`src/big_mood_detector/interfaces/cli/`

The files in this directory are maintained temporarily for backward compatibility but should not be used for new development.

### Deprecated Files:
- `process_data.py` → Use `bmd process` command
- `predict.py` → Use `bmd predict` command

### New Unified CLI:
```bash
# Install the CLI
pip install -e .

# Use the unified commands
bmd process --input export.xml --output features.csv
bmd predict --input export.xml --format json
bmd serve --port 8000
bmd watch /path/to/health/data
```

These deprecated files will be removed in a future version.