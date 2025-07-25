# Root Directory Structure

## Essential Files (Keep in Root)

### Documentation
- `README.md` - Project overview and quick start
- `CLAUDE.md` - AI agent guide (v0.4.0)
- `CHANGELOG.md` - Version history
- `CONTRIBUTING.md` - Contribution guidelines
- `ROADMAP_TO_MVP_V1.0.md` - Current development roadmap
- `LICENSE` - Apache 2.0 license
- `NOTICE` - Legal notices

### Configuration
- `pyproject.toml` - Python project configuration
- `Makefile` - Build and development commands
- `mypy.ini` - Type checking configuration
- `alembic.ini` - Database migration configuration
- `.env.example` - Environment variables template

### Development Tools
- `.gitignore` - Git ignore patterns
- `.gitattributes` - Git attributes
- `.pre-commit-config.yaml` - Pre-commit hooks
- `.python-version` - Python version specification
- `.cursorrules` - Cursor IDE rules

### Docker
- `Dockerfile` - Main Docker image
- `Dockerfile.alpine` - Lightweight Alpine image
- `docker-compose.yml` - Docker compose configuration
- `.dockerignore` - Docker ignore patterns

### Build Tools
- `mkdocs.yml` - Documentation site configuration
- `make-wsl.sh` - WSL-specific make wrapper

## Documentation Organization

```
docs/
├── setup/                  # Setup and installation guides
│   ├── SETUP_GUIDE.md
│   ├── DATA_SETUP_GUIDE.md
│   ├── DOCKER_SETUP_GUIDE.md
│   └── PC_MIGRATION_CHECKLIST.md
├── deployment/             # Deployment documentation
│   ├── DEPLOYMENT_READINESS.md
│   └── DATA_FILES_MANIFEST.md
├── training/              # ML training documentation
│   ├── PAT_CONV_L_ACHIEVEMENT.md
│   ├── TRAINING_SUMMARY.md
│   └── TRAINING_OUTPUT_STRUCTURE.md
└── archive/               # Old documentation
```

## Source Code Organization

```
src/
└── big_mood_detector/     # Main application code
    ├── domain/           # Business logic
    ├── application/      # Use cases
    ├── infrastructure/   # External integrations
    └── interfaces/       # CLI/API
```

## Key Directories

- `model_weights/` - ML model files
- `scripts/` - Utility scripts
- `tests/` - Test suite
- `data/` - Data files (git-ignored)
- `logs/` - Log files (git-ignored)
- `.venv-wsl/` - Virtual environment (git-ignored)