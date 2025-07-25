# Documentation Structure

## Organization

```
docs/
├── README.md              # Documentation index
├── license.md             # License information
│
├── api/                   # API documentation
├── architecture/          # System architecture diagrams
├── assets/                # Images and media
├── clinical/              # Clinical validation & requirements
├── deployment/            # Deployment guides
├── developer/             # Developer documentation
├── literature/            # Research papers
├── performance/           # Performance benchmarks
├── planning/              # Project planning docs
├── refactoring/           # Refactoring guides
├── setup/                 # Installation & setup
├── training/              # ML model training docs
├── user/                  # End-user guides
├── user-guide/            # Application workflows
│
├── javascripts/           # MkDocs JavaScript
├── stylesheets/           # MkDocs CSS
│
└── archive/               # Old/outdated docs
```

## Key Documents by Audience

### For Users
- `user/QUICK_START_GUIDE.md` - Get started in 5 minutes
- `user-guide/APPLICATION_WORKFLOW.md` - How to use the app

### For Developers
- `developer/ARCHITECTURE_OVERVIEW.md` - System design
- `setup/SETUP_GUIDE.md` - Development environment
- `api/` - REST API documentation

### For ML Engineers
- `training/PAT_CONV_L_ACHIEVEMENT.md` - Best model details (0.5929 AUC)
- `training/TRAINING_SUMMARY.md` - Current models status
- `training/TRAINING_OUTPUT_STRUCTURE.md` - File organization

### For Researchers
- `clinical/` - Clinical requirements and validation
- `literature/` - Research papers
- `performance/` - Benchmarks and optimization

## Recent Additions (July 2025)

1. **PAT-Conv-L Documentation** - Achieved 0.5929 AUC
2. **Training Infrastructure** - Clean organization
3. **Deployment Readiness** - Production checklist
4. **Setup Guides** - WSL, Docker, data setup

## Archive

Old documentation moved to `archive/` includes:
- Work summaries
- Triage plans
- Old status reports
- Superseded training docs