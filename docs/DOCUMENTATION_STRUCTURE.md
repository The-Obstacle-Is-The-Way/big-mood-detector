# Documentation Structure Guide

**Last Updated**: July 25, 2025  
**Version**: v0.4.1

## Overview

This guide documents the organization and structure of the Big Mood Detector documentation, following professional documentation best practices.

## Directory Structure

```
docs/
├── api/                 # API endpoint documentation
│   ├── clinical.md      # Clinical assessment endpoints
│   ├── features.md      # Feature extraction endpoints
│   ├── health.md        # Health check endpoints
│   ├── labels.md        # Episode labeling endpoints
│   └── predictions.md   # Prediction endpoints
├── architecture/        # System architecture docs
│   └── xml_optimization_solution.md
├── archive/             # Historical documentation
│   ├── 202507/         # July 2025 archive
│   │   ├── planning/   # Old planning docs
│   │   └── ...         # Other archived docs
│   └── pat_experiments/ # PAT experiment records
├── assets/             # Images and media
├── clinical/           # Clinical/medical documentation
│   ├── CLINICAL_DOSSIER.md
│   ├── CLINICAL_REQUIREMENTS_DOCUMENT.md
│   └── README.md       # Medical disclaimer
├── deployment/         # Deployment guides
│   ├── DATA_FILES_MANIFEST.md
│   └── DEPLOYMENT_READINESS.md
├── developer/          # Developer documentation
│   ├── ARCHITECTURE_OVERVIEW.md
│   ├── DUAL_PIPELINE_ARCHITECTURE.md
│   ├── GIT_WORKFLOW.md
│   ├── MODEL_WEIGHT_ARCHITECTURE.md
│   └── ...
├── javascripts/        # MkDocs JS assets
├── literature/         # Research papers
│   └── converted_markdown/
├── performance/        # Performance documentation
│   └── OPTIMIZATION_TRACKING.md
├── planning/           # Active planning docs
│   └── docs_inventory.json
├── refactoring/        # Refactoring history
│   ├── CHANGELOG.md
│   └── DESIGN_PATTERNS_CATALOG.md
├── setup/              # Installation guides
│   ├── DATA_SETUP_GUIDE.md
│   ├── DOCKER_SETUP_GUIDE.md
│   ├── PC_MIGRATION_CHECKLIST.md
│   └── SETUP_GUIDE.md
├── stylesheets/        # MkDocs CSS
├── training/           # ML training documentation
│   ├── PAT_DEPRESSION_TRAINING.md
│   ├── PAT_L_PAPER_REPLICATION.md
│   └── ...
└── user/               # User documentation
    ├── APPLICATION_WORKFLOW.md
    ├── QUICK_START_GUIDE.md
    └── README.md

```

## Documentation Categories

### 1. User Documentation (`/user/`)
- **Purpose**: Help end users get started and use the application
- **Audience**: Non-technical users, clinicians, researchers using the tool
- **Content**: Quick start guides, workflows, CLI reference

### 2. Developer Documentation (`/developer/`)
- **Purpose**: Technical documentation for contributors and developers
- **Audience**: Software engineers, ML engineers, open source contributors
- **Content**: Architecture, APIs, development setup, code organization

### 3. Clinical Documentation (`/clinical/`)
- **Purpose**: Medical and clinical aspects of the system
- **Audience**: Healthcare professionals, researchers, regulatory reviewers
- **Content**: Clinical validation, medical disclaimers, requirements

### 4. Training Documentation (`/training/`)
- **Purpose**: Document ML model training processes and results
- **Audience**: ML researchers, data scientists
- **Content**: Training logs, experiments, results, methodologies

### 5. API Documentation (`/api/`)
- **Purpose**: REST API endpoint reference
- **Audience**: Frontend developers, integration engineers
- **Content**: Endpoint specifications, request/response formats

### 6. Setup Documentation (`/setup/`)
- **Purpose**: Installation and configuration guides
- **Audience**: System administrators, developers
- **Content**: Platform-specific setup, Docker, data preparation

## Best Practices

### 1. File Naming
- Use descriptive UPPERCASE names for major documents (e.g., `README.md`, `SETUP_GUIDE.md`)
- Use lowercase with underscores for technical docs (e.g., `xml_optimization_solution.md`)
- Include version or date in archived files

### 2. Content Organization
- Start each document with metadata (Last Updated, Version)
- Use clear hierarchical headings
- Include a table of contents for long documents
- Cross-reference related documents

### 3. Maintenance
- Regular audits using `scripts/inventory_docs.py`
- Archive outdated content to `/archive/YYYYMM/`
- Update version references during releases
- Keep `docs_inventory.json` current

### 4. MkDocs Integration
- Configure `mkdocs.yml` to match directory structure
- Use material theme for professional appearance
- Enable search and navigation features
- Deploy to GitHub Pages for public access

## Document Lifecycle

### Active Documents
- Keep in appropriate category directory
- Update version references with each release
- Review quarterly for accuracy

### Archiving Process
1. Create timestamped directory in `/archive/YYYYMM/`
2. Move outdated documents preserving directory structure
3. Update any references to archived documents
4. Document reason for archival in commit message

### Deprecation
- Mark deprecated content clearly at the top of documents
- Provide links to updated information
- Archive after one release cycle

## Quality Standards

### Required Elements
- [ ] Clear title and purpose
- [ ] Last updated date
- [ ] Version reference (if applicable)
- [ ] Target audience identification
- [ ] Cross-references to related docs

### Writing Style
- Use clear, concise language
- Define technical terms on first use
- Include examples for complex concepts
- Use diagrams where helpful

## Regular Maintenance Tasks

### Weekly
- Review and merge documentation PRs
- Update user guides based on feedback

### Monthly
- Run documentation audit script
- Archive outdated content
- Update cross-references

### Quarterly
- Comprehensive documentation review
- Update MkDocs configuration
- Refresh deployment guides

## Contact

For documentation questions or suggestions:
- Open an issue with `docs:` prefix
- Tag @documentation-maintainers
- See CONTRIBUTING.md for guidelines