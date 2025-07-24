# Documentation Overhaul Plan

## 🎯 Goal
Create a clean, professional documentation structure with ONE authoritative document per section that clearly links back to the root README.

## 📋 Current State Analysis

### Problems:
1. Multiple conflicting READMEs (root, docs/, docs/index.md)
2. Outdated information scattered across files
3. No clear navigation structure
4. Mixed audience content (users, devs, clinicians)
5. Old v0.2.0 references when we're at v0.4.0

### Strengths:
1. Rich research literature collection
2. Detailed technical documentation exists
3. Good clinical validation docs
4. Strong API documentation

## 🏗️ Proposed Structure

```
big-mood-detector/
├── README.md                    # Main entry point - polished, clinical focus
├── CHANGELOG.md                 # Version history (already good)
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # Apache 2.0
├── CLAUDE.md                    # AI agent guide (already good)
└── docs/
    ├── README.md               # Simple index pointing to sections
    ├── quickstart.md           # 5-minute guide for new users
    ├── clinical.md             # Clinical validation & research
    ├── technical.md            # Architecture & implementation
    ├── api.md                  # API reference
    └── [folders]/              # Keep existing structure for deep dives
```

## 📝 One Document Per Section Plan

### 1. Root README.md
**Purpose**: Professional GitHub landing page with clinical relevance
**Content**:
- Clear value proposition with clinical focus
- Real performance metrics (honest about 0.56 AUC)
- Quick start with `big-mood` CLI commands
- Links to key docs
- Medical disclaimer
- Beautiful but not corny

### 2. docs/quickstart.md
**Purpose**: Get users running in 5 minutes
**Content**:
- Installation (`pip install big-mood-detector`)
- CLI commands with examples
- What to expect from predictions
- Link to clinical.md for interpretation

### 3. docs/clinical.md
**Purpose**: Clinical validation and research foundation
**Content**:
- Research papers summary
- Performance metrics with context
- DSM-5 criteria
- Limitations and caveats
- Links to full papers in literature/

### 4. docs/technical.md
**Purpose**: Technical overview for developers
**Content**:
- Architecture diagram
- Key components (Temporal Ensemble, PAT, XGBoost)
- Performance benchmarks
- Links to detailed developer docs

### 5. docs/api.md
**Purpose**: API reference for integrations
**Content**:
- REST endpoints
- Request/response examples
- Authentication
- Links to OpenAPI spec

## 🧹 Cleanup Actions

### Archive These:
- docs/IMPORTANT_PLEASE_READ.md → Merge key points into README
- docs/index.md → Replace with simple navigation
- docs/CDS_ROUGH_DRAFT_NOT_IMPLEMENTED_YET.md → Archive
- docs/MODEL_LABELING_REQUIREMENTS.md → Archive
- docs/PERFORMANCE_*.md → Archive (issues resolved)

### Keep But Don't Feature:
- literature/ → Reference from clinical.md
- developer/ → Reference from technical.md
- models/ → Reference from technical.md
- archive/ → Historical reference only

## 🔗 Linking Strategy

Each section doc should:
1. Start with its purpose
2. Provide essential information
3. Link to root README for overview
4. Link to detailed docs for deep dives
5. Use consistent formatting

Example:
```markdown
# Clinical Validation

> For overview, see [Big Mood Detector README](../README.md)

This document covers the clinical research foundation...

## Quick Links
- [Back to Overview](../README.md)
- [Technical Details](technical.md)
- [API Reference](api.md)

## Content...

---
*For detailed research papers, see [literature/](literature/)*
```

## 📅 Implementation Steps

1. **Create new section docs** (quickstart, clinical, technical, api)
2. **Update root README** with proper CLI, clinical focus, clean structure
3. **Archive outdated docs**
4. **Update docs/README.md** as simple index
5. **Test all links and commands**
6. **Update references to v0.4.0**

## ✅ Success Criteria

- [ ] One authoritative document per major topic
- [ ] All docs link cleanly to root README
- [ ] No conflicting information
- [ ] Clear navigation path
- [ ] All CLI commands documented and working
- [ ] Clinical relevance emphasized
- [ ] Professional but approachable tone