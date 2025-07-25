# Documentation Triage Plan - v0.4.0

Based on inventory analysis of 160 docs. Execution date: 2025-07-24

## Summary
- 🔴 **DELETE**: 5 files (redundant)
- 🟡 **ARCHIVE**: 78 files (outdated/draft)
- ⚠️ **UPDATE**: 12 files (minor fixes needed)
- ✅ **KEEP**: 65 files (current and accurate)

## DELETE These (Redundant)

```bash
# These are duplicates or offer no value
rm docs/IMPORTANT_PLEASE_READ.md  # Outdated v0.2 info
rm docs/CDS_ROUGH_DRAFT_NOT_IMPLEMENTED_YET.md  # Never implemented
rm docs/MODEL_LABELING_REQUIREMENTS.md  # Superseded by labeling CLI
rm docs/PERFORMANCE_FIX_ISSUE_29.md  # Issue resolved
rm docs/PERFORMANCE_INVESTIGATION.md  # Investigation complete
```

## ARCHIVE These (Historical Value)

Create `docs/archive/202507/` and move:

### Old Version Docs (v0.2/v0.3)
- All files with OLD-v0.2 or OLD-v0.3 tags
- archive/V0.2.0_RELEASE_NOTES.md
- archive/V0.3.0_SAFE_MIGRATION_PLAN.md
- archive/HONEST_STATE_OF_V0.2.0.md

### Outdated Architecture
- api/python/*.md → These describe code structure, not API
- architecture/bulletproof_pipeline_summary.md → Old pipeline design
- models/ensemble/CURRENT_ENSEMBLE_EXPLANATION.md → Describes fake ensemble

### Planning/Draft Docs
- archive/phase*/ → All phase planning docs
- archive/*_PLAN.md → All planning docs
- archive/ROADMAP*.md → Old roadmaps

## UPDATE These (Minor Fixes)

| File | Required Changes |
|------|-----------------|
| docs/README.md | Remove ensemble reference |
| api/predictions.md | Remove TensorFlow, update to PyTorch |
| developer/API_REFERENCE.md | Remove TF references, update ensemble description |
| developer/ARCHITECTURE_OVERVIEW.md | Update ensemble → temporal separation |
| developer/model_integration_guide.md | TF → PyTorch, update PAT info |
| clinical/CLINICAL_DOSSIER.md | Remove ensemble references |
| user/QUICK_START_GUIDE.md | Update CLI examples to use `big-mood` |
| user/ADVANCED_USAGE.md | Update ensemble → temporal approach |

## KEEP These (Already Good)

### Core Documentation
- ✅ clinical/README.md
- ✅ developer/README.md  
- ✅ user/README.md
- ✅ Root README.md

### Technical Docs
- ✅ developer/DEPLOYMENT_GUIDE.md
- ✅ developer/DUAL_PIPELINE_ARCHITECTURE.md
- ✅ developer/GIT_WORKFLOW.md
- ✅ developer/MODEL_WEIGHT_ARCHITECTURE.md
- ✅ developer/SECURITY.md

### API Docs
- ✅ api/clinical.md
- ✅ api/features.md
- ✅ api/health.md
- ✅ api/labels.md

### Literature (All 14 research papers)
- ✅ All files in literature/converted_markdown/

## Execution Steps

1. **Create archive directory**
   ```bash
   mkdir -p docs/archive/202507
   ```

2. **Run deletion script**
   ```bash
   # Delete redundant files
   rm docs/IMPORTANT_PLEASE_READ.md
   rm docs/CDS_ROUGH_DRAFT_NOT_IMPLEMENTED_YET.md
   rm docs/MODEL_LABELING_REQUIREMENTS.md
   rm docs/PERFORMANCE_FIX_ISSUE_29.md
   rm docs/PERFORMANCE_INVESTIGATION.md
   ```

3. **Archive with banner**
   ```bash
   # Add archive banner to files before moving
   for f in <files-to-archive>; do
     echo -e "> **Archived 2025-07-24** – No longer relevant as of v0.4.0\n\n$(cat $f)" > $f
     mv $f docs/archive/202507/
   done
   ```

4. **Update remaining docs**
   - Replace "ensemble" → "temporal separation"
   - Replace "TensorFlow" → "PyTorch"
   - Update CLI examples to use `big-mood`
   - Update version references to v0.4.0

## Success Metrics

After cleanup:
- No files mention v0.2 or v0.3 (except CHANGELOG)
- No files mention TensorFlow (except research papers)
- All "ensemble" references updated to "temporal"
- All CLI examples use `big-mood` command
- Archive folder clearly dated