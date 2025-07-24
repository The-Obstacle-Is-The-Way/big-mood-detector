# Documentation Ownership

Last updated: 2025-07-24

## Ownership Matrix

| Path | Primary Owner | Backup | Review Required |
|------|--------------|--------|-----------------|
| `/docs/user/` | @The-Obstacle-Is-The-Way | @claude | User experience changes |
| `/docs/clinical/` | @The-Obstacle-Is-The-Way | @clinical-lead | Clinical accuracy |
| `/docs/developer/` | @The-Obstacle-Is-The-Way | @claude | Architecture changes |
| `/docs/api/` | @The-Obstacle-Is-The-Way | @claude | API contract changes |
| `/docs/literature/` | @research-lead | @The-Obstacle-Is-The-Way | New papers only |
| `/docs/models/` | @ml-lead | @The-Obstacle-Is-The-Way | Model updates |
| `README.md` | @The-Obstacle-Is-The-Way | @product | Major changes |

## Responsibilities

### Primary Owner
- Keep documentation accurate and up-to-date
- Review PRs affecting their area
- Ensure consistency with code changes
- Archive outdated content

### Backup Owner
- Review when primary is unavailable
- Assist with major updates
- Knowledge transfer

## Update Process

1. **Code changes** → Update docs in same PR
2. **Breaking changes** → Flag in PR description
3. **New features** → Add user documentation
4. **Bug fixes** → Update if documented behavior changes

## Quality Standards

- ✅ All code examples must work
- ✅ Version numbers must be current
- ✅ Links must not be broken
- ✅ No TODO/FIXME in published docs
- ✅ Clear "Last updated" dates

## Archival Policy

Documents are archived when:
- Information is >1 year old and unused
- Feature is deprecated/removed
- Superseded by newer documentation

Archive location: `/docs/archive/YYYYMM/`