# 🌳 Git Workflow & Branching Strategy

## Branch Structure

```
main (production-ready)
  ↓
staging (pre-production testing) 
  ↓
development (active development)
```

## Branch Purposes

### 🚀 `main` - Production Branch
- **Purpose**: Production-ready, stable releases
- **Protection**: Protected, requires PR reviews
- **Deploy**: Automatic deployment to production
- **Merges**: Only from `staging` via PR

### 🧪 `staging` - Pre-Production Branch  
- **Purpose**: Final testing before production
- **Protection**: Requires PR from `development`
- **Deploy**: Staging environment for QA
- **Merges**: Only from `development` via PR

### 🛠 `development` - Active Development Branch
- **Purpose**: Daily development work
- **Protection**: Base for feature branches
- **Deploy**: Development environment
- **Merges**: Feature branches merge here

## Workflow

### 🔄 Daily Development Flow
```bash
# 1. Start new feature
git checkout development
git pull origin development
git checkout -b feature/your-feature-name

# 2. Work on feature
# ... make changes ...
git add .
git commit -m "feat: implement your feature"

# 3. Push and create PR to development
git push -u origin feature/your-feature-name
# Create PR: feature/your-feature-name → development
```

### 🚀 Release Flow
```bash
# 1. Development → Staging
git checkout staging
git pull origin staging
# Create PR: development → staging

# 2. Test in staging environment
# Run full test suite, manual QA

# 3. Staging → Main (Production)
git checkout main
git pull origin main  
# Create PR: staging → main
```

## Commands Reference

### 🛠 Setup (One-time)
```bash
git clone <repo>
cd big-mood-detector
git checkout development  # Work in development by default
```

### 📊 Status Check
```bash
git branch -a              # Show all branches
git status                 # Current branch status
make quality               # Run full quality checks
```

### 🔄 Sync Branches
```bash
# Sync development with latest changes
git checkout development
git pull origin development

# Sync staging with development
git checkout staging  
git pull origin staging
git merge development
git push origin staging

# Sync main with staging  
git checkout main
git pull origin main
git merge staging
git push origin main
```

### 🧹 Cleanup
```bash
# Delete merged feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## Current Status (2025-01-23)

✅ **Branch Structure Created**
- `main`: Production-ready baseline
- `staging`: Pre-production testing
- `development`: Active development ← **You are here**

✅ **All Synchronized**  
- All branches pushed to origin
- Clean working tree
- Ready for development

## Next Steps

1. **Work in `development`** - All daily work happens here
2. **Create feature branches** - For larger features
3. **PR to development** - Merge features back
4. **Periodic staging releases** - When ready for testing
5. **Production releases** - When staging is validated

---
*Last updated: 2025-01-23 | Current branch: `development`* 