# Claude Code GitHub Actions - Quick Setup Guide

## ✅ What We've Done

1. **Created GitHub Actions Workflow** (`.github/workflows/claude-code.yml`)
   - Uses Claude Opus 4 model for best quality
   - Triggers on @claude mentions
   - Integrated with your make commands (test, lint, type-check)

2. **Created 5 Strategic Issues** (for near-future tasks):
   - #6: SQLite persistence for episodes
   - #7: Enhanced CSV export with configurable columns  
   - #8: BDD integration tests
   - #9: Statistics dashboard
   - #10: Multi-rater collaboration

## 🚀 What Happens Next

1. **While You're Away**:
   - Claude will see the @claude mentions in the issues
   - It will create PRs implementing each feature
   - PRs will be fully tested and linted
   - You'll have PRs ready to review when you return

2. **When You Return**:
   - Review the PRs Claude created
   - Merge the ones that look good
   - Continue with your TDD workflow, now accelerated

## 🔧 Final Setup Steps

You need to add these secrets to your GitHub repository:

```bash
# Option 1: Use GitHub CLI (recommended)
gh secret set ANTHROPIC_API_KEY --body "your-api-key-here"

# Option 2: Via GitHub UI
# Go to Settings → Secrets and variables → Actions
# Add: ANTHROPIC_API_KEY = your-api-key
```

## 📝 How to Use Going Forward

1. **Create an Issue** describing what you need
2. **Tag @claude** in the issue body
3. **Claude creates a PR** automatically
4. **Review and merge** when ready

## 🎯 Strategic Workflow

The issues are designed to be:
- **Near-future** (not immediate - you're still coding current features)
- **Self-contained** (Claude can implement without blocking you)
- **TDD-focused** (all include test requirements)
- **Progressive** (each builds on previous work)

## 🔄 The Loop

```
You code → Create future issues → @claude → PRs ready → Review/merge → Accelerated progress
```

This creates a continuous pipeline where Claude is always working a few steps ahead!