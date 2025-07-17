# Claude Code GitHub Actions Setup Dossier

## Executive Summary

Claude Code is an agentic coding tool from Anthropic that can be integrated with GitHub to enable autonomous development workflows. This dossier provides comprehensive setup instructions for enabling Claude Code to automatically create PRs from issues, perform code reviews, implement features, and handle various development tasks through GitHub Actions.

**Key Point**: Setup is **per-repository**, not global. Each repository requires its own configuration.

## System Overview

### What Claude Code Can Do Autonomously

1. **Automated PR Creation**: Transform issues into working pull requests
2. **Code Implementation**: Build features based on natural language descriptions
3. **Bug Fixes**: Identify and fix errors with ready-to-merge PRs
4. **Code Reviews**: Analyze PRs for bugs, style, and standards compliance
5. **Documentation Generation**: Auto-generate docs for merged PRs
6. **Refactoring**: Improve code quality and maintainability
7. **CI/CD Integration**: Access workflow runs, job logs, and test results

### What Claude Code Cannot Do

- Submit formal GitHub PR reviews (only comments)
- Approve PRs (security restriction)
- Post multiple comments (updates single comment)
- Execute commands outside repository context
- Access private resources without authentication

## Prerequisites

1. **Anthropic Account**:
   - Active account at [console.anthropic.com](https://console.anthropic.com)
   - API key with active billing
   - Pro/Max plans recommended for higher usage

2. **GitHub Repository**:
   - Admin access to the repository
   - Ability to add secrets and install GitHub Apps

3. **Claude Code CLI** (optional but recommended):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

## Installation Methods

### Method 1: Automated Setup (Recommended)

1. **Navigate to your repository**:
   ```bash
   cd your-repository
   ```

2. **Launch Claude Code**:
   ```bash
   claude
   ```

3. **Run the installer**:
   ```
   /install-github-app
   ```

4. **If you get permission errors**:
   ```bash
   gh auth refresh -h github.com -s workflow
   ```
   Then retry `/install-github-app`

This automated method will:
- Install the Claude GitHub App
- Configure required secrets
- Create the workflow file
- Set up proper permissions

### Method 2: Manual Setup

#### Step 1: Install Claude GitHub App

1. Visit [https://github.com/apps/claude](https://github.com/apps/claude)
2. Click "Install" and select your repository
3. Grant permissions for:
   - Contents (read/write)
   - Issues (read/write)
   - Pull requests (read/write)
   - Actions (read) - for CI/CD integration

#### Step 2: Add Repository Secrets

Navigate to `https://github.com/YOUR-ORG/YOUR-REPO/settings/secrets/actions`

Add the following secrets:

1. **ANTHROPIC_API_KEY**: Your API key from Anthropic console
2. **APP_ID**: GitHub App ID (obtained after app installation)
3. **APP_PRIVATE_KEY**: GitHub App private key content

#### Step 3: Create Workflow File

Create `.github/workflows/claude.yml`:

```yaml
name: Claude PR Action

permissions:
  contents: write
  pull-requests: write
  issues: write
  id-token: write
  actions: read  # Required for CI/CD integration

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  issues:
    types: [opened, assigned]

jobs:
  claude-pr:
    # Only run when @claude is mentioned
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'issues' && contains(github.event.issue.body, '@claude'))
    
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        
      - name: Generate GitHub App token
        id: app-token
        uses: actions/create-github-app-token@v2
        with:
          app-id: ${{ secrets.APP_ID }}
          private-key: ${{ secrets.APP_PRIVATE_KEY }}
          
      - name: Run Claude Code Action
        uses: anthropics/claude-code-action@latest
        with:
          github_token: ${{ steps.app-token.outputs.token }}
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          model: "claude-opus-4-20250514"  # Latest Claude 4 model
          allowed_tools: "Bash(git:*),View,GlobTool,GrepTool,BatchTool,FileEditTool,CreateFileTool"
```

## Configuration

### 1. Project Guidelines (CLAUDE.md)

Create a `CLAUDE.md` file in your repository root:

```markdown
# Claude Code Guidelines

## Code Standards
- Use TypeScript for all new files
- Follow ESLint and Prettier configurations
- Write comprehensive tests for new features
- Use async/await for asynchronous operations
- Avoid console.log in production code

## Architecture
- Follow existing folder structure
- Implement proper error handling
- Add JSDoc comments for public APIs
- Keep functions small and focused

## Git Conventions
- Use conventional commits (feat:, fix:, docs:, etc.)
- Reference issue numbers in commits
- Keep PRs focused on single features
- Write descriptive PR descriptions

## Testing
- Maintain 80% code coverage
- Write unit tests for all new functions
- Include integration tests for APIs
- Run tests before creating PRs
```

### 2. Custom Commands

Create custom commands in `.claude/commands/`:

**Example**: `.claude/commands/fix-issue.md`
```markdown
Find and fix GitHub issue #$ARGUMENTS. Follow these steps:

1. Use `gh issue view $ARGUMENTS` to understand the issue
2. Locate relevant code in the codebase
3. Implement a solution addressing the root cause
4. Write appropriate tests
5. Ensure all tests pass
6. Create a descriptive commit
7. Push changes and create a PR
8. Link the PR to the original issue

Use conventional commit format and reference the issue number.
```

### 3. Advanced Workflows

#### Auto-Documentation for Merged PRs

Create `.github/workflows/auto-doc.yml`:

```yaml
name: Auto-generate PR Documentation

on:
  pull_request:
    types: [closed]
    branches:
      - main

jobs:
  generate-documentation:
    if: |
      github.event.pull_request.merged == true &&
      github.event.pull_request.user.type != 'Bot' &&
      !startsWith(github.event.pull_request.title, 'docs:')
    
    runs-on: ubuntu-latest
    
    permissions:
      contents: write
      pull-requests: write
      
    steps:
      - uses: textcortex/claude-code-pr-autodoc-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          output_directory: "docs/prs"
          min_lines_changed: 10
          commit_tag: "docs"
```

#### Custom Claude Code Base Action

For complex automation workflows:

```yaml
- name: Custom Claude Task
  uses: anthropics/claude-code-base-action@beta
  with:
    prompt: |
      Analyze the codebase and:
      1. Find all TODO comments
      2. Create issues for each TODO
      3. Prioritize based on code complexity
      4. Generate a summary report
    allowed_tools: "Bash(git:*),View,GlobTool,GrepTool,BatchTool"
    anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    max_turns: 10
```

## Usage Examples

### Basic Usage

1. **Implement a Feature from an Issue**:
   ```
   @claude implement the user authentication feature described in this issue
   ```

2. **Fix a Bug**:
   ```
   @claude fix the TypeError in the payment processing module
   ```

3. **Code Review**:
   ```
   @claude review this PR for security vulnerabilities and performance issues
   ```

4. **Refactoring**:
   ```
   @claude refactor the data fetching logic in src/api/users.js to use modern patterns
   ```

### Advanced Usage

1. **Multi-Step Implementation**:
   ```
   @claude Based on this issue:
   1. Create the database schema
   2. Implement the API endpoints
   3. Add comprehensive tests
   4. Update the documentation
   5. Create a PR with all changes
   ```

2. **Custom Command Usage**:
   ```
   @claude /project:fix-issue 123
   ```

## MCP (Model Context Protocol) Integration

Create `.mcp.json` for external tool integration:

```json
{
  "mcpServers": {
    "puppeteer": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-puppeteer"],
      "env": {
        "PUPPETEER_HEADLESS": "true"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-memory"]
    }
  }
}
```

## Performance Optimization

### Using Depot Runners (Optional)

For faster and cheaper execution, replace `ubuntu-latest` with:

```yaml
runs-on: depot-ubuntu-latest
```

Benefits:
- 50% cost reduction
- Faster CPUs and RAM
- Better network performance

## Security Best Practices

1. **API Key Management**:
   - Never commit API keys
   - Use GitHub Secrets exclusively
   - Rotate keys regularly

2. **Permissions**:
   - Grant minimum required permissions
   - Review app permissions regularly
   - Use OIDC for cloud providers when possible

3. **Workflow Security**:
   - Limit workflow triggers
   - Validate PR sources
   - Use environment protection rules

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   ```bash
   gh auth refresh -h github.com -s workflow
   ```

2. **App Installation Failures**:
   - Ensure admin access to repository
   - Check GitHub App permissions
   - Verify secret configuration

3. **Claude Not Responding**:
   - Check API key validity
   - Verify billing status
   - Review GitHub Actions logs

### Debug Mode

For detailed debugging:
```bash
claude --mcp-debug
```

## Cost Considerations

- **API Usage**: Monitor at console.anthropic.com
- **GitHub Actions**: Standard GitHub pricing applies
- **Optimization**: Use rate limiting for public repos

## Integration with Development Workflow

### 1. Issue-to-PR Workflow
1. Create detailed issue with requirements
2. Tag `@claude` to implement
3. Claude creates PR with implementation
4. Review and merge

### 2. Continuous Improvement
1. Claude monitors merged PRs
2. Auto-generates documentation
3. Updates knowledge base
4. Improves future implementations

## Summary for Your AI Agent

**To enable Claude Code GitHub Actions in your repository:**

1. **Install**: Run `/install-github-app` in Claude Code CLI or manually install GitHub App
2. **Configure**: Add ANTHROPIC_API_KEY, APP_ID, and APP_PRIVATE_KEY as secrets
3. **Deploy**: Create `.github/workflows/claude.yml` with provided configuration
4. **Customize**: Add CLAUDE.md for project guidelines
5. **Use**: Tag `@claude` in issues/PRs to trigger autonomous actions

**Key Points**:
- Setup is per-repository, not global
- Claude can create PRs, fix bugs, and implement features autonomously
- Respects project guidelines in CLAUDE.md
- Runs securely on GitHub Actions infrastructure
- Integrates with existing CI/CD pipelines

**Next Steps**:
1. Complete installation in your repository
2. Test with simple issue implementation
3. Gradually expand usage to complex tasks
4. Monitor and optimize based on results