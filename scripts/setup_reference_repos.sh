#!/bin/bash
# Setup reference repositories for CLI best practices

echo "🚀 Setting up reference repositories for CLI inspiration..."

# Create reference_repos directory
mkdir -p reference_repos
cd reference_repos

# Awesome CLI repos to learn from:

echo "📦 1. Cloning Typer - Modern Python CLI framework by FastAPI creator"
git clone --depth 1 https://github.com/tiangolo/typer.git

echo "📦 2. Cloning Rich - Beautiful terminal formatting"
git clone --depth 1 https://github.com/Textualize/rich.git

echo "📦 3. Cloning Click examples - The foundation we're using"
git clone --depth 1 https://github.com/pallets/click.git

echo "📦 4. Cloning httpie - Excellent CLI UX example"
git clone --depth 1 https://github.com/httpie/httpie.git

echo "📦 5. Cloning Poetry - Great example of complex CLI with subcommands"
git clone --depth 1 https://github.com/python-poetry/poetry.git

echo "📦 6. Cloning Questionary - Interactive CLI prompts"
git clone --depth 1 https://github.com/tmbo/questionary.git

echo "📦 7. Cloning Python Prompt Toolkit - Advanced interactive features"
git clone --depth 1 https://github.com/prompt-toolkit/python-prompt-toolkit.git

echo "📦 8. Cloning Textual - TUI framework for complex interfaces"
git clone --depth 1 https://github.com/Textualize/textual.git

echo "📦 9. Cloning Black - Clean CLI with good error handling"
git clone --depth 1 https://github.com/psf/black.git

echo "📦 10. Cloning GitHub CLI - Excellent UX patterns"
git clone --depth 1 https://github.com/cli/cli.git gh-cli

# Health/science specific CLIs
echo "📦 11. Cloning SciKit-Learn - Scientific Python CLI patterns"
git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git

echo "📦 12. Cloning DVC - Data science CLI with progress tracking"
git clone --depth 1 https://github.com/iterative/dvc.git

cd ..

echo "✅ Reference repositories cloned successfully!"
echo ""
echo "Key patterns to study:"
echo "  - typer: Type hints → CLI args automatically"
echo "  - rich: Beautiful tables, progress bars, panels"
echo "  - httpie: Intuitive command structure"
echo "  - poetry: Complex subcommand organization"
echo "  - questionary: Interactive prompts done right"
echo "  - gh-cli: Excellent error messages and help"
echo "  - dvc: Progress tracking for long operations"