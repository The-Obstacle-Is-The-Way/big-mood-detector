#!/usr/bin/env python3
"""Fix type annotation errors for mypy compliance."""

import re
from pathlib import Path


def fix_api_routes():
    """Fix missing return type annotations in API routes."""
    api_files = [
        "src/big_mood_detector/interfaces/api/clinical_routes.py",
        "src/big_mood_detector/interfaces/api/main.py"
    ]

    for file_path in api_files:
        path = Path(file_path)
        if not path.exists():
            continue

        content = path.read_text()

        # Fix async def functions without return types
        # Pattern: async def function_name(...):
        # Replace with: async def function_name(...) -> JSONResponse:

        # For routes that return dict/response
        content = re.sub(
            r'(async def \w+\([^)]*\))(\s*:)',
            r'\1 -> JSONResponse\2',
            content
        )

        # For health_check and root that return dict
        content = content.replace(
            'async def health_check() -> JSONResponse:',
            'async def health_check() -> JSONResponse:'
        )
        content = content.replace(
            'async def root() -> JSONResponse:',
            'async def root() -> dict[str, Any]:'
        )

        # Add imports if needed
        if 'JSONResponse' in content and 'from fastapi.responses import JSONResponse' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from fastapi'):
                    lines.insert(i + 1, 'from fastapi.responses import JSONResponse')
                    break
            content = '\n'.join(lines)

        if 'dict[str, Any]' in content and 'from typing import Any' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('from typing') or line.startswith('import'):
                    if 'from typing' in line:
                        # Add Any to existing typing import
                        if 'Any' not in line:
                            content = content.replace(line, line.rstrip() + ', Any')
                    break
            else:
                # No typing import found, add one
                lines.insert(0, 'from typing import Any')
                content = '\n'.join(lines)

        path.write_text(content)
        print(f"Fixed {file_path}")


def fix_pat_loader():
    """Fix type annotations in PAT loader."""
    path = Path("src/big_mood_detector/infrastructure/ml_models/pat_loader_direct.py")
    if not path.exists():
        return

    content = path.read_text()

    # Fix weights annotation
    content = content.replace(
        'self.weights = {}',
        'self.weights: dict[str, Any] = {}'
    )

    # Add type annotations to functions
    replacements = [
        ('def scaled_dot_product_attention(query, key, value, mask=None):',
         'def scaled_dot_product_attention(query: Any, key: Any, value: Any, mask: Any = None) -> Any:'),
        ('def point_wise_feed_forward_network(d_model, dff):',
         'def point_wise_feed_forward_network(d_model: int, dff: int) -> Any:'),
        ('def positional_encoding(position, d_model):',
         'def positional_encoding(position: int, d_model: int) -> Any:'),
        ('def create_padding_mask(seq):',
         'def create_padding_mask(seq: Any) -> Any:'),
        ('def create_look_ahead_mask(size):',
         'def create_look_ahead_mask(size: int) -> Any:')
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    # Add Any import if needed
    if 'from typing import Any' not in content:
        lines = content.split('\n')
        for _i, line in enumerate(lines):
            if line.startswith('from typing') or line.startswith('import'):
                if 'from typing' in line and 'Any' not in line:
                    content = content.replace(line, line.rstrip() + ', Any')
                break

    path.write_text(content)
    print("Fixed pat_loader_direct.py")


def fix_ensemble_orchestrator():
    """Fix type issues in ensemble orchestrator."""
    path = Path("src/big_mood_detector/application/ensemble_orchestrator.py")
    if not path.exists():
        return

    content = path.read_text()

    # Fix the validation function annotation
    content = content.replace(
        'def _validate_input(features, records):',
        'def _validate_input(features: Any, records: Any) -> None:'
    )

    # Fix date type issues
    content = content.replace(
        'end_date=records[-1].end_date.date(),',
        'end_date=records[-1].end_date.date() if hasattr(records[-1].end_date, "date") else records[-1].end_date,'
    )

    path.write_text(content)
    print("Fixed ensemble_orchestrator.py")


def main():
    """Run all fixes."""
    print("Fixing type annotation errors...")

    fix_api_routes()
    fix_pat_loader()
    fix_ensemble_orchestrator()

    print("\nDone! Run 'make type-check' to verify.")


if __name__ == "__main__":
    main()
