#!/usr/bin/env python3
"""Fix common type annotation errors in the codebase."""

import re
from pathlib import Path


def fix_file(file_path: Path) -> bool:
    """Fix type annotations in a single file."""
    try:
        content = file_path.read_text()
        original_content = content

        # Track what imports we need to add
        needs_any = False
        needs_ndarray = False
        needs_callable = False

        # Fix dict without type parameters
        if re.search(r'\bdict\s*[|,)\]]', content):
            content = re.sub(r'\b(dict)(\s*[|,)\]])', r'dict[str, Any]\2', content)
            needs_any = True

        # Fix list without type parameters (be more careful here)
        if re.search(r':\s*list\s*[|,)\]]', content):
            content = re.sub(r'(:\s*)(list)(\s*[|,)\]])', r'\1list[Any]\3', content)
            needs_any = True

        # Fix ndarray without type parameters
        if re.search(r'\bndarray\s*[|,)\]]', content):
            content = re.sub(r'\b(ndarray)(\s*[|,)\]])', r'NDArray[np.float32]\2', content)
            needs_ndarray = True

        # Fix Callable without type parameters
        if re.search(r'\bCallable\s*[|,)\]]', content):
            content = re.sub(r'\b(Callable)(\s*[|,)\]])', r'Callable[..., Any]\2', content)
            needs_callable = True
            needs_any = True

        # Add imports if needed
        if needs_any or needs_ndarray or needs_callable:
            # Find the right place to add imports (after existing imports)
            import_lines = []

            # Check what's already imported
            has_any = 'from typing import' in content and 'Any' in content
            has_ndarray = 'from numpy.typing import NDArray' in content

            if needs_any and not has_any:
                # Add Any to existing typing import or create new one
                if 'from typing import' in content:
                    content = re.sub(
                        r'(from typing import )([^\\n]+)',
                        lambda m: m.group(1) + add_to_import_list(m.group(2), 'Any'),
                        content,
                        count=1
                    )
                else:
                    # Find where to insert the import
                    lines = content.split('\n')
                    insert_pos = find_import_position(lines)
                    lines.insert(insert_pos, 'from typing import Any')
                    content = '\n'.join(lines)

            if needs_ndarray and not has_ndarray:
                # Add numpy.typing import
                lines = content.split('\n')
                insert_pos = find_import_position(lines)

                # Check if numpy is already imported
                if 'import numpy as np' in content:
                    # Add after numpy import
                    for i, line in enumerate(lines):
                        if 'import numpy as np' in line:
                            lines.insert(i + 1, 'from numpy.typing import NDArray')
                            break
                else:
                    lines.insert(insert_pos, 'import numpy as np')
                    lines.insert(insert_pos + 1, 'from numpy.typing import NDArray')

                content = '\n'.join(lines)

        # Only write if we made changes
        if content != original_content:
            file_path.write_text(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def add_to_import_list(import_str: str, new_import: str) -> str:
    """Add an import to an existing import list."""
    imports = [imp.strip() for imp in import_str.split(',')]
    if new_import not in imports:
        imports.append(new_import)
    return ', '.join(sorted(imports))

def find_import_position(lines: list[str]) -> int:
    """Find the best position to insert an import."""
    # After docstring and other imports
    in_docstring = False
    last_import = 0

    for i, line in enumerate(lines):
        if line.strip().startswith('"""'):
            in_docstring = not in_docstring
        elif not in_docstring:
            if line.strip().startswith(('import ', 'from ')):
                last_import = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                # First non-import, non-empty, non-comment line
                if last_import > 0:
                    return last_import
                else:
                    return i

    return last_import

def main():
    """Fix type errors in all Python files."""
    src_dir = Path("src/big_mood_detector")

    files_to_fix = [
        "application/services/label_service.py",
        "infrastructure/writers/chunked_writer.py",
        "domain/services/risk_level_assessor.py",
        "infrastructure/settings/environment.py",
        "infrastructure/settings/config.py",
        "infrastructure/ml_models/pat_loader_direct.py",
        "domain/services/interpolation_strategies.py",
        "domain/services/activity_sequence_extractor.py",
        "infrastructure/logging/logger.py",
        "domain/services/temporal_feature_calculator.py",
        "domain/services/sparse_data_handler.py",
        "domain/services/pat_sequence_builder.py",
        "application/services/temporal_ensemble_orchestrator.py",
        "infrastructure/ml_models/pat_model.py",
        "infrastructure/ml_models/pat_depression_head.py",
        "infrastructure/fine_tuning/population_trainer.py",
        "infrastructure/fine_tuning/personal_calibrator.py",
        "application/use_cases/predict_mood_ensemble_use_case.py",
        "application/use_cases/process_health_data_use_case.py",
        "infrastructure/di/container.py",
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        full_path = src_dir / file_path
        if full_path.exists():
            if fix_file(full_path):
                print(f"Fixed: {file_path}")
                fixed_count += 1
        else:
            print(f"Not found: {file_path}")

    print(f"\nFixed {fixed_count} files")

    # Also handle the scipy/tensorflow import issues
    print("\nNote: You may need to install type stubs:")
    print("  pip install types-tensorflow")
    print("\nOr add to mypy.ini:")
    print("  [[tool.mypy.overrides]]")
    print("  module = \"scipy.*\"")
    print("  ignore_missing_imports = true")

if __name__ == "__main__":
    main()
