#!/usr/bin/env python3
"""Fix remaining type annotation errors."""

import re
from pathlib import Path


def fix_ndarray_references(content: str) -> str:
    """Fix np.NDArray references to just NDArray."""
    # Replace np.NDArray with NDArray
    content = re.sub(r'\bnp\.NDArray\b', 'NDArray', content)
    return content

def fix_generic_types(content: str) -> tuple[str, set[str]]:
    """Fix missing type parameters for generic types."""
    imports_needed = set()

    # Fix standalone dict
    content = re.sub(r':\s*dict\s*([=,)\]])', r': dict[str, Any]\1', content)
    if 'dict[str, Any]' in content:
        imports_needed.add('Any')

    # Fix standalone list
    content = re.sub(r':\s*list\s*([=,)\]])', r': list[Any]\1', content)
    if 'list[Any]' in content:
        imports_needed.add('Any')

    # Fix standalone ndarray
    content = re.sub(r':\s*ndarray\s*([=,)\]])', r': NDArray[np.float32]\1', content)
    if 'NDArray[np.float32]' in content:
        imports_needed.add('NDArray')

    # Fix standalone Callable
    content = re.sub(r':\s*Callable\s*([=,)\]])', r': Callable[..., Any]\1', content)
    if 'Callable[..., Any]' in content:
        imports_needed.add('Callable')
        imports_needed.add('Any')

    return content, imports_needed

def ensure_imports(content: str, imports_needed: set[str]) -> str:
    """Ensure necessary imports are present."""
    lines = content.split('\n')

    # Find import section
    import_section_start = -1
    import_section_end = -1
    in_docstring = False

    for i, line in enumerate(lines):
        if '"""' in line:
            in_docstring = not in_docstring
        if not in_docstring and line.strip().startswith(('import ', 'from ')):
            if import_section_start == -1:
                import_section_start = i
            import_section_end = i + 1

    if import_section_start == -1:
        # No imports found, add after module docstring
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('"""'):
                import_section_start = i
                import_section_end = i
                break

    # Check what's already imported
    content_str = '\n'.join(lines)

    new_imports = []

    if 'Any' in imports_needed and 'from typing import' in content_str and 'Any' not in content_str:
        # Add Any to existing typing import
        for i, line in enumerate(lines):
            if 'from typing import' in line:
                # Parse the imports
                match = re.match(r'from typing import (.+)', line)
                if match:
                    imports = [imp.strip() for imp in match.group(1).split(',')]
                    if 'Any' not in imports:
                        imports.append('Any')
                        lines[i] = f"from typing import {', '.join(sorted(imports))}"
                break
    elif 'Any' in imports_needed and 'Any' not in content_str:
        new_imports.append('from typing import Any')

    if 'Callable' in imports_needed and 'Callable' not in content_str:
        if 'from typing import' in content_str:
            for i, line in enumerate(lines):
                if 'from typing import' in line:
                    match = re.match(r'from typing import (.+)', line)
                    if match:
                        imports = [imp.strip() for imp in match.group(1).split(',')]
                        if 'Callable' not in imports:
                            imports.append('Callable')
                            lines[i] = f"from typing import {', '.join(sorted(imports))}"
                    break
        else:
            new_imports.append('from typing import Callable')

    if 'NDArray' in imports_needed and 'from numpy.typing import NDArray' not in content_str:
        # Check if numpy is imported
        has_numpy = 'import numpy as np' in content_str
        if has_numpy:
            # Find numpy import and add NDArray import after it
            for i, line in enumerate(lines):
                if 'import numpy as np' in line:
                    lines.insert(i + 1, 'from numpy.typing import NDArray')
                    break
        else:
            new_imports.append('import numpy as np')
            new_imports.append('from numpy.typing import NDArray')

    # Insert new imports
    if new_imports:
        for imp in reversed(new_imports):
            lines.insert(import_section_end, imp)

    return '\n'.join(lines)

def fix_file(file_path: Path) -> bool:
    """Fix type errors in a file."""
    try:
        content = file_path.read_text()
        original = content

        # Fix np.NDArray references
        content = fix_ndarray_references(content)

        # Fix generic types
        content, imports_needed = fix_generic_types(content)

        # Ensure imports
        if imports_needed:
            content = ensure_imports(content, imports_needed)

        if content != original:
            file_path.write_text(content)
            print(f"Fixed: {file_path.name}")
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix remaining type errors."""
    src_dir = Path("src/big_mood_detector")

    # Get all Python files
    py_files = list(src_dir.rglob("*.py"))

    fixed_count = 0
    for file_path in py_files:
        if fix_file(file_path):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")

    # Also need to handle scipy/tensorflow in mypy.ini
    print("\nNote: scipy and tensorflow imports are already configured in mypy.ini to ignore missing imports")

if __name__ == "__main__":
    main()
