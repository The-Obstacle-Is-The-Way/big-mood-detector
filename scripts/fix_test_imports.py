#!/usr/bin/env python3
"""Fix test imports to move them inside test methods for better coverage."""

import re
import sys
from pathlib import Path
from typing import List, Tuple

def find_big_mood_imports(content: str) -> List[Tuple[int, str]]:
    """Find all lines that import from big_mood_detector at the top level."""
    lines = content.split('\n')
    imports = []
    in_class = False
    in_function = False
    indent_level = 0
    
    for i, line in enumerate(lines):
        # Track if we're inside a class or function
        stripped = line.strip()
        if stripped.startswith('class '):
            in_class = True
            indent_level = len(line) - len(line.lstrip())
        elif stripped.startswith('def ') and not in_class:
            in_function = True
            indent_level = len(line) - len(line.lstrip())
        elif line and len(line) - len(line.lstrip()) <= indent_level:
            if in_function:
                in_function = False
            elif in_class:
                in_class = False
        
        # Only capture top-level imports (not inside class or function)
        if not in_class and not in_function and line.strip():
            if line.strip().startswith('from big_mood_detector'):
                imports.append((i, line))
            elif line.strip().startswith('import big_mood_detector'):
                imports.append((i, line))
    
    return imports

def extract_import_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Extract a complete import block including multi-line imports."""
    import_lines = []
    i = start_idx
    
    while i < len(lines):
        line = lines[i]
        import_lines.append(line)
        
        # Check if this is a multi-line import
        if '(' in line and ')' not in line:
            # Continue until we find the closing parenthesis
            i += 1
            while i < len(lines) and ')' not in lines[i]:
                import_lines.append(lines[i])
                i += 1
            if i < len(lines):
                import_lines.append(lines[i])
        
        i += 1
        break
    
    return '\n'.join(import_lines), i - start_idx

def find_usage_locations(content: str, imported_names: List[str]) -> List[Tuple[str, int]]:
    """Find where imported names are used in test methods."""
    lines = content.split('\n')
    usages = []
    current_method = None
    
    for i, line in enumerate(lines):
        # Track current method
        if line.strip().startswith('def test_') or line.strip().startswith('def test'):
            current_method = line.strip().split('(')[0].replace('def ', '')
        
        # Check for usage of imported names
        if current_method:
            for name in imported_names:
                if name in line and not line.strip().startswith('from') and not line.strip().startswith('import'):
                    usages.append((current_method, i))
                    break
    
    return usages

def extract_imported_names(import_statement: str) -> List[str]:
    """Extract the names being imported from an import statement."""
    names = []
    
    # Handle "from X import Y" style
    if 'from' in import_statement and 'import' in import_statement:
        # Extract everything after 'import'
        parts = import_statement.split('import', 1)[1]
        
        # Handle multi-line imports
        if '(' in parts:
            parts = parts.replace('(', '').replace(')', '')
        
        # Split by comma and clean up
        for part in parts.split(','):
            name = part.strip()
            if name:
                # Handle "X as Y" - we want X
                if ' as ' in name:
                    name = name.split(' as ')[0].strip()
                names.append(name)
    
    return names

def process_file(file_path: Path) -> bool:
    """Process a single test file to move imports inside methods."""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Find top-level big_mood_detector imports
        imports = find_big_mood_imports(content)
        if not imports:
            return False
        
        print(f"\nProcessing {file_path}")
        print(f"Found {len(imports)} top-level imports from big_mood_detector")
        
        lines = content.split('\n')
        
        # Collect all imports and their names
        all_imports = []
        for idx, import_line in imports:
            import_block, block_size = extract_import_block(lines, idx)
            imported_names = extract_imported_names(import_block)
            all_imports.append((idx, import_block, imported_names, block_size))
        
        # Remove imports from top of file (in reverse order to maintain indices)
        for idx, import_block, _, block_size in reversed(all_imports):
            # Remove the import lines
            for i in range(block_size):
                lines[idx + i] = ''
        
        # Find where each imported name is used and add imports there
        methods_with_imports = set()
        
        for _, import_block, imported_names, _ in all_imports:
            # Find all test methods
            for i, line in enumerate(lines):
                if line.strip().startswith('def test_') or (line.strip().startswith('def ') and 'test' in line):
                    method_name = line.strip().split('(')[0].replace('def ', '')
                    
                    # Check if this method uses any of the imported names
                    method_uses_import = False
                    method_start = i
                    method_indent = len(line) - len(line.lstrip())
                    
                    # Find method body
                    j = i + 1
                    while j < len(lines):
                        method_line = lines[j]
                        
                        # Check if we're still in the method
                        if method_line.strip() and len(method_line) - len(method_line.lstrip()) <= method_indent:
                            break
                        
                        # Check for usage
                        for name in imported_names:
                            if name in method_line and not method_line.strip().startswith('#'):
                                method_uses_import = True
                                break
                        
                        if method_uses_import:
                            break
                        
                        j += 1
                    
                    if method_uses_import and method_name not in methods_with_imports:
                        methods_with_imports.add(method_name)
                        
                        # Find the docstring end
                        doc_end = i + 1
                        in_docstring = False
                        docstring_delim = None
                        
                        for k in range(i + 1, min(i + 20, len(lines))):
                            line_content = lines[k].strip()
                            if line_content.startswith('"""') or line_content.startswith("'''"):
                                if not in_docstring:
                                    in_docstring = True
                                    docstring_delim = '"""' if line_content.startswith('"""') else "'''"
                                    if line_content.endswith(docstring_delim) and len(line_content) > 3:
                                        # Single line docstring
                                        doc_end = k + 1
                                        break
                                elif docstring_delim in line_content:
                                    doc_end = k + 1
                                    break
                            elif in_docstring and docstring_delim in line_content:
                                doc_end = k + 1
                                break
                            elif not in_docstring and line_content and not line_content.startswith('#'):
                                doc_end = k
                                break
                        
                        # Insert import after docstring
                        indent = ' ' * (method_indent + 4)
                        import_lines = import_block.split('\n')
                        formatted_import = []
                        
                        for imp_line in import_lines:
                            if imp_line.strip():
                                formatted_import.append(indent + imp_line.strip())
                        
                        # Add import and blank line
                        lines[doc_end] = formatted_import[0] + '\n' + indent + '\n' + lines[doc_end]
                        for k, imp_line in enumerate(formatted_import[1:]):
                            lines[doc_end] = lines[doc_end].replace('\n' + lines[doc_end].split('\n')[-1], '\n' + imp_line + '\n' + lines[doc_end].split('\n')[-1])
        
        # Clean up multiple blank lines
        new_content = '\n'.join(lines)
        new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
        
        # Write back if changed
        if new_content != original_content:
            file_path.write_text(new_content)
            print(f"✓ Fixed imports in {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests"
    
    # Find all test files
    test_files = list(test_dir.rglob("*test*.py"))
    
    # Filter out reference_repos
    test_files = [f for f in test_files if "reference_repos" not in str(f)]
    
    print(f"Found {len(test_files)} test files to check")
    
    fixed_count = 0
    for test_file in sorted(test_files):
        if process_file(test_file):
            fixed_count += 1
    
    print(f"\n✓ Fixed {fixed_count} files")

if __name__ == "__main__":
    main()