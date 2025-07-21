#!/usr/bin/env python3
"""
Fix test imports by moving big_mood_detector imports inside test methods.
This helps ensure coverage starts before imports happen.
"""

import ast
import re
from pathlib import Path
from typing import List, Set, Tuple

class ImportFinder(ast.NodeVisitor):
    """Find imports from big_mood_detector and where they're used."""
    
    def __init__(self):
        self.big_mood_imports = []
        self.imported_names = {}
        self.name_usage = {}
        self.current_function = None
        
    def visit_ImportFrom(self, node):
        """Visit 'from X import Y' statements."""
        if node.module and node.module.startswith('big_mood_detector'):
            # Record the import
            import_info = {
                'lineno': node.lineno,
                'end_lineno': node.end_lineno or node.lineno,
                'module': node.module,
                'names': []
            }
            
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                import_info['names'].append(name)
                self.imported_names[name] = node.module
                
            self.big_mood_imports.append(import_info)
            
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Track current function."""
        old_function = self.current_function
        self.current_function = node.name
        
        # Track name usage in this function
        if node.name not in self.name_usage:
            self.name_usage[node.name] = set()
            
        self.generic_visit(node)
        self.current_function = old_function
        
    def visit_Name(self, node):
        """Track usage of imported names."""
        if self.current_function and node.id in self.imported_names:
            self.name_usage[self.current_function].add(node.id)
        self.generic_visit(node)


def process_file(file_path: Path) -> bool:
    """Process a single file to move imports."""
    try:
        content = file_path.read_text()
        
        # Skip if no big_mood_detector imports at top level
        if not re.search(r'^from big_mood_detector', content, re.MULTILINE):
            return False
            
        # Parse the file
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return False
            
        # Find imports and usage
        finder = ImportFinder()
        finder.visit(tree)
        
        if not finder.big_mood_imports:
            return False
            
        print(f"\nProcessing {file_path}")
        print(f"Found {len(finder.big_mood_imports)} imports from big_mood_detector")
        
        # Build a mapping of which functions need which imports
        function_imports = {}
        for func_name, used_names in finder.name_usage.items():
            if used_names:
                imports_needed = set()
                for name in used_names:
                    module = finder.imported_names.get(name)
                    if module:
                        imports_needed.add((module, name))
                if imports_needed:
                    function_imports[func_name] = imports_needed
        
        # Read file lines
        lines = content.split('\n')
        
        # Remove top-level imports (in reverse order)
        for import_info in reversed(finder.big_mood_imports):
            start = import_info['lineno'] - 1
            end = import_info['end_lineno']
            
            # Handle multi-line imports
            while start > 0 and lines[start - 1].strip().endswith('\\'):
                start -= 1
            
            # Check if it's a multiline import with parentheses
            if start > 0 and '(' in lines[start - 1] and 'import' in lines[start - 1]:
                start -= 1
                # Find the closing parenthesis
                while end < len(lines) and ')' not in lines[end - 1]:
                    end += 1
                    
            # Remove the lines
            for i in range(start, end):
                if i < len(lines):
                    lines[i] = ''
                    
        # Add imports inside functions
        modified = False
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for function definitions
            if line.strip().startswith('def ') and ('test' in line or line.strip().startswith('def test_')):
                func_match = re.match(r'\s*def\s+(\w+)', line)
                if func_match:
                    func_name = func_match.group(1)
                    
                    if func_name in function_imports:
                        modified = True
                        indent = len(line) - len(line.lstrip()) + 4
                        
                        # Find where to insert (after docstring if any)
                        insert_pos = i + 1
                        
                        # Skip docstring
                        if insert_pos < len(lines):
                            next_line = lines[insert_pos].strip()
                            if next_line.startswith('"""') or next_line.startswith("'''"):
                                # Find end of docstring
                                delim = '"""' if next_line.startswith('"""') else "'''"
                                if next_line.endswith(delim) and len(next_line) > 3:
                                    insert_pos += 1
                                else:
                                    insert_pos += 1
                                    while insert_pos < len(lines):
                                        if delim in lines[insert_pos]:
                                            insert_pos += 1
                                            break
                                        insert_pos += 1
                        
                        # Group imports by module
                        imports_by_module = {}
                        for module, name in function_imports[func_name]:
                            if module not in imports_by_module:
                                imports_by_module[module] = []
                            imports_by_module[module].append(name)
                        
                        # Create import statements
                        import_lines = []
                        for module, names in imports_by_module.items():
                            if len(names) == 1:
                                import_lines.append(f"{' ' * indent}from {module} import {names[0]}")
                            else:
                                import_lines.append(f"{' ' * indent}from {module} import (")
                                for name in sorted(names):
                                    import_lines.append(f"{' ' * (indent + 4)}{name},")
                                import_lines.append(f"{' ' * indent})")
                        
                        # Insert imports
                        if import_lines:
                            import_block = '\n'.join(import_lines)
                            
                            # Add extra newline after imports
                            import_block += '\n'
                            
                            # Insert at the position
                            lines[insert_pos] = import_block + '\n' + lines[insert_pos]
            
            i += 1
        
        if modified:
            # Clean up extra blank lines
            new_content = '\n'.join(lines)
            new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
            new_content = re.sub(r'^\n+', '', new_content)  # Remove leading newlines
            
            # Ensure file ends with newline
            if new_content and not new_content.endswith('\n'):
                new_content += '\n'
            
            file_path.write_text(new_content)
            print(f"✓ Fixed {file_path}")
            return True
            
        return False
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Find all test files
    test_files = []
    for pattern in ["*test*.py", "test_*.py"]:
        test_files.extend(tests_dir.rglob(pattern))
    
    # Remove duplicates and filter
    test_files = list(set(test_files))
    test_files = [f for f in test_files if "reference_repos" not in str(f)]
    test_files = [f for f in test_files if "__pycache__" not in str(f)]
    
    print(f"Found {len(test_files)} test files")
    
    fixed = 0
    for file_path in sorted(test_files):
        if process_file(file_path):
            fixed += 1
    
    print(f"\nTotal files fixed: {fixed}")


if __name__ == "__main__":
    main()