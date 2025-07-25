#!/usr/bin/env python3
"""Update all model path references in the codebase to use new structure."""

import re
from pathlib import Path

# Path mappings
PATH_MAPPINGS = {
    # Old pretrained paths → New pretrained paths
    'model_weights/pat/pretrained/PAT-L_29k_weights.h5': 'model_weights/pretrained/PAT-L_29k_weights.h5',
    'model_weights/pat/pretrained/PAT-M_29k_weights.h5': 'model_weights/pretrained/PAT-M_29k_weights.h5',
    'model_weights/pat/pretrained/PAT-S_29k_weights.h5': 'model_weights/pretrained/PAT-S_29k_weights.h5',
    
    # Old PyTorch paths → Production path (for the best model)
    'model_weights/pat/pytorch/pat_conv_l_simple_best.pth': 'model_weights/production/pat_conv_l_v0.5929.pth',
    
    # Generic patterns
    r'model_weights/pat/pretrained/PAT-(\w+)_29k_weights\.h5': r'model_weights/pretrained/PAT-\1_29k_weights.h5',
}

def update_file(file_path: Path) -> bool:
    """Update model paths in a single file."""
    try:
        content = file_path.read_text()
        original = content
        
        # Apply mappings
        for old_path, new_path in PATH_MAPPINGS.items():
            if old_path.startswith('r'):
                # Regex pattern
                content = re.sub(old_path[1:], new_path[1:], content)
            else:
                # Direct replacement
                content = content.replace(old_path, new_path)
        
        if content != original:
            file_path.write_text(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def find_files_with_model_paths():
    """Find all Python files that might reference model paths."""
    src_dir = Path("src")
    scripts_dir = Path("scripts")
    
    files_to_check = []
    for directory in [src_dir, scripts_dir]:
        if directory.exists():
            files_to_check.extend(directory.rglob("*.py"))
    
    return files_to_check

def main():
    """Update all model paths in the codebase."""
    print("🔍 Finding files with model path references...")
    
    files = find_files_with_model_paths()
    updated_count = 0
    
    for file_path in files:
        if update_file(file_path):
            print(f"  Updated: {file_path}")
            updated_count += 1
    
    print(f"\n✅ Updated {updated_count} files")
    
    # Also update key configuration files
    config_files = [
        Path("core/config.py"),
        Path("infrastructure/settings/config.py"),
        Path(".env.example"),
    ]
    
    for config_file in config_files:
        if config_file.exists():
            if update_file(config_file):
                print(f"  Updated config: {config_file}")

if __name__ == "__main__":
    main()