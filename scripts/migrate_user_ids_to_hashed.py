#!/usr/bin/env python3
"""
One-time migration script to hash existing plain user IDs in baseline directories.

This script scans the baseline directory for non-hashed user folders and 
migrates them to use SHA-256 hashed IDs for privacy compliance.
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path


def is_hashed_id(folder_name: str) -> bool:
    """Check if a folder name looks like a SHA-256 hash (64 hex chars)."""
    if len(folder_name) != 64:
        return False
    try:
        int(folder_name, 16)
        return True
    except ValueError:
        return False


def hash_user_id(user_id: str) -> str:
    """Hash user ID using same method as privacy module."""
    # Get salt from environment or use default
    salt = os.getenv("USER_ID_SALT", "big-mood-detector-default-salt")
    salted_id = f"{salt}:{user_id}"
    hash_object = hashlib.sha256(salted_id.encode("utf-8"))
    return hash_object.hexdigest()


def migrate_baseline_directories(base_path: Path, dry_run: bool = True) -> None:
    """
    Migrate plain user ID directories to hashed format.
    
    Args:
        base_path: Path to baseline storage directory
        dry_run: If True, only print what would be done without making changes
    """
    if not base_path.exists():
        print(f"Error: Baseline directory does not exist: {base_path}")
        sys.exit(1)
    
    print(f"Scanning baseline directory: {base_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    print("-" * 60)
    
    migrated_count = 0
    already_hashed_count = 0
    
    for user_dir in base_path.iterdir():
        if not user_dir.is_dir():
            continue
            
        folder_name = user_dir.name
        
        # Skip if already hashed
        if is_hashed_id(folder_name):
            already_hashed_count += 1
            continue
        
        # Hash the user ID
        hashed_id = hash_user_id(folder_name)
        new_path = base_path / hashed_id
        
        print(f"\nFound plain user ID: {folder_name}")
        print(f"  Hashed ID: {hashed_id}")
        print(f"  Old path: {user_dir}")
        print(f"  New path: {new_path}")
        
        if not dry_run:
            # Check if target already exists
            if new_path.exists():
                print(f"  WARNING: Target path already exists! Skipping.")
                continue
            
            # Move the directory
            try:
                shutil.move(str(user_dir), str(new_path))
                print(f"  ✓ Migrated successfully")
                migrated_count += 1
            except Exception as e:
                print(f"  ✗ Error during migration: {e}")
        else:
            migrated_count += 1
    
    print("\n" + "-" * 60)
    print(f"Summary:")
    print(f"  Already hashed: {already_hashed_count}")
    print(f"  To migrate: {migrated_count}")
    
    if dry_run and migrated_count > 0:
        print("\nTo perform the actual migration, run with --execute flag")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate plain user IDs to hashed format for privacy compliance"
    )
    parser.add_argument(
        "baseline_dir",
        type=Path,
        help="Path to baseline storage directory (e.g., data/baselines)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default is dry-run)"
    )
    
    args = parser.parse_args()
    
    # Add src to Python path for imports
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    migrate_baseline_directories(args.baseline_dir, dry_run=not args.execute)