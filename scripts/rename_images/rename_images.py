#!/usr/bin/env python3

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

# Define supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

def is_image(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS

def generate_unique_filename(prefix='image', extension='png'):
    unique_id = uuid.uuid4().hex
    return f"{prefix}_{unique_id}{extension}"

def backup_images(root_dir, backup_dir, dry_run=True):
    """
    Copies all image files from root_dir to backup_dir, preserving directory structure.
    """
    path = Path(root_dir)
    backup_path = Path(backup_dir)
    renamed_files = []

    if dry_run:
        print(f"[Dry Run] Would create backup directory: {backup_path}")
    else:
        if not backup_path.exists():
            try:
                backup_path.mkdir(parents=True, exist_ok=True)
                print(f"Created backup directory: {backup_path}")
            except Exception as e:
                print(f"Error creating backup directory {backup_path}: {e}")
                return renamed_files

    for file_path in path.rglob('*'):
        if file_path.is_file() and is_image(file_path):
            # Determine the relative path to maintain directory structure
            relative_path = file_path.relative_to(path)
            backup_file_path = backup_path / relative_path

            if dry_run:
                print(f"[Dry Run] Would backup: {file_path} -> {backup_file_path}")
            else:
                # Ensure the parent directory exists in backup
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(file_path, backup_file_path)
                    print(f"Backed up: {file_path} -> {backup_file_path}")
                    renamed_files.append((file_path, backup_file_path))
                except Exception as e:
                    print(f"Error backing up {file_path}: {e}")

    return renamed_files

def rename_images(root_dir, prefix='image', dry_run=True, override_extension=None):
    """
    Renames all image files in root_dir and its subdirectories.
    """
    path = Path(root_dir)
    renamed_files = []

    for file_path in path.rglob('*'):
        if file_path.is_file() and is_image(file_path):
            original_extension = file_path.suffix.lower()
            extension = override_extension if override_extension else original_extension
            if override_extension:
                extension = f".{override_extension.lower().lstrip('.')}"
            
            new_filename = generate_unique_filename(prefix, extension)
            new_file_path = file_path.with_name(new_filename)
            
            # Check for name collision
            if new_file_path.exists():
                print(f"Skipping {file_path}: target {new_file_path} already exists.")
                continue
            
            if dry_run:
                print(f"[Dry Run] Would rename: {file_path} -> {new_file_path}")
            else:
                try:
                    file_path.rename(new_file_path)
                    print(f"Renamed: {file_path} -> {new_file_path}")
                    renamed_files.append((file_path, new_file_path))
                except Exception as e:
                    print(f"Error renaming {file_path}: {e}")
    
    return renamed_files

def main():
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Backup and batch rename image files in a directory and its subdirectories."
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Path to the root directory containing images to backup and rename.'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='image',
        help='Prefix for the new file names.'
    )
    parser.add_argument(
        '--backup-dir',
        type=str,
        default='backup_images',
        help='Directory where backups will be stored.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making any changes.'
    )
    parser.add_argument(
        '--override-extension',
        type=str,
        default=None,
        help='Override the file extension for renamed files (e.g., png, jpg).'
    )
    
    args = parser.parse_args()

    root_dir = args.directory
    backup_dir = args.backup_dir
    prefix = args.prefix
    dry_run = args.dry_run
    override_extension = args.override_extension

    # Step 1: Backup Original Images
    print("\n=== Starting Backup Process ===")
    backup_images(root_dir, backup_dir, dry_run=dry_run)
    
    # Step 2: Rename Original Images
    print("\n=== Starting Renaming Process ===")
    rename_images(root_dir, prefix=prefix, dry_run=dry_run, override_extension=override_extension)

    print("\n=== Operation Completed ===")

if __name__ == "__main__":
    main()
